import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from modules.attentions.deformable_attention import DABlock
from modules.aggregation.feature_pyramid import FP
from modules.aggregation.interactive_aggregation import IAM
from utils.config import cam_sigma, cam_w

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def refine_cams(cam_original, image_shape, using_sigmoid=True):
    if image_shape[0] != cam_original.size(2) or image_shape[1] != cam_original.size(3):
        cam_original = F.interpolate(
            cam_original, image_shape, mode="bilinear", align_corners=True
        )
    B, C, H, W = cam_original.size()
    cams = []
    for idx in range(C):
        cam = cam_original[:, idx, :, :]
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        norm = cam_max - cam_min
        norm[norm == 0] = 1e-5
        cam = (cam - cam_min) / norm
        cam = cam.view(B, H, W).unsqueeze(1)
        cams.append(cam)
    cams = torch.cat(cams, dim=1)
    if using_sigmoid:
        cams = torch.sigmoid(cam_w * (cams - cam_sigma))
    return cams


class DAModule(nn.Module):
    def __init__(self, in_channels, fmap_size, out_channels, norm_layer=None):
        super(DAModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conva = nn.Sequential(conv3x3(in_channels, out_channels),
                                   norm_layer(out_channels))
        self.da = DABlock(fmap_size, out_channels)
        self.convb = nn.Sequential(conv3x3(out_channels, out_channels),
                                   norm_layer(out_channels))

    def forward(self, x):
        output = self.conva(x)
        output, att_mask = self.da(output)
        output = self.convb(output)
        return output, att_mask


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias), self.weight

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, stride_list=[1, 2, 2, 2], use_maxpooling=True, use_aggregation=True, use_cam=True, dilations=None,
                 norm_layer=None, num_classes=2, zero_init_residual=False, groups=1, width_per_group=64,
                 operations=([True,True,True],
                  [True,True,True,True,True,True],
                  [True,True,False,False,True,False],
                  [False,False,False,False,False,False],
                  [False,False,False])
                 ):
        '''

        :param block:
        :param layers:
        :param stride_list:
        :param use_maxpooling:
        :param use_aggregation: adopt the aggregation branch. defaults: True or False
        :param dilations:
        :param norm_layer:
        :param num_classes:
        :param zero_init_residual:
        :param groups:
        :param width_per_group:
        :param operations:  upsample or downsample operations in the aggregation branch. defaults: all Ture -> vanilla ResNet
        '''
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_maxpooling = use_maxpooling

        self.use_aggregation = use_aggregation
        self.use_cam = use_cam

        self.inplanes = 64
        if dilations is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            dilations = [1, 1, 1, 1]
        # print(dilations)
        if len(dilations) != 4:
            raise ValueError("dilations should be None "
                             "or a 4-element tuple, got {}".format(dilations))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.use_maxpooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride_list[0],
                                       dilate=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride_list[1],
                                       dilate=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_list[2],
                                       dilate=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride_list[3],
                                       dilate=dilations[3])

        num_feautures = 512 * block.expansion
        num_inner = num_feautures // 4

        # deformable attention
        fmap_size = 64
        self.da_layer = DAModule(num_feautures, fmap_size, num_inner, norm_layer)

        self.conv = nn.Sequential(
            conv3x3(num_feautures + num_inner, 512),
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.cls_fc = Linear(512, num_classes, bias=False)

        if self.use_aggregation:
            # interactive aggregation module
            self.iam_layer = IAM(iC_list=(64, 256, 512, 1024, 2048), oC_list=(512, 512, 512, 512, 512), operations=operations)
            # feature pyramid module, refine type = ['conv', 'non_local']
            self.fpm_layer = FP(in_channels=512,num_levels=5,refine_level=2,refine_type='conv')

            self.aux_fc = Linear(512, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilate, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilate,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        im_h = x.size(2)
        im_w = x.size(3)
        x = self.conv1(x)
        x = self.bn1(x)
        x_top = self.relu(x)

        if self.use_maxpooling:
            e1 = self.layer1(self.maxpool(x_top))
        else:
            e1 = self.layer1(x_top)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        e4_Da, att_mask = self.da_layer(e4)
        e4_cat = self.conv(torch.cat((e4, e4_Da), dim=1))

        outs = self.globalavgpool(e4_cat)
        outs = torch.flatten(outs, 1)

        out_classes, fc_w = self.cls_fc(outs)
        cam_classes = self.relu(
            F.conv2d(e4_cat, fc_w.detach().unsqueeze(2).unsqueeze(3), bias=None, stride=1, padding=0))

        cam_classes_refined = refine_cams(cam_classes, (im_h, im_w), using_sigmoid=True)

        if self.use_aggregation:
            am0, am1, am2, am3, am4 = self.iam_layer(x_top, e1, e2, e3, e4)
            pfs = self.fpm_layer((am0, am1, am2, am3, am4))
            B_pfs, C_pfs, H_pfs, W_pfs = pfs.size()
            pfs = pfs.reshape(B_pfs, C_pfs, H_pfs*W_pfs)

            aux_out = torch.einsum('b m n, b c n -> b c m', att_mask, pfs)
            aux_out = aux_out.reshape(B_pfs, C_pfs, H_pfs, W_pfs)
            aux_out = self.globalavgpool(aux_out)
            aux_out = torch.flatten(aux_out, 1)
            aux_classes, aux_w = self.aux_fc(aux_out)

            if self.use_cam:
                return out_classes, cam_classes_refined, cam_classes, aux_classes
            else:
                return out_classes, aux_classes

        else:
            if self.use_cam:
                return out_classes, cam_classes_refined, cam_classes
            else:
                return out_classes


def _resnet(arch, block, layers, stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, progress, **kwargs):
    model = ResNet(block, layers, stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, **kwargs)
    return model


def resnet18(stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer,
                   progress, **kwargs)


def resnet34(stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer,
                   progress, **kwargs)


def resnet50(stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer,
                   progress, **kwargs)


def resnet101(stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer,
                   progress, **kwargs)


def resnet152(stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer,
                   progress, **kwargs)


def resnext50_32x4d(stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer,
                   progress, **kwargs)


def resnext101_32x8d(stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], stride_list, use_maxpooling, use_aggregation, use_cam, dilations, norm_layer,
                   progress, **kwargs)