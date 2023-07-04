# Codes of 'DABlock' will be available soon.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

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
