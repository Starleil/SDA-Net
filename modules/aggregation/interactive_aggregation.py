# Codes of 'conv_2nV1' and 'conv_3nV1' will be available soon.
import torch
from torch import nn
import torch.nn.functional as F

# from utils.tensor_ops import cus_sample

def cus_sample(feat, **kwargs):
    """
    :param feat:
    :param kwargs:
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)

class IAM(nn.Module):
    def __init__(self, iC_list, oC_list, operations):
        '''

        :param iC_list: default(64, 256, 512, 1024, 2048)
        :param oC_list: default(64, 64, 64, 64, 64)
        :param operations: decide the upsample or downsample in each conv_nV1
                           default:              ([True,True,True],
                                                  [True,True,True,True,True,True],
                                                  [True,True,True,True,True,True],
                                                  [True,True,True,True,True,True],
                                                  [True,True,True]
                                                  )
        '''
        super(IAM, self).__init__()
        ic0, ic1, ic2, ic3, ic4 = iC_list
        oc0, oc1, oc2, oc3, oc4 = oC_list
        op0, op1, op2, op3, op4 = operations
        self.conv0 = conv_2nV1(in_hc=ic0, in_lc=ic1, out_c=oc0, main=0, operations=op0)
        self.conv1 = conv_3nV1(in_hc=ic0, in_mc=ic1, in_lc=ic2, out_c=oc1, operations=op1)
        self.conv2 = conv_3nV1(in_hc=ic1, in_mc=ic2, in_lc=ic3, out_c=oc2, operations=op2)
        self.conv3 = conv_3nV1(in_hc=ic2, in_mc=ic3, in_lc=ic4, out_c=oc3, operations=op3)
        self.conv4 = conv_2nV1(in_hc=ic3, in_lc=ic4, out_c=oc4, main=1, operations=op4)

    def forward(self, *xs):
        # default fmapsize:
        # in_data_2 (256, 256), in_data_4 (128, 128), in_data_8 (64, 64), in_data_16 (32,32), in_data_32 (16,16)
        out_xs = []
        out_xs.append(self.conv0(xs[0], xs[1]))
        out_xs.append(self.conv1(xs[0], xs[1], xs[2]))
        out_xs.append(self.conv2(xs[1], xs[2], xs[3])) ######
        out_xs.append(self.conv3(xs[2], xs[3], xs[4]))
        out_xs.append(self.conv4(xs[3], xs[4]))

        return out_xs
