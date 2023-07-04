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


class DABlock(nn.Module):
    def __init__(
            self, fmap_size, n_head_channels, n_heads=1, n_groups=1,
            attn_drop=0.0, stride=1,
            offset_range_factor=2, use_pe=True
    ):
        '''
        :param fmap_size:
        :param n_head_channels: input channels
        :param n_heads: 1
        :param n_groups: 1
        :param attn_drop: 0.0
        :param stride: down_sample factor in offset generation ==1
        :param offset_range_factor: scale of offset, maximum offset
        :param use_pe: the use of relative position bias, True
        '''
        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.kv_h, self.kv_w = fmap_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.offset_range_factor = offset_range_factor

        kk = 3

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels), # kk=kernel_size, kk//2=padding
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            # the component in Deformable relative position bias
            self.rpe_table = nn.Parameter(
                torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
            )
            trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

        self.gamma = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg   # ∆p = θoffset(q)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor) # ∆p ← − s tanh (∆p)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device) # a uniform grid of points p generated as the ref

        if self.offset_range_factor >= 0:
            pos = offset + reference # p + ∆p
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg  # sampling function φ(·; ·)

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe: # Deformable relative position bias.
            # relative position bias encoding
            rpe_table = self.rpe_table
            rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

            q_grid = self._get_ref_points(H, W, B, dtype, device)

            displacement = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                               n_sample,
                                                                                               2).unsqueeze(1)).mul(
                0.5)

            attn_bias = F.grid_sample(
                input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                grid=displacement[..., (1, 0)],
                mode='bilinear', align_corners=True
            )  # B * g, h_g, HW, Ns

            attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)

            attn = attn + attn_bias

        attn = F.softmax(attn, dim=2) # attention mask
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.gamma * out + x

        return out, attn

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