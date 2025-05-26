import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv,  GhostConv

__all__ = (
    "GhostSPPFCSPC",
    "GCA",
    "MBAD",
    "SPDConv",
    "CSPOmniKernel",
)

class GhostSPPFCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.375, k=5):
        super(GhostSPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, g=1)
        self.cv2 = Conv(c1, c_, 1, 1, g=1)
        self.cv3 = GhostConv(c_, c_, 3, 1, g=1)
        self.cv4 = Conv(c_, c_, 1, 1, g=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1, g=1)
        self.cv6 = GhostConv(c_, c_, 3, 1, g=1)
        self.cv7 = Conv(2 * c_, c2, 1, 1, g=1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class GCA(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(GCA, self).__init__()
        self.spatial = nn.Sequential(
            Conv(in_channels, in_channels // 2, k=3, s=2),
            nn.ReLU(),
            Conv(in_channels // 2, 1, k=1)
        )
        self.conv_fuse = Conv(out_channels, out_channels, k=1)
        self.channel = nn.ReLU()

    def forward(self, x):

        input1, input2 = x[0], x[1]
        b, c, h, w = input1.shape
        B, C, H, W = input2.shape
        a = C // c

        att_map = F.softmax(self.spatial(input1).view(b, 1, 1, -1), dim=-1)  # (B, 1, H*W)
        input2_reshaped = input2.view(b, c * a, -1, 1)  # (B, 2C, H*W, 1
        x_out = torch.matmul(att_map, input2_reshaped)  # (B, 1, H*W) * (B, 2C, H*W, 1)
        x_out = self.channel(self.conv_fuse.forward_fuse(x_out))  # Reshape back to (B, C2, H, W)
        x_out = x_out + input2

        return x_out


class CropLayer(nn.Module):
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0 and self.cols_to_crop == 0:
            return input
        elif self.rows_to_crop > 0 and self.cols_to_crop == 0:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, :]
        elif self.rows_to_crop == 0 and self.cols_to_crop > 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class Asymmetric_DConv(nn.Module):
    def __init__(self, in_c, out_c, padding, kernel_size, stride, dilations, padding_mode='zeros', use_affine=True):
        super(Asymmetric_DConv, self).__init__()
        center_offset_from_origin_border = padding - kernel_size // 2
        ver_pad_or_crop = (0, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, 0)
        if center_offset_from_origin_border >= 0:
            self.ver_conv_crop_layer = nn.Identity()
            ver_conv_padding = ver_pad_or_crop
            self.hor_conv_crop_layer = nn.Identity()
            hor_conv_padding = hor_pad_or_crop
        else:
            self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
            ver_conv_padding = (0, 0)
            self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
            hor_conv_padding = (0, 0)
        self.ver_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(kernel_size, 1),
                                  stride=stride,
                                  padding=(dilations, 0), dilation=(dilations, 1), groups=1, bias=False,
                                  padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, kernel_size),
                                  stride=stride,
                                  padding=(0, dilations), dilation=(1, dilations), groups=1, bias=False,
                                  padding_mode=padding_mode)
        self.ver_bn = nn.BatchNorm2d(num_features=out_c, affine=use_affine)
        self.hor_bn = nn.BatchNorm2d(num_features=out_c, affine=use_affine)

    def forward(self, x):
        vertical_outputs = self.ver_conv_crop_layer(x)
        vertical_outputs = self.ver_conv(vertical_outputs)
        vertical_outputs = self.ver_bn(vertical_outputs)
        horizontal_outputs = self.hor_conv_crop_layer(x)
        horizontal_outputs = self.hor_conv(horizontal_outputs)
        horizontal_outputs = self.hor_bn(horizontal_outputs)
        return horizontal_outputs + vertical_outputs

class MBAD(nn.Module):
    def __init__(self, in_c, out_c,kernel_size=3):
        super(MBAD, self).__init__()
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, stride=1, padding=1, groups=1,dilation=1, bias=False),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(True))
        self.stage2_drate_1 = Asymmetric_DConv(in_c=out_c // 2, out_c=out_c // 8, padding=1, stride=1, kernel_size=kernel_size, dilations=1)
        self.stage2_drate_2 = Asymmetric_DConv(in_c=out_c // 2, out_c=out_c // 8, padding=2, stride=1, kernel_size=kernel_size, dilations=2)
        self.stage2_drate_4 = Asymmetric_DConv(in_c=out_c // 2, out_c=out_c // 8, padding=4, stride=1, kernel_size=kernel_size, dilations=4)
        self.stage2_drate_8 = Asymmetric_DConv(in_c=out_c // 2, out_c=out_c // 8, padding=8, stride=1, kernel_size=kernel_size, dilations=8)
        self.identity_conv = Conv(in_c, out_c, 1, 1)

    def forward(self, x):
        stage1_out = self.stage1_conv(x)
        stage2_out = torch.cat([self.stage2_drate_1(stage1_out), self.stage2_drate_2(stage1_out), self.stage2_drate_4(stage1_out), self.stage2_drate_8(stage1_out)],dim=1)
        return torch.cat([stage2_out, stage1_out], dim=1) + self.identity_conv(x)


class SPDConv(nn.Module):
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta

class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        x_sca = self.fgm(x_sca)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))
