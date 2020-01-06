# -*- coding: utf-8 -*-

"""
    StyleGAN2 pytorch

    @author: samuel ko
    @date:   2019.12.13

    @notice: 1) fused_conv: unsupport
             2) 4x4 upfirdn kernel: (transfer to 3x3 upfirdn kernel)

    @date:   2019.12.18

    @update: 1) fix upfirdn2d [1, 3, 1] to original [1, 3, 3, 1] in D_stylegan2 and Upsample2d.
             2) use_wscale = True (default), gain = 1 (default).
             3) update he_std calculation method to coordinate with the original repo.


    @date:   2019.12.20

    @update: 1) split ModulatedConv2d into 2 part.

    @date:   2020.01.02

    @update: 1) refact initialization part.
    @ref:    https://stackoverflow.com/questions/51136581/how-to-create-a-normal-distribution-in-pytorch
"""

import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch.nn import ModuleList

from utils.libs import _setup_kernel, _approximate_size
from opts.opts import TrainOptions, INFO

import copy
from tqdm import tqdm

from torchvision.utils import save_image
from matplotlib import pyplot as plt
from utils.utils import plotLossCurve


# from utils.libs import ShrinkFun

# shrink_fun = ShrinkFun.apply


# =========================================================================
#   Define components for G_mapping & G_synthesis_stylegan2 & D_stylegan2
# =========================================================================

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)  # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class BiasAdd(nn.Module):

    def __init__(self,
                 channels,
                 opts,
                 act='linear', alpha=None, gain=None, lrmul=1):
        """
            BiasAdd
        """
        super(BiasAdd, self).__init__()

        self.opts = opts
        self.bias = torch.nn.Parameter((torch.zeros(channels, 1, 1) * lrmul))

        self.act = act
        self.alpha = alpha if alpha is not None else 0.2
        self.gain = gain if gain is not None else 1.0

    def forward(self, x):
        # Pass Add bias.
        x += self.bias

        # Evaluate activation function.
        if self.act == "linear":
            pass
        elif self.act == 'lrelu':
            x = F.leaky_relu(x, self.alpha, inplace=True)
            x = x * np.sqrt(2)  # original repo def_gain=np.sqrt(2).

        # Scale by gain.
        if self.gain != 1:
            x = x * self.gain

        return x


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=1,
                 use_wscale=True,
                 lrmul=1.0,
                 bias=True,
                 act='lrelu',
                 mode='normal'):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain / np.sqrt((in_channels * out_channels))  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels).normal_(0, init_std))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

        self.act = act
        self.mode = mode

    def forward(self, x):
        if self.bias is not None and self.mode != 'modulate':
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        elif self.bias is not None and self.mode == 'modulate':
            # original
            # out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul) + 1
            # re-implement
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)

        if self.act == 'lrelu':
            out = F.leaky_relu(out, 0.2, inplace=True)
            out = out * np.sqrt(2)  # original repo def_gain=np.sqrt(2).
            return out
        elif self.act == 'linear':
            return out

        return out


class Conv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=1,
                 use_wscale=True,
                 lrmul=1,
                 bias=True,
                 act='linear'):
        super().__init__()

        assert kernel_size >= 1 and kernel_size % 2 == 1
        he_std = gain / np.sqrt((input_channels * output_channels * kernel_size * kernel_size))  # He init
        self.kernel_size = kernel_size
        self.act = act

        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.empty(output_channels, input_channels, kernel_size, kernel_size).normal_(0, init_std))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            out = F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)

        if self.act == 'lrelu':
            out = F.leaky_relu(out, 0.2, inplace=True)
            out = out * np.sqrt(2)  # original repo def_gain=np.sqrt(2).
            return out
        elif self.act == 'linear':
            return out


class FromRGB(nn.Module):
    """
        default non_linearity: LeakyReLU(0.2), def_gain= np.sqrt(2).
    """

    def __init__(self, input_channels, output_channels, use_wscale=True, lrmul=1):
        super().__init__()
        self.conv = Conv2d(input_channels=input_channels,
                           output_channels=output_channels,
                           kernel_size=1, use_wscale=use_wscale, lrmul=lrmul)

    def forward(self, x):
        x, y = x
        y1 = self.conv(y)
        out = F.leaky_relu(y1, 0.2, inplace=True)
        out = out * np.sqrt(2)  # original repo def_gain=np.sqrt(2).
        return out if x is None else out + x


class ModulatedConv2d(nn.Module):
    """
        Modulated convolution layer for G_synthesis_stylegan2.

        @date: 2019.12.19
        @update: 1) initialization update (He init).
                 2) refact ModulatedConv2d.
    """

    def __init__(self, input_channels, output_channels,
                 kernel_size,
                 opts,
                 k=[1, 3, 3, 1],
                 dlatent_size=512,
                 up=False,
                 down=False,
                 demodulate=True,
                 gain=1,
                 use_wscale=True,
                 lrmul=1,
                 fused_modconv=True):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1

        self.demodulate = demodulate
        self.fused_modconv = fused_modconv
        self.up, self.down = up, down
        self.fmaps = output_channels
        self.opts = opts

        self.conv = Conv2d(input_channels=input_channels,
                           output_channels=output_channels,
                           kernel_size=1, use_wscale=use_wscale, lrmul=lrmul)

        he_std = gain / np.sqrt((input_channels * output_channels * kernel_size * kernel_size))  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.w = torch.nn.Parameter(torch.empty(output_channels, input_channels, kernel_size, kernel_size).normal_(0, init_std))
        self.convH, self.convW = self.w.shape[2:]

        self.dense = FC(dlatent_size, input_channels, gain, lrmul=lrmul, use_wscale=use_wscale, mode='modulate',
                        act='linear')

        if self.up:
            factor = 2
            self.k = _setup_kernel(k) * (gain * (factor ** 2))  # 4 x 4
            self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
            self.k = torch.flip(self.k, [2, 3])
            self.k = torch.nn.Parameter(self.k, requires_grad=False)

            self.p = self.k.shape[0] - factor - (kernel_size - 1)

            self.padx0, self.pady0 = (self.p + 1) // 2 + factor - 1, (self.p + 1) // 2 + factor - 1
            self.padx1, self.pady1 = self.p // 2 + 1, self.p // 2 + 1

            self.kernelH, self.kernelW = self.k.shape[2:]

    def forward(self, x):
        x, y = x
        if len(y.shape) > 2:
            # y is dlatent in ToRGB.
            y = y.squeeze(1)
        # x Input: N, C, H, W (NxCx4x4, NxCx8x8, NxCx16x16, ...)
        # y Input: Disentangled latents(W) [minibatch, 1, dlatent_size].

        # Modulate.
        s = self.dense(y)  # [BI] Transform incoming W to style.

        # OIkk ---> BkkOI  ---> BkkIO
        self.ww = (self.w * self.w_lrmul).unsqueeze(0)
        self.ww = self.ww.repeat(s.shape[0], 1, 1, 1, 1)
        self.ww = self.ww.permute(0, 3, 4, 1, 2)
        self.ww = self.ww * s.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [BkkOI] Scale input feature maps.
        self.ww = self.ww.permute(0, 1, 2, 4, 3)  # [BkkIO]

        # Demodulate.
        if self.demodulate:
            d = torch.mul(self.ww, self.ww)
            d = torch.rsqrt(torch.sum(d, dim=[1, 2, 3]) + 1e-8)  # [BO] Scaling factor.
            self.ww = self.ww * (d.unsqueeze(1).unsqueeze(1).unsqueeze(1))  # [BkkIO] Scale output feature maps.

        # Reshape/scale input.
        if self.fused_modconv:
            x = x.view(1, -1, x.shape[2], x.shape[3])  # Fused => reshape minibatch to convolution groups.
            self.w_new = torch.reshape(self.ww.permute(0, 4, 3, 1, 2),
                                       (-1, x.shape[1], self.ww.shape[1], self.ww.shape[2]))
        else:
            x = x * (s.unsqueeze(-1).unsqueeze(-1))  # [BIhw] Not fused => scale input activations.
            self.w_new = self.w * self.w_lrmul

        # Convolution with optional up/downsampling.
        if self.up:
            outC, inC, convH, convW = self.w_new.shape[0], self.w_new.shape[1], self.w_new.shape[2], self.w_new.shape[3]

            # Transpose Weight
            num_groups = x.shape[1] // inC if (x.shape[1] // inC) >= 1 else 1
            self.w_new = self.w_new.reshape(-1, num_groups, inC, convH, convW)
            self.w_new = self.w_new.flip([3, 4])
            self.w_new = self.w_new.permute(2, 1, 0, 3, 4)
            self.w_new = self.w_new.reshape(inC, outC, convH, convW)

            x = F.conv_transpose2d(x, self.w_new, stride=2)

            # step 2: upfirdn2d
            y = x.clone()
            y = y.reshape([-1, x.shape[2], x.shape[3], 1])  # N C H W ---> N*C H W 1

            inC, inH, inW = x.shape[1:]
            # 1) Upsample
            y = y.reshape(-1, inH, inW, 1)

            # 2) Pad (crop if negative).
            y = F.pad(y, (0, 0,
                          max(self.pady0, 0), max(self.pady1, 0),
                          max(self.padx0, 0), max(self.padx1, 0),
                          0, 0
                          ))
            y = y[:,
                max(-self.pady0, 0): y.shape[1] - max(-self.pady1, 0),
                max(-self.padx0, 0): y.shape[2] - max(-self.padx1, 0),
                :]

            # 3) Convolve with filter.
            y = y.permute(0, 3, 1, 2)  # N*C H W 1 --> N*C 1 H W
            y = y.reshape(-1, 1, inH + self.pady0 + self.pady1, inW + self.padx0 + self.padx1)
            y = F.conv2d(y, self.k)
            y = y.view(-1, 1, inH + self.pady0 + self.pady1 - self.kernelH + 1,
                       inW + self.padx0 + self.padx1 - self.kernelW + 1)

            # 4) Downsample (throw away pixels).
            if inH != y.shape[1] or inH % 2 != 0:
                inH = inW = _approximate_size(inH)
                y = F.interpolate(y, size=(inH, inW), mode='bilinear')
            y = y.permute(0, 2, 3, 1)
            x = y.reshape(-1, inC, inH, inW)

        elif self.down:
            pass
        else:
            x = F.conv2d(x,
                         self.w_new,
                         padding=self.w_new.shape[2] // 2)
        # Reshape/scale output.
        if self.fused_modconv:
            x = x.reshape(-1, self.fmaps, x.shape[2], x.shape[3])  # Fused => reshape convolution groups back to minibatch.
        elif self.demodulate:
            x = x * d.unsqueeze(-1).unsqueeze(-1)                  # [BOhw] Not fused => scale output activations.

        return x


class ToRGB(nn.Module):
    """
        default non_linearity: LeakyReLU(0.2), def_gain= np.sqrt(2).

        2019.12.18 fix.
    """

    def __init__(self, input_channels, output_channels,
                 res,
                 opts,
                 use_wscale=True,
                 lrmul=1,
                 gain=1,
                 fused_modconv=True):
        super().__init__()
        assert res >= 2

        self.modulated_conv2d = ModulatedConv2d(input_channels=input_channels,
                                                output_channels=output_channels,
                                                kernel_size=1,
                                                up=False,
                                                use_wscale=use_wscale,
                                                lrmul=lrmul,
                                                gain=gain,
                                                demodulate=False,
                                                fused_modconv=fused_modconv,
                                                opts=opts)
        self.biasAdd = BiasAdd(opts=opts,
                               act='linear',
                               channels=output_channels)

        self.res = res
        self.opts = opts

    def forward(self, x):
        x, y, dlatent = x
        dlatent = dlatent[:, self.res * 2 - 3]

        x = self.modulated_conv2d([x, dlatent])
        t = self.biasAdd(x)

        return t if y is None else y + t


class GLayer(nn.Module):
    """
        GLayer.
    """

    def __init__(self, input_channels, output_channels,
                 layer_idx,
                 opts,
                 k=[1, 3, 3, 1],
                 randomize_noise=True,
                 up=False,
                 use_wscale=True,
                 lrmul=1,
                 fused_modconv=True,
                 act='lrelu'):
        super().__init__()

        self.randomize_noise = randomize_noise
        self.opts = opts
        self.up = up
        self.layer_idx = layer_idx

        self.modulated_conv2d = ModulatedConv2d(input_channels=input_channels,
                                                output_channels=output_channels,
                                                kernel_size=3,
                                                k=k,
                                                use_wscale=use_wscale,
                                                lrmul=lrmul,
                                                demodulate=True,
                                                fused_modconv=fused_modconv,
                                                up=up,
                                                opts=opts)

        # fixme: when you calling .to() on the parameter, it means that you are creating a non-leaf variable!
        self.noise_strength = torch.nn.Parameter(torch.zeros(1))
        self.biasAdd = BiasAdd(act=act,
                               channels=output_channels,
                               opts=opts)

    def forward(self, x):
        x, dlatent = x
        if len(dlatent.shape) > 2:
            dlatent = dlatent[:, self.layer_idx]

        x = self.modulated_conv2d([x, dlatent])

        noise = 0
        if self.randomize_noise:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(self.opts.device)

        x = x + noise * self.noise_strength
        x = self.biasAdd(x)

        return x


class Upsample2d(nn.Module):
    def __init__(self,
                 opts,
                 k=[1, 3, 3, 1],
                 factor=2,
                 down=1,
                 gain=1):
        """
            Upsample2d method in G_synthesis_stylegan2.
        :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                  The default is `[1] * factor`, which corresponds to average pooling.
        :param factor: Integer downsampling factor (default: 2).
        :param gain:   Scaling factor for signal magnitude (default: 1.0).

            Returns: Tensor of the shape `[N, C, H // factor, W // factor]`
        """
        super().__init__()
        assert isinstance(factor, int) and factor >= 1, "factor must be larger than 1! (default: 2)"

        self.gain = gain
        self.factor = factor
        self.opts = opts

        self.k = _setup_kernel(k) * (self.gain * (factor ** 2))  # 4 x 4
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
        self.k = torch.flip(self.k, [2, 3])
        self.k = nn.Parameter(self.k, requires_grad=False)

        self.p = self.k.shape[0] - self.factor

        self.padx0, self.pady0 = (self.p + 1) // 2 + factor - 1, (self.p + 1) // 2 + factor - 1
        self.padx1, self.pady1 = self.p // 2, self.p // 2

        self.kernelH, self.kernelW = self.k.shape[2:]
        self.down = down

    def forward(self, x):
        y = x.clone()
        y = y.reshape([-1, x.shape[2], x.shape[3], 1])  # N C H W ---> N*C H W 1

        inC, inH, inW = x.shape[1:]
        # step 1: upfirdn2d

        # 1) Upsample
        y = torch.reshape(y, (-1, inH, 1, inW, 1, 1))
        y = F.pad(y, (0, 0, self.factor - 1, 0, 0, 0, self.factor - 1, 0, 0, 0, 0, 0))
        y = torch.reshape(y, (-1, 1, inH * self.factor, inW * self.factor))

        # 2) Pad (crop if negative).
        y = F.pad(y, (0, 0,
                      max(self.pady0, 0), max(self.pady1, 0),
                      max(self.padx0, 0), max(self.padx1, 0),
                      0, 0
                      ))
        y = y[:,
            max(-self.pady0, 0): y.shape[1] - max(-self.pady1, 0),
            max(-self.padx0, 0): y.shape[2] - max(-self.padx1, 0),
            :]

        # 3) Convolve with filter.
        y = y.permute(0, 3, 1, 2)  # N*C H W 1 --> N*C 1 H W
        y = y.reshape(-1, 1, inH * self.factor + self.pady0 + self.pady1, inW * self.factor + self.padx0 + self.padx1)
        y = F.conv2d(y, self.k)
        y = y.view(-1, 1,
                   inH * self.factor + self.pady0 + self.pady1 - self.kernelH + 1,
                   inW * self.factor + self.padx0 + self.padx1 - self.kernelW + 1)

        # 4) Downsample (throw away pixels).
        if inH * self.factor != y.shape[1]:
            y = F.interpolate(y, size=(inH * self.factor, inW * self.factor), mode='bilinear')
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(-1, inC, inH * self.factor, inW * self.factor)

        return y


class ConvDownsample2d(nn.Module):
    def __init__(self,
                 kernel_size,
                 input_channels,
                 output_channels,
                 k=[1, 3, 3, 1],
                 factor=2,
                 gain=1,
                 use_wscale=True,
                 lrmul=1,
                 bias=False,
                 act='linear'):
        """
            ConvDownsample2D method in D_stylegan2.
        :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                  The default is `[1] * factor`, which corresponds to average pooling.
        :param factor: Integer downsampling factor (default: 2).
        :param gain:   Scaling factor for signal magnitude (default: 1.0).

            Returns: Tensor of the shape `[N, C, H // factor, W // factor]`
        """
        super().__init__()
        assert isinstance(factor, int) and factor >= 1, "factor must be larger than 1! (default: 2)"
        assert kernel_size >= 1 and kernel_size % 2 == 1

        he_std = gain / np.sqrt((input_channels * output_channels * kernel_size * kernel_size))  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        # https://discuss.pytorch.org/t/gaussian-distribution/35031
        self.weight = torch.nn.Parameter(torch.empty(output_channels, input_channels, kernel_size, kernel_size).normal_(0, init_std))
        self.convH, self.convW = self.weight.shape[2:]

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

        self.gain = gain
        self.factor = factor
        self.act = act

        self.k = _setup_kernel(k) * self.gain  # 3 x 3. (original 4 x 4).
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
        self.k = torch.flip(self.k, [2, 3])
        self.k = nn.Parameter(self.k, requires_grad=False)

        self.p = (self.k.shape[-1] - self.factor) + (self.convW - 1)

        self.padx0, self.pady0 = (self.p + 1) // 2, (self.p + 1) // 2
        self.padx1, self.pady1 = self.p // 2, self.p // 2

        self.kernelH, self.kernelW = self.k.shape[2:]

    def forward(self, x):

        y = x.clone()
        y = y.reshape([-1, x.shape[2], x.shape[3], 1])  # N C H W ---> N*C H W 1

        inC, inH, inW = x.shape[1:]
        # step 1: upfirdn2d
        # 1) Upsample
        y = torch.reshape(y, (-1, inH, inW, 1))

        # 2) Pad (crop if negative).
        y = F.pad(y, (0, 0,
                      max(self.pady0, 0), max(self.pady1, 0),
                      max(self.padx0, 0), max(self.padx1, 0),
                      0, 0
                      ))
        y = y[:,
            max(-self.pady0, 0): y.shape[1] - max(-self.pady1, 0),
            max(-self.padx0, 0): y.shape[2] - max(-self.padx1, 0),
            :]

        # 3) Convolve with filter.
        y = y.permute(0, 3, 1, 2)  # N*C H W 1 --> N*C 1 H W
        y = y.reshape(-1, 1, inH + self.pady0 + self.pady1, inW + self.padx0 + self.padx1)
        y = F.conv2d(y, self.k)
        y = y.view(-1, 1, inH + self.pady0 + self.pady1 - self.kernelH + 1,
                   inW + self.padx0 + self.padx1 - self.kernelW + 1)

        # 4) Downsample (throw away pixels).
        if inH != y.shape[1]:
            y = F.interpolate(y, size=(inH, inW), mode='bilinear')
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(-1, inC, inH, inW)

        # step 2: downsample (in general, stride = self.factor = 2)
        if self.bias is not None:
            x1 = F.conv2d(y,
                          self.weight * self.w_lrmul,
                          self.bias * self.b_lrmul,
                          stride=self.factor,
                          padding=self.convW // 2)
        else:
            x1 = F.conv2d(y,
                          self.weight * self.w_lrmul,
                          stride=self.factor,
                          padding=self.convW // 2)
        # step 3: non-linearity.
        if self.act == 'lrelu':
            out = F.leaky_relu(x1, 0.2, inplace=True)
            out = out * np.sqrt(2)  # original repo def_gain=np.sqrt(2).
        else:
            out = x1

        return out


class GBlock(nn.Module):
    """
        G_stylegan2 Basic Block.
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 layer_idx,
                 opts,
                 k=[1, 3, 3, 1],
                 use_wscale=True,
                 lrmul=1,
                 architecture='skip'):
        super().__init__()

        self.arch = architecture

        self.conv0up = GLayer(input_channels, output_channels,
                              layer_idx,
                              up=True,
                              k=k,
                              use_wscale=use_wscale,
                              lrmul=lrmul,
                              opts=opts)
        self.conv1 = GLayer(output_channels, output_channels,
                            layer_idx + 1,
                            up=False,
                            k=k,
                            use_wscale=use_wscale,
                            lrmul=lrmul,
                            opts=opts)

    def forward(self, x):
        x, dlatent = x
        x = self.conv0up([x, dlatent])
        x = self.conv1([x, dlatent])

        if self.arch == 'resnet':
            raise Exception("unsupported resnet architecture yet~")

        return x


class DBlock(nn.Module):
    """
        D_stylegan2 Basic Block.
    """

    def __init__(self, in1, in2, out3,
                 use_wscale=True,
                 lrmul=1,
                 resample_kernel=[1, 3, 3, 1],
                 architecture='resnet'):
        super().__init__()

        self.arch = architecture

        self.conv0 = Conv2d(input_channels=in1,
                            output_channels=in2,
                            kernel_size=3,
                            use_wscale=use_wscale,
                            lrmul=lrmul,
                            bias=True,
                            act='lrelu')

        self.conv1_down = ConvDownsample2d(kernel_size=3,
                                           input_channels=in2,
                                           output_channels=out3,
                                           k=resample_kernel,
                                           bias=True,
                                           act='lrelu')

        self.res_conv2_down = ConvDownsample2d(kernel_size=1,
                                               input_channels=in1,
                                               output_channels=out3,
                                               k=resample_kernel,
                                               bias=False)

    def forward(self, x):
        t = x.clone()

        x = self.conv0(x)
        x = self.conv1_down(x)

        if self.arch == 'resnet':
            t = self.res_conv2_down(t)
            x = (x + t) * (1 / np.sqrt(2))
        return x


# =========================================================================
#   Minibatch standard deviation layer. (D_stylegan2)
# =========================================================================

class Minibatch_stddev_layer(nn.Module):
    """
        Minibatch standard deviation layer. (D_stylegan2)
    """

    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()

        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        n, c, h, w = x.shape

        group_size = min(n, self.group_size)  # Minibatch must be divisible by (or smaller than) group_size.
        y = x.view(group_size, -1,
                   self.num_new_features,
                   c // self.num_new_features,
                   h, w)  # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = y - torch.mean(y, dim=0, keepdim=True)  # [GMncHW] Subtract mean over group.
        y = torch.mean(y ** 2, dim=0)  # [MncHW]  Calc variance over group.
        y = torch.sqrt(y + 1e-8)  # [MncHW]  Calc stddev over group.
        y = torch.mean(y, dim=[2, 3, 4], keepdim=True)  # [Mn111]  Take average over fmaps and pixels.
        y = torch.mean(y, dim=2)  # [Mn11]   Split channels into c channel groups
        # How to tile a tensor? https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853
        y = y.repeat(group_size, 1, h, w)  # [NnHW]  Replicate over group and pixels.

        return torch.cat([x, y], 1)  # [NCHW]  Append as new fmap.


# =========================================================================
#   Define G_mapping
#   1) support variant mapping_fmaps in G_mapping.
# =========================================================================

class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=1024,
                 label_size=0,  # Label dimensionality, 0 if no labels. (non-support)
                 mapping_layers=8,  # Number of mapping layers.
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,  # Enable equalized learning rate?
                 lrmul=0.01,  # Learning rate multiplier for the mapping layers.
                 gain=1  # original gain in tensorflow.
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.mapping_layers = mapping_layers

        self.fc1 = FC(self.mapping_fmaps, dlatent_size, gain=gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc_layers = ModuleList([])
        for _ in range(2, mapping_layers + 1):
            self.fc_layers.append(FC(dlatent_size, dlatent_size, gain=gain, lrmul=lrmul, use_wscale=use_wscale))

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)

        out = self.fc1(x)
        for fc in self.fc_layers:
            out = fc(out)

        out = out.unsqueeze(1)
        out = out.repeat(1, self.num_layers, 1)

        return out


# =========================================================================
#   Define G_synthesis_stylegan2
# =========================================================================

class G_synthesis_stylegan2(nn.Module):
    def __init__(self,
                 opts,
                 fmap_base=8 << 10,  # stylegan1 8192 (8 << 10), stylegan2 16384 (16 << 10)
                 num_channels=3,  # Number of output color channels.
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 resolution=1024,  # Output resolution.
                 randomize_noise=True,
                 # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_min=1,  # Minimum number of feature maps in any layer.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
                 use_wscale=True,  # Enable equalized learning rate?
                 lrmul=1,  # Learning rate multiplier for the mapping layers.
                 gain=1,  # original gain in tensorflow.
                 act='lrelu',  # Activation function: 'linear', 'lrelu'.
                 resample_kernel=[1, 3, 3, 1],
                 # Low-pass filter to apply when resampling activations. None = no filtering.
                 fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
                 ):
        super(G_synthesis_stylegan2, self).__init__()

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.nf = lambda stage: np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
        assert architecture in ['orig', 'skip', 'resnet']
        num_layers = resolution_log2 * 2 - 2

        self.arch = architecture
        self.act = act
        self.resolution_log2 = resolution_log2
        self.opts = opts

        # Primary inputs.
        self.x = torch.nn.Parameter(torch.randn(1, self.nf(1), 4, 4))
        # self.x = torch.randn(1, self.nf(1), 4, 4).to(self.opts.device)

        # layer0
        self.rgb0 = ToRGB(input_channels=self.nf(1),
                          output_channels=num_channels,
                          res=2,
                          opts=opts)
        self.glayer0 = GLayer(input_channels=self.nf(1),
                              output_channels=self.nf(1),
                              layer_idx=0,
                              k=resample_kernel,
                              randomize_noise=randomize_noise,
                              act=self.act,
                              up=False,
                              opts=opts)

        # rgb layers & block layers.
        self.rgb_layers = ModuleList([ToRGB(input_channels=self.nf(3),
                                            output_channels=num_channels,
                                            res=3,
                                            opts=opts,
                                            fused_modconv=fused_modconv)])
        self.block_layers = ModuleList([GBlock(input_channels=self.nf(2),
                                               output_channels=self.nf(3),
                                               layer_idx=1,
                                               opts=opts)])

        for res in range(4, self.resolution_log2 + 1):
            self.rgb_layers.append(ToRGB(input_channels=self.nf(res),
                                         output_channels=num_channels,
                                         res=res,
                                         opts=opts,
                                         fused_modconv=fused_modconv))
            self.block_layers.append(GBlock(input_channels=self.nf(res - 1),
                                            output_channels=self.nf(res),
                                            layer_idx=(res - 2) * 2 - 1,
                                            opts=opts))

        # upsample layer
        self.upsample2d = Upsample2d(opts=opts)

        self.tanh = torch.nn.Tanh()

    def forward(self, dlatent):
        # Early Layers
        y = None
        x = self.x.repeat(dlatent.shape[0], 1, 1, 1)
        x = self.glayer0([x, dlatent[:, 0]])

        if self.arch == 'skip':
            y = self.rgb0([x, y, dlatent])

        # Main layers.
        for res, (rgb, block) in enumerate(zip(self.rgb_layers, self.block_layers)):
            x = block([x, dlatent])
            if self.arch == 'skip':
                y = self.upsample2d(y)
            if self.arch == 'skip' or (res + 3) == self.resolution_log2:
                y = rgb([x, y, dlatent])

        # [-1, 1]
        # y = shrink_fun(y)
        # y = y / torch.max(torch.abs(y))
        # y = self.tanh(y)
        return y


# =========================================================================
#   Combine G_mapping & G_synthesis_stylegan2 in G_stylegan2
# =========================================================================

class G_stylegan2(nn.Module):
    def __init__(self,
                 opts,
                 return_dlatents=True,
                 fmap_base=8 << 10,  # stylegan1 8192 (8 << 10), stylegan2 16384 (16 << 10)
                 num_channels=3,  # Number of output color channels.
                 mapping_fmaps=512,
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 resolution=1024,  # Output resolution.
                 mapping_layers=8,  # Number of mapping layers.
                 randomize_noise=True,
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_min=1,  # Minimum number of feature maps in any layer.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 architecture='skip',  # Architecture: 'orig', 'skip'.
                 act='lrelu',  # Activation function: 'linear', 'lrelu'.
                 lrmul=0.01,  # Learning rate multiplier for the mapping layers.
                 gain=1,  # original gain in tensorflow.
                 truncation_psi=0.7,  # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=8,  # Number of layers for which to apply the truncation trick. None = disable.
                 ):
        super().__init__()
        assert architecture in ['orig', 'skip']

        self.return_dlatents = return_dlatents
        self.num_channels = num_channels

        self.g_mapping = G_mapping(mapping_fmaps=mapping_fmaps,
                                   dlatent_size=dlatent_size,
                                   resolution=resolution,
                                   mapping_layers=mapping_layers,
                                   lrmul=lrmul,
                                   gain=gain)

        self.g_synthesis = G_synthesis_stylegan2(resolution=resolution,
                                                 architecture=architecture,
                                                 randomize_noise=randomize_noise,
                                                 fmap_base=fmap_base,
                                                 fmap_min=fmap_min,
                                                 fmap_max=fmap_max,
                                                 fmap_decay=fmap_decay,
                                                 act=act,
                                                 opts=opts)

        self.truncation_cutoff = truncation_cutoff
        self.truncation_psi = truncation_psi

    def forward(self, x):
        dlatents1 = self.g_mapping(x)
        num_layers = dlatents1.shape[1]

        # Apply truncation trick.
        if self.truncation_psi and self.truncation_cutoff:
            batch_avg = torch.mean(dlatents1, dim=1, keepdim=True)
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                coefs[:, i, :] *= self.truncation_psi
            """Linear interpolation.
               a + (b - a) * t
            """
            dlatents1 = batch_avg + (dlatents1 - batch_avg) * torch.Tensor(coefs).to(dlatents1.device)

        out = self.g_synthesis(dlatents1)

        if self.return_dlatents:
            return out, dlatents1
        else:
            return out


# =========================================================================
#   Define D_stylegan2
#   1) support structure == origin & resnet. (skip is unsupport here.)
#   2) multi-label unsupport.
#   3) Almost coord with the original! (2019.12.18)
# =========================================================================

class D_stylegan2(nn.Module):
    def __init__(self,
                 resolution=1024,
                 fmap_base=8 << 10,  # stylegan1 8192 (8 << 10), stylegan2 16384 (16 << 10)
                 num_channels=3,
                 label_size=0,  # Dimensionality of the labels, 1 if no labels. Overridden based on dataset.
                 structure='resnet',  # Architecture: 'orig', 'resnet' (skip unsupported).
                 fmap_max=512,
                 fmap_min=1,
                 fmap_decay=1.0,
                 mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
                 mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
                 resample_kernel=[1, 3, 3, 1]
                 # Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        """
            Noitce: we only support input pic with height == width.

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4 and self.resolution_log2 >= 4
        self.nf = lambda stage: np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

        assert structure in ['orig', 'skip', 'resnet']

        if structure == 'skip':
            raise Exception("skip in Discriminator is unsupported yet~")

        self.structure = structure
        self.label_size = label_size
        self.mbstd_group_size = mbstd_group_size

        # sub network
        self.fromrgb = FromRGB(input_channels=3,
                               output_channels=self.nf(self.resolution_log2 - 1),
                               use_wscale=True)

        # dblock layers
        self.block_layers = ModuleList([])

        for res in range(self.resolution_log2, 4, -1):
            self.block_layers.append(DBlock(in1=self.nf(res - 1), in2=self.nf(res - 1), out3=self.nf(res - 2),
                                            resample_kernel=resample_kernel))

        for res in range(4, 2, -1):
            self.block_layers.append(
                DBlock(in1=self.nf(res), in2=self.nf(res - 1), out3=self.nf(res - 2), resample_kernel=resample_kernel))

        # 4x4
        self.minibatch_stddev = Minibatch_stddev_layer(mbstd_group_size, mbstd_num_features)
        self.conv_last = Conv2d(input_channels=self.nf(2) + mbstd_num_features,
                                output_channels=self.nf(1),
                                kernel_size=3,
                                act='lrelu')
        self.fc_last1 = FC(in_channels=fmap_base,
                           out_channels=self.nf(0),
                           act='lrelu')
        self.fc_last2 = FC(in_channels=self.nf(0),
                           out_channels=1,
                           act='linear')

    def forward(self, input):

        x_origin = None
        y = input
        # 1) Main Layers.
        x = self.fromrgb([x_origin, y])
        for dblock in self.block_layers:
            x = dblock(x)

        # 2) Final layers (4 x 4).
        if self.mbstd_group_size > 1:
            x = self.minibatch_stddev(x)
        x = self.conv_last(x)

        out = x

        _, c, h, w = out.shape
        out = out.view(-1, h * w * c)

        out = self.fc_last1(out)

        # 3) Output
        if self.label_size == 0:
            out = self.fc_last2(out)
            return out


# =========================================================================
#   Define StyleGAN2
# =========================================================================

class StyleGAN2:
    """ Unconditional StyleGAN2
    """

    def __init__(self,
                 opts,
                 use_ema=True,
                 ema_decay=0.999):
        """ constructor for the class """

        self.start_epoch = 0
        self.opts = opts
        # Create the model
        self.G = G_stylegan2(opts=opts,
                             fmap_base=opts.fmap_base,
                             resolution=opts.resolution,
                             mapping_layers=opts.mapping_layers,
                             return_dlatents=opts.return_latents,
                             architecture='skip')

        self.D = D_stylegan2(fmap_base=opts.fmap_base,
                             resolution=opts.resolution,
                             structure='resnet')

        # Load the pre-trained weight
        if os.path.exists(opts.resume):
            INFO("Load the pre-trained weight!")
            state = torch.load(opts.resume)
            self.G.load_state_dict(state['G'])
            self.D.load_state_dict(state['D'])
            self.start_epoch = state['start_epoch']
        else:
            INFO("Pre-trained weight cannot load successfully, train from scratch!")

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            INFO("Multiple GPU:" + str(torch.cuda.device_count()) + "\t GPUs")
            self.G = torch.nn.DataParallel(self.G)
            self.D = torch.nn.DataParallel(self.D)
        self.G.to(opts.device)
        self.D.to(opts.device)

        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        if self.use_ema:
            from utils.libs import update_average

            # create a shadow copy of the generator
            self.Gs = copy.deepcopy(self.G)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            Gs_beta = 0.99
            self.ema_updater(self.Gs, self.G, beta=Gs_beta)

        # by default the generator and discriminator are in eval mode
        self.G.eval()
        self.D.eval()
        if self.use_ema:
            self.Gs.eval()

    def optimize_G(self,
                   gen_optim,
                   dlatent,
                   real_batch,
                   loss_fn):
        """
               performs one step of weight update on generator using the batch of data
               :param gen_optim: generator optimizer
               :param dlatent: input noise of sample generation
               :param real_batch: real samples batch
                                  should contain a list of tensors at different scales
               :param loss_fn: loss function to be used (object of GANLoss)
               :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.G(dlatent)
        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        # if self.use_ema is true, apply the moving average here:
        if self.use_ema:
            self.ema_updater(self.Gs, self.G, self.ema_decay)

        return loss.mean().item()

    def optimize_D(self,
                   dis_optim,
                   dlatent,
                   real_batch,
                   loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param dlatent: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """
        # generate a batch of samples
        fake_samples = self.G(dlatent)
        fake_samples = fake_samples.detach()

        loss = loss_fn.dis_loss(real_batch, fake_samples)

        # optimize discriminator
        dis_optim.zero_grad()
        loss.backward()
        dis_optim.step()

        return loss.mean().item()

    def train(self,
              data_loader,
              gen_optim,
              dis_optim,
              loss_fn,
              scheduler_gen,
              scheduler_dis
              ):
        """
           Method for training the network
           1) data_loader.    Dataloader in PyTorch.
           2) gen_optim.      torch.optim.Optimizer for Generator.
           3) dis_optim.      torch.optim.Optimizer for Discriminator.
           4) loss_fn.        loss/ganloss.py StyleLoss.
           5) scheduler_gen.  scheduler_gen.
           6) scheduler_dis.  scheduler_dis.
        """

        # turn the generator and discriminator into train mode
        self.G.train()
        self.D.train()

        # Train
        fix_z = torch.randn([self.opts.batch_size, 512]).to(self.opts.device)
        softplus = torch.nn.Softplus()
        Loss_D_list = [0.0]
        Loss_G_list = [0.0]
        for ep in range(self.start_epoch, self.opts.epoch):
            bar = tqdm(data_loader)
            loss_D_list = []
            loss_G_list = []
            for i, (real_img,) in enumerate(bar):
                real_img = real_img.to(self.opts.device)
                latents = torch.randn([real_img.size(0), 512]).to(self.opts.device)

                # optimize the discriminator:
                d_loss = self.optimize_D(dis_optim, latents,
                                         real_img, loss_fn)

                # optimize the generator:
                g_loss = self.optimize_G(gen_optim, latents,
                                         real_img, loss_fn)

                loss_G_list.append(g_loss)
                loss_D_list.append(d_loss)

                # Output training stats
                bar.set_description(
                    "Epoch {} [{}, {}] [G]: {} [D]: {}".
                        format(ep, i + 1, len(data_loader), loss_G_list[-1], loss_D_list[-1]))

            # Save the result
            Loss_G_list.append(np.mean(loss_G_list))
            Loss_D_list.append(np.mean(loss_D_list))

            # Check how the generator is doing by saving G's output on fixed_noise
            with torch.no_grad():
                if self.opts.return_latents:
                    fake_img = self.G(fix_z)[0].detach().cpu()
                else:
                    fake_img = self.G(fix_z).detach().cpu()
                save_image(fake_img, os.path.join(self.opts.det, 'images', str(ep) + '.png'), nrow=4, normalize=True)

            # Save model
            state = {
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'Loss_G': Loss_G_list,
                'Loss_D': Loss_D_list,
                'start_epoch': ep,
            }
            torch.save(state, os.path.join(self.opts.det, 'models', 'latest.pth'))

            scheduler_gen.step()
            scheduler_dis.step()

        # Plot the total loss curve
        Loss_D_list = Loss_D_list[1:]
        Loss_G_list = Loss_G_list[1:]
        plotLossCurve(self.opts, Loss_D_list, Loss_G_list)


if __name__ == "__main__":
    # 1) G_mapping ok.
    # data = torch.randn([1, 512])
    # g = G_mapping()
    # print(g(data).shape)

    # 2) D_stylegan2 ok (upfirdn).
    from loss.loss import D_logistic_r1

    data = torch.randn(1, 3, 256, 256).cuda()
    print(torch.max(data))
    print(torch.min(data))
    # fake = torch.randn(1, 3, 256, 256).cuda()
    d = D_stylegan2(resolution=256,
                    structure='resnet',
                    resample_kernel=[1, 3, 3, 1]).cuda()
    print(d(data))
    # https://discuss.pytorch.org/t/one-of-the-differentiated-tensors-does-not-require-grad/54694
    # D_logistic_r1(fake, data, d)

    # 3) G_synthesis_stylegan2 Early Layers
    # data = torch.randn([5, 18, 512])
    # g_syn = G_synthesis_stylegan2(resolution=128)
    # print(g_syn(data).shape)

    # 3.1) ModulatedConv2d(up=True) UpConvsample2d
    # y = torch.randn([1, 512])
    # x = torch.randn(1, 3, 128, 128)
    # up1 = ModulatedConv2d(up=True,
    #                       input_channels=3,
    #                       output_channels=6,
    #                       kernel_size=3)
    # print(up1([x, y]).shape)

    # 3.2) Upsample2d
    # data = torch.randn(1, 3, 128, 128)
    # up = Upsample2d()
    # print(up(data).shape)

    # 4) G_combination
    # opts = TrainOptions().parse()
    # data = torch.randn([5, 512])
    # g = G_stylegan2(resolution=256,
    #                 mapping_layers=5,
    #                 opts=opts)
    # print(g(data).shape)
    # print(g(data).shape)
    # torch.Size([5, 3, 256, 256])
