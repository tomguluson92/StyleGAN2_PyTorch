# -*- coding: utf-8 -*-

"""
    StyleGAN2 pytorch

    @author: samuel ko
    @date:   2019.12.13

    @notice: 1) fused_conv: unsupport
             2) 4x4 upfirdn kernel: (transfer to 3x3 upfirdn kernel)
"""

import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from collections import OrderedDict

from utils.libs import _setup_kernel
from opts.opts import TrainOptions


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
        self.bias = (torch.nn.Parameter(torch.zeros(channels, 1, 1)) * lrmul).to(self.opts.device)

        self.act = act
        self.alpha = alpha if alpha is not None else 0.2
        self.gain = gain if gain is not None else 1.0

    def forward(self, x):
        # Pass Add bias.
        x = x + self.bias

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
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True,
                 act='lrelu',
                 mode='normal'):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
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
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul + 1)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)

        if self.act == 'lrelu':
            out = F.leaky_relu(out, 0.2, inplace=True)
            # out = F.leaky_relu(out, 0.2)
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
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()

        assert kernel_size >= 1 and kernel_size % 2 == 1
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


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
        x = self.conv(x)
        out = F.leaky_relu(x, 0.2, inplace=True)
        # out = F.leaky_relu(x, 0.2)
        out = out * np.sqrt(2)  # original repo def_gain=np.sqrt(2).
        return out


class ModulatedConv2d(nn.Module):
    """
        Modulated convolution layer for G_synthesis_stylegan2.
    """

    def __init__(self, input_channels, output_channels,
                 kernel_size,
                 opts,
                 dlatent_size=512,
                 up=False, down=False,
                 demodulate=True,
                 gain=1,
                 use_wscale=True, lrmul=1,
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

        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.w = torch.nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std).to(self.opts.device)
        self.convH, self.convW = self.w.shape[2:]

        self.dense = FC(dlatent_size, input_channels, gain, lrmul=lrmul, use_wscale=use_wscale, mode='modulate')

        self.upsample = UpConvsample2d(kernel_size=kernel_size, opts=opts)

    def forward(self, x):
        x, y = x
        if len(y.shape) > 2:
            # y is dlatent in ToRGB.
            y = y.squeeze(1)
        # x Input: N, C, H, W (NxCx4x4, NxCx8x8, NxCx16x16, ...)
        # y Input: Disentangled latents(W) [minibatch, 1, dlatent_size].

        # Modulate.
        s = self.dense(y)  # [BI] Transform incoming W to style.
        self.ww = self.w.unsqueeze(0) * s.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(self.opts.device)  # [BOIkk] Scale input feature maps.

        # Demodulate.
        if self.demodulate:
            d = torch.mul(self.ww, self.ww)
            d = torch.rsqrt(torch.mean(d, dim=[2, 3, 4], keepdim=True) + 1e-8)  # [BO] Scaling factor.
            self.ww = self.ww * d  # [BOIkk] Scale output feature maps.

        # Reshape/scale input.
        if self.fused_modconv:
            # x = x.reshape(1, -1, x.shape[2], x.shape[3])  # Fused => reshape minibatch to convolution groups.
            self.w_new = self.w.view(-1, self.ww.shape[2], self.ww.shape[3], self.ww.shape[4])
        else:
            x = x * s.unsqueeze(-1).unsqueeze(-1).to(self.opts.device)  # [BIhw] Not fused => scale input activations.
            self.w_new = self.w

        # Convolution with optional up/downsampling.
        if self.up:
            x = self.upsample([x, self.w_new])
        elif self.down:
            pass
        else:
            x = F.conv2d(x,
                         self.w_new * self.w_lrmul,
                         padding=self.w_new.shape[2] // 2)
        # Reshape/scale output.
        if self.fused_modconv:
            x = x.view(-1, self.fmaps, x.shape[2], x.shape[3])  # Fused => reshape convolution groups back to minibatch.
        elif self.demodulate:
            x = x * d.squeeze(-1)  # [BOhw] Not fused => scale output activations.

        return x


class ToRGB(nn.Module):
    """
        default non_linearity: LeakyReLU(0.2), def_gain= np.sqrt(2).
    """

    def __init__(self, input_channels, output_channels,
                 res,
                 opts,
                 use_wscale=True,
                 lrmul=1,
                 fused_modconv=False):
        super().__init__()
        assert res >= 2

        self.modulated_conv2d = ModulatedConv2d(input_channels=input_channels,
                                                output_channels=output_channels,
                                                kernel_size=1,
                                                use_wscale=use_wscale,
                                                lrmul=lrmul,
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
        x = self.biasAdd(x)

        if y is None:
            return x
        else:
            return x + y


class GLayer(nn.Module):
    """
        GLayer.
    """

    def __init__(self, input_channels, output_channels,
                 layer_idx,
                 opts,
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
                                                use_wscale=use_wscale,
                                                lrmul=lrmul,
                                                demodulate=True,
                                                fused_modconv=fused_modconv,
                                                up=up,
                                                opts=opts)

        self.noise_strength = torch.nn.Parameter(torch.zeros(1)).to(self.opts.device)
        self.biasAdd = BiasAdd(act=act, channels=output_channels, opts=opts)

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
                 k=[1, 3, 1],
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

        self.k = _setup_kernel(k) * (self.gain * (factor ** 2))  # 3 x 3 (original 4 x 4 in tf).
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
        # todo: must wrap self.k into nn.Parameter, or the VRAM exceeds the limits.
        self.k = nn.Parameter(self.k, requires_grad=False).to(self.opts.device)

        self.p = self.k.shape[0] - self.factor

        # fixme: incompatible with original version.
        self.padx0, self.pady0 = (self.p + 1) // 2 + factor - 1, (self.p + 1) // 2 + factor - 1
        # self.padx0, self.pady0 = 0, 0
        self.padx1, self.pady1 = self.p // 2, self.p // 2
        # self.padx1, self.pady1 = 0, 0

        self.kernelH, self.kernelW = self.k.shape[2:]
        self.down = down

    def forward(self, x):
        inC, inH, inW = x.shape[1:]
        # step 1: upfirdn2d

        # 1) Upsample
        x = torch.reshape(x, (-1, inH, 1, inW, 1, inC))
        x = F.pad(x, (0, 0, self.factor - 1, 0, 0, 0, self.factor - 1, 0, 0, 0, 0, 0))
        x = torch.reshape(x, (-1, inC, inH * self.factor, inW * self.factor))

        # 2) Pad (crop if negative).
        x = F.pad(x, (
            max(self.pady0, 0), max(self.pady1, 0),
            max(self.padx0, 0), max(self.padx1, 0),
            0, 0,
            0, 0,
        ))
        x = x[:, :,
            max(-self.pady0, 0): x.shape[2] - max(-self.pady1, 0),
            max(-self.padx0, 0): x.shape[3] - max(-self.padx1, 0)]
        # 3) Convolve with filter.

        x = x.view(-1, 1, inH * self.factor + self.pady0 + self.pady1, inW * self.factor + self.padx0 + self.padx1)
        x = F.conv2d(x, self.k, padding=self.kernelH // 2)
        x = x.view(-1, inC,
                   inH * self.factor + self.pady0 + self.pady1,
                   inW * self.factor + self.padx0 + self.padx1)

        # Downsample (throw away pixels).
        return x[:, :, ::self.down, ::self.down]


class UpConvsample2d(nn.Module):
    def __init__(self,
                 opts,
                 kernel_size=3,
                 k=[1, 3, 1],
                 factor=2,
                 gain=1,
                 down=1,
                 use_wscale=True,
                 lrmul=1,
                 bias=True):
        """
            UpConvsample2D method in G_synthesis_stylegan2.
        :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                  The default is `[1] * factor`, which corresponds to average pooling.
        :param factor: Integer downsampling factor (default: 2).
        :param gain:   Scaling factor for signal magnitude (default: 1.0).

            Returns: Tensor of the shape `[N, C, H * factor, W * factor]`
        """
        super().__init__()
        assert isinstance(factor, int) and factor >= 1, "factor must be larger than 1! (default: 2)"

        self.gain = gain
        self.factor = factor

        self.k = _setup_kernel(k) * (self.gain * (factor ** 2))  # 3 x 3 (original 4 x 4 in tf).
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
        # todo: must wrap self.k into nn.Parameter, or the VRAM exceeds the limits.
        self.k = nn.Parameter(self.k, requires_grad=False).to(opts.device)

        # default conv kernel size is 3(in tf version).
        self.p = self.k.shape[0] - self.factor + kernel_size - 1

        # fixme: incompatible with original version.
        self.padx0, self.pady0 = (self.p + 1) // 2 + factor - 1, (self.p + 1) // 2 + factor - 1
        # self.padx0, self.pady0 = 0, 0
        self.padx1, self.pady1 = self.p // 2, self.p // 2
        # self.padx1, self.pady1 = 0, 0

        self.kernelH, self.kernelW = self.k.shape[2:]
        self.down = down

    def forward(self, x):
        x, w = x
        inC, outC, convH, convW = w.shape[0], w.shape[1], w.shape[2], w.shape[3]
        # Determine data dimensions.
        # num_groups = x.shape[1] // inC if x.shape[1] // inC > 0 else 1

        w = w.reshape(outC, inC, convH, convW)
        x = F.conv_transpose2d(x, w, stride=self.factor)

        inC, inH, inW = x.shape[1:]

        # step 2: upfirdn2d
        if self.factor == 2:
            factor = 1
        elif self.factor == 0:
            self.factor = 1

        # 1) Upsample
        x = torch.reshape(x, (-1, inH, 1, inW, 1, inC))
        x = F.pad(x, (0, 0, factor - 1, 0, 0, 0, factor - 1, 0, 0, 0, 0, 0))
        x = torch.reshape(x, (-1, inC, inH * factor, inW * factor))

        # 2) Pad (crop if negative).
        x = F.pad(x, (
            max(self.pady0, 0), max(self.pady1, 0),
            max(self.padx0, 0), max(self.padx1, 0),
            0, 0,
            0, 0,
        ))
        x = x[:, :,
            max(-self.pady0, 0): x.shape[2] - max(-self.pady1, 0),
            max(-self.padx0, 0): x.shape[3] - max(-self.padx1, 0)]
        # 3) Convolve with filter.

        x = x.view(-1, 1, inH * factor + self.pady0 + self.pady1, inW * factor + self.padx0 + self.padx1)
        x = F.conv2d(x, self.k)
        x = x.view(-1, inC,
                   inH * factor,
                   inW * factor)

        # Downsample (throw away pixels).
        if x.shape[3] % 2 != 0:
            self.down = x.shape[3] - 1
            return x[:, :, :self.down, :self.down]
        else:
            return x


class ConvDownsample2d(nn.Module):
    def __init__(self,
                 kernel_size,
                 input_channels,
                 output_channels,
                 k=[1, 3, 1],
                 factor=2,
                 gain=1,
                 use_wscale=True,
                 lrmul=1,
                 bias=True):
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

        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        # https://discuss.pytorch.org/t/gaussian-distribution/35031
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        self.convH, self.convW = self.weight.shape[2:]

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

        self.gain = gain
        self.factor = factor

        self.k = _setup_kernel(k) * self.gain  # 3 x 3. (original 4 x 4).
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0).to('cuda')
        # todo: must wrap self.k into nn.Parameter, or the VRAM exceeds the limits.
        # self.k = nn.Parameter(self.k, requires_grad=False)
        self.k = nn.Parameter(self.k, requires_grad=False)

        self.p = (self.k.shape[0] - self.factor) + (self.convW - 1)

        # fixme: incompatible with original version.
        self.padx0, self.pady0 = (self.p + 1) // 2, (self.p + 1) // 2
        # self.padx0, self.pady0 = 0, 0
        self.padx1, self.pady1 = self.p // 2, self.p // 2
        # self.padx1, self.pady1 = 0, 0

        self.kernelH, self.kernelW = self.k.shape[2:]

    def forward(self, x):

        inC, inH, inW = x.shape[1:]
        # step 1: upfirdn2d
        # 1) Upsample
        # x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
        # x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
        # x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])
        # torch
        # x = torch.reshape(x, (-1, inH, 1, inW, 1, inC))
        # x = F.pad(x, (0, 0, 0, 0, 0, upy-1, 0, 0, 0, upx-1, 0, 0))
        # x = torch.reshape(x, (-1, inH * upy, inW * upx, inC))

        # 2) Pad (crop if negative).
        # x = F.pad(x, (0, 0,
        #               0, 0,
        #               max(self.pady0, 0), max(self.pady1, 0),
        #               max(self.padx0, 0), max(self.padx1, 0)
        #               ))
        # x = x[:, :,
        #     max(-self.pady0, 0): inH - max(-self.pady1, 0),
        #     max(-self.padx0, 0): inW - max(-self.padx1, 0)]

        # 3) Convolve with filter.
        # 4) Downsample (throw away pixels).
        # x = x[:, :, ::1, ::1]
        x = x.view(-1, 1, inH, inW)
        x = F.conv2d(x, self.k, padding=self.kernelH // 2)
        x = x.view(-1, inC, inH, inW)

        # step 2: downsample (in general, stride = self.factor = 2)
        x = F.conv2d(x,
                     self.weight * self.w_lrmul,
                     self.bias * self.b_lrmul,
                     stride=self.factor,
                     padding=self.convW // 2)
        # step 3: non-linearity.
        out = F.leaky_relu(x, 0.2, inplace=True)
        # out = F.leaky_relu(x, 0.2)
        out = out * np.sqrt(2)  # original repo def_gain=np.sqrt(2).

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
                 use_wscale=True,
                 lrmul=1,
                 architecture='skip'):
        super().__init__()

        self.arch = architecture

        self.conv0up = GLayer(input_channels, output_channels,
                              layer_idx,
                              up=True,
                              use_wscale=use_wscale,
                              lrmul=lrmul,
                              opts=opts)
        self.conv1 = GLayer(output_channels, output_channels,
                            layer_idx + 1,
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

    def __init__(self, in1, in2, out3, use_wscale=True, lrmul=1,
                 architecture='resnet'):
        super().__init__()

        self.arch = architecture

        self.conv0 = Conv2d(input_channels=in1,
                            output_channels=in2,
                            kernel_size=3, use_wscale=use_wscale, lrmul=lrmul,
                            bias=True)

        self.conv1_down = ConvDownsample2d(kernel_size=3,
                                           input_channels=in2,
                                           output_channels=out3)

        self.res_conv2_down = ConvDownsample2d(kernel_size=1,
                                               input_channels=in1,
                                               output_channels=out3)

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
                 gain=2 ** (0.5)  # original gain in tensorflow.
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.mapping_layers = mapping_layers

        self.fc1 = FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc2 = FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc3 = FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc4 = FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc5 = FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc6 = FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc7 = FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        self.fc8 = FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)

        out = self.fc1(x)
        for _ in range(2, self.mapping_layers + 1):
            out = getattr(self, 'fc{}'.format(_))(out)

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
                 lrmul=0.01,  # Learning rate multiplier for the mapping layers.
                 gain=2 ** (0.5),  # original gain in tensorflow.
                 act='lrelu',  # Activation function: 'linear', 'lrelu'.
                 resample_kernel=[1, 3, 1],
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
        self.x = nn.Parameter(torch.randn(1, self.nf(1), 4, 4), requires_grad=False).to(self.opts.device)

        # layer0
        self.rgb0 = ToRGB(input_channels=self.nf(1),
                          output_channels=num_channels,
                          res=2,
                          opts=opts)
        self.glayer0 = GLayer(input_channels=self.nf(1),
                              output_channels=self.nf(1),
                              layer_idx=0,
                              randomize_noise=randomize_noise,
                              act=self.act,
                              up=False,
                              opts=opts)

        # rgb layer
        self.rgb3 = ToRGB(input_channels=self.nf(3),
                          output_channels=num_channels,
                          res=3,
                          opts=opts)
        self.rgb4 = ToRGB(input_channels=self.nf(4),
                          output_channels=num_channels,
                          res=4,
                          opts=opts)
        self.rgb5 = ToRGB(input_channels=self.nf(5),
                          output_channels=num_channels,
                          res=5,
                          opts=opts)
        self.rgb6 = ToRGB(input_channels=self.nf(6),
                          output_channels=num_channels,
                          res=6,
                          opts=opts)
        self.rgb7 = ToRGB(input_channels=self.nf(7),
                          output_channels=num_channels,
                          res=7,
                          opts=opts)
        self.rgb8 = ToRGB(input_channels=self.nf(8),
                          output_channels=num_channels,
                          res=8,
                          opts=opts)
        self.rgb9 = ToRGB(input_channels=self.nf(9),
                          output_channels=num_channels,
                          res=9,
                          opts=opts)
        self.rgb10 = ToRGB(input_channels=self.nf(10),
                           output_channels=num_channels,
                           res=10,
                           opts=opts)

        # block layer
        self.block3 = GBlock(input_channels=self.nf(2),
                             output_channels=self.nf(3),
                             layer_idx=1,
                             opts=opts)
        self.block4 = GBlock(input_channels=self.nf(3),
                             output_channels=self.nf(4),
                             layer_idx=3,
                             opts=opts)
        self.block5 = GBlock(input_channels=self.nf(4),
                             output_channels=self.nf(5),
                             layer_idx=5,
                             opts=opts)
        self.block6 = GBlock(input_channels=self.nf(5),
                             output_channels=self.nf(6),
                             layer_idx=7,
                             opts=opts)
        self.block7 = GBlock(input_channels=self.nf(6),
                             output_channels=self.nf(7),
                             layer_idx=9,
                             opts=opts)
        self.block8 = GBlock(input_channels=self.nf(7),
                             output_channels=self.nf(8),
                             layer_idx=11,
                             opts=opts)
        self.block9 = GBlock(input_channels=self.nf(8),
                             output_channels=self.nf(9),
                             layer_idx=13,
                             opts=opts)
        self.block10 = GBlock(input_channels=self.nf(9),
                              output_channels=self.nf(10),
                              layer_idx=15,
                              opts=opts)
        # upsample layer
        self.upsample2d = Upsample2d(opts=opts)


    def forward(self, dlatent):
        # Early Layers
        y = None
        x = self.x.repeat(dlatent.shape[0], 1, 1, 1)
        x = self.glayer0([x, dlatent[:, 0]])

        if self.arch == 'skip':
            y = self.rgb0([x, y, dlatent])

        # Main layers.
        # for res in range(3, self.resolution_log2 + 1):
        for res in range(3, self.resolution_log2+1):
            x = getattr(self, 'block{}'.format(res))([x, dlatent])
            if self.arch == 'skip':
                y = self.upsample2d(y)
            if self.arch == 'skip' or res == self.resolution_log2:
                y = getattr(self, 'rgb{}'.format(res))([x, y, dlatent])

        # [-1, 1]
        y = y / torch.max(torch.abs(y))
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
                 gain=2 ** (0.5)  # original gain in tensorflow.
                 ):
        super().__init__()
        assert architecture in ['orig', 'skip']

        self.return_dlatents = return_dlatents

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

    def forward(self, x):
        dlatent = self.g_mapping(x)
        x = self.g_synthesis(dlatent)

        if self.return_dlatents:
            return x, dlatent
        else:
            return x


# =========================================================================
#   Define D_stylegan2
#   1) support structure == origin & resnet. (skip is unsupport here.)
#   2) multi-label unsupport.
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
                 mbstd_num_features=2,  # Number of features for the minibatch standard deviation layer.
                 resample_kernel=[1, 3, 3, 3, 1]
                 # Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        """
            Noitce: we only support input pic with height == width.

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        self.nf = lambda stage: np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
        # fromrgb: fixed mode
        self.fromrgb = Conv2d(num_channels, self.nf(self.resolution_log2 - 1), kernel_size=1)

        assert structure in ['orig', 'skip', 'resnet']

        if structure == 'skip':
            raise Exception("skip in Discriminator is unsupported yet~")

        self.structure = structure
        self.label_size = label_size
        self.mbstd_group_size = mbstd_group_size

        # sub network
        self.fromrgb = FromRGB(input_channels=3, output_channels=self.nf(self.resolution_log2 - 1))

        self.dblock10 = DBlock(in1=self.nf(9), in2=self.nf(9), out3=self.nf(8))
        self.dblock9 = DBlock(in1=self.nf(8), in2=self.nf(8), out3=self.nf(7))
        self.dblock8 = DBlock(in1=self.nf(7), in2=self.nf(7), out3=self.nf(6))
        self.dblock7 = DBlock(in1=self.nf(6), in2=self.nf(6), out3=self.nf(5))
        self.dblock6 = DBlock(in1=self.nf(5), in2=self.nf(5), out3=self.nf(4))
        self.dblock5 = DBlock(in1=self.nf(4), in2=self.nf(4), out3=self.nf(3))
        self.dblock4 = DBlock(in1=self.nf(4), in2=self.nf(3), out3=self.nf(2))
        self.dblock3 = DBlock(in1=self.nf(3), in2=self.nf(2), out3=self.nf(1))
        self.dblock2 = DBlock(in1=self.nf(2), in2=self.nf(1), out3=self.nf(0))

        # 4x4
        self.minibatch_stddev = Minibatch_stddev_layer(mbstd_group_size, mbstd_num_features)
        self.conv_last = Conv2d(input_channels=self.nf(2) + mbstd_num_features,
                                output_channels=self.nf(1),
                                kernel_size=3)
        self.fc_last1 = FC(in_channels=fmap_base,
                           out_channels=self.nf(0))
        self.fc_last2 = FC(in_channels=self.nf(0),
                           out_channels=1,
                           act='linear')

    def forward(self, input):

        # 1) Main Layers.
        x = self.fromrgb(input)
        for res in range(self.resolution_log2, 2, -1):
            x = getattr(self, 'dblock{}'.format(res))(x)
        # 2) Final layers (4 x 4).
        if self.mbstd_group_size > 1:
            x = self.minibatch_stddev(x)
        x = self.conv_last(x)

        out = F.leaky_relu(x, 0.2, inplace=True)
        # out = F.leaky_relu(x, 0.2)
        out = out * np.sqrt(2)

        _, c, h, w = out.shape

        out = out.view(-1, h * w * c)
        out = self.fc_last1(out)

        # 3) Output
        if self.label_size == 0:
            out = self.fc_last2(out)
            return out
        else:
            raise Exception("Sorry, multi-label unsupported right now.")


if __name__ == "__main__":
    # 1) G_mapping ok.
    # data = torch.randn([1, 512])
    # g = G_mapping()
    # print(g(data).shape)

    # 2) D_stylegan2 ok (upfirdn).
    from loss.loss import D_logistic_r1
    data = torch.randn(1, 3, 256, 256).cuda()
    fake = torch.randn(1, 3, 256, 256).cuda()
    d = D_stylegan2(resolution=256, structure='resnet').cuda()
    # print(d(data).shape)
    # https://discuss.pytorch.org/t/one-of-the-differentiated-tensors-does-not-require-grad/54694
    D_logistic_r1(fake, data, d)

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


