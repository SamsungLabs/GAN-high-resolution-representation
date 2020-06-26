import math
import torch
import torch.nn as nn
import numpy as np
from lib.nn.layers.cuda.upfirdn2d import upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = epsilon

    def forward(self, x):
        y = torch.mean(torch.pow(x, 2), dim=1, keepdim=True)

        y = y + self.eps
        y = x * torch.rsqrt(y)
        return y


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Bias(nn.Module):
    def __init__(self, units):
        super(Bias, self).__init__()
        self._units = units
        self.bias = nn.Parameter(torch.Tensor(1, units, 1, 1))
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        y = x + self.bias
        return y


class AddNoise(nn.Module):
    def __init__(self, channels, fixed=False, per_channel=True):
        super(AddNoise, self).__init__()
        self.fixed = fixed
        self.fixed_noise = None
        scale_channels = channels if per_channel else 1
        self.scale_factors = nn.Parameter(torch.Tensor(1, scale_channels, 1, 1))
        nn.init.constant_(self.scale_factors, 0)

    def forward(self, x, noise=None):

        bs, _, h, w = x.size()

        if noise is None:
            if self.fixed:
                if self.fixed_noise is not None:
                    noise = self.fixed_noise
                else:
                    noise = torch.randn(1, 1, h, w, device=x.device)
                    self.fixed_noise = noise
                noise = noise.repeat(bs, 1, 1, 1)
            else:
                noise = torch.randn(bs, 1, h, w, device=x.device)

        noise_scaled = self.scale_factors * noise

        y = x + noise_scaled
        return y


class MinibatchStdLayerStylegan2(nn.Module):
    def __init__(self, group_size, num_new_features=1, eps=1e-8):
        super(MinibatchStdLayerStylegan2, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features
        self.eps = eps

    def forward(self, x):
        bs, c, h, w = x.size()
        y = x[None,:,:,:,:]                                         # [1NCHW]   input shape
        group_size = min(bs, self.group_size)
        n_feat = self.num_new_features
        new_shape = (group_size, -1, n_feat, c // n_feat, h, w)
        y = torch.reshape(y, shape=new_shape)                       # [GMncHW]  split minibatch into M groups of size G.
        y = y - torch.mean(y, 0, keepdim=True)                      # [GMncHW]  subtract mean over group.
        y = torch.mean(y**2, 0)                                     # [MncHW]   calc variance over group.
        y = torch.sqrt(y + self.eps)                                # [MncHW]   calc stddev over group.
        y = torch.mean(y, dim=(2, 3, 4), keepdim=True)              # [Mn111]   take average over fmaps and pixels.
        y = torch.mean(y, dim=(2,))                                 # [Mn11]    take average over fmaps and pixels.
        y = y.repeat((group_size, 1, h, w))                         # [N1HW]    replicate over group.

        return torch.cat((x, y), dim=1)


class Conv2dModulated(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size, kernel_size, padding=0,
                 demodulate=True, fused_modconv=True,
                 upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(padding, tuple):
            padding = (padding, padding)

        self.padding = padding
        self.kernel_size = kernel_size

        self.demodulate = demodulate
        self.fused_modconv = fused_modconv

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))

        self.affine = nn.Linear(latent_size, in_channels, bias=True)
        self.affine._wscale_params = {'gain': 1.0}

        self.upsample = upsample
        self.downsample = downsample

        if self.upsample:
            factor = 2
            assert kernel_size[0] == kernel_size[1]
            p = (len(blur_kernel) - factor) - (kernel_size[0] - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = BlurKernel(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if self.downsample:
            factor = 2
            assert kernel_size[0] == kernel_size[1]
            p = (len(blur_kernel) - factor) + (kernel_size[0] - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = BlurKernel(blur_kernel, pad=(pad0, pad1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, w):

        s = self.affine(w) + 1

        bs = w.shape[0]
        weight = self.weight
        weight_m = weight.unsqueeze(0) # [1OIkk]
        weight_m = s[:, np.newaxis, :, np.newaxis, np.newaxis] * weight_m # [BOIkk]

        d = None
        if self.demodulate:
            d = torch.rsqrt(torch.sum(weight_m ** 2, dim=(2, 3, 4)) + 1e-8) # [BO]
            weight_m = d[:, :, np.newaxis, np.newaxis, np.newaxis] * weight_m

        if not self.fused_modconv:
            x = s[:, :, np.newaxis, np.newaxis] * x  # x is [BIhw]

        if self.downsample:
            x = self.blur(x)

        if self.fused_modconv:
            weight = weight_m.view((-1, self.in_channels, self.kernel_size[0], self.kernel_size[1]))  # [(B*O)Ikk]
            x = x.reshape((1, x.size(0) * x.size(1), x.size(2), x.size(3)))

        if self.downsample:
            x = torch.conv2d(input=x, weight=weight, bias=None, stride=2, padding=0,
                             groups=bs if self.fused_modconv else 1)
        elif self.upsample:
            if self.fused_modconv:
                weight_m = weight_m.view(bs, self.out_channels, self.in_channels,
                                         self.kernel_size[0], self.kernel_size[1])
                weight = weight_m.transpose(1, 2).reshape(-1, self.out_channels,
                                                          self.kernel_size[0], self.kernel_size[1]) # [(B*I)Okk]

            x = torch.conv_transpose2d(input=x, weight=weight, padding=0, stride=2,
                                       groups=bs if self.fused_modconv else 1)
        else:
            x = torch.conv2d(input=x, weight=weight, bias=None, padding=self.padding,
                             groups=bs if self.fused_modconv else 1)

        if self.fused_modconv:
            x = torch.reshape(x, (-1, self.out_channels, x.size(2), x.size(3)))
        elif self.demodulate:
            x = d[:, :, np.newaxis, np.newaxis] * x # x is (batch_size, channels, height, width)

        if self.upsample:
            x = self.blur(x)

        return x


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class UpsampleKernel(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class BlurKernel(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out