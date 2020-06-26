import numpy as np
import torch
import torch.nn as nn
import math

from lib.nn.layers.ops import PixelNorm
from lib.nn.layers.ops import AddNoise, BlurKernel, Bias, Conv2dModulated, UpsampleKernel
from lib.nn.layers.cuda.fused_bias_act import FusedLeakyReLU
import torch.nn.functional as F


class Conv2dModulatedMod(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size, kernel_size, padding=0,
                 demodulate=True, fused_modconv=True, return_features_before_conv=False,
                 upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_features_before_conv = return_features_before_conv

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

        features = None
        if not self.fused_modconv:
            x = s[:, :, np.newaxis, np.newaxis] * x  # x is [BIhw]
            features = x

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

        if self.return_features_before_conv:
            return x, features
        else:
            return x


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * np.sqrt(2.)


class StyleGeneratorBlock(nn.Module):
    def __init__(self, conv_size, latent_size, in_channels=None, use_first_conv=False, upsample=False):
        super(StyleGeneratorBlock, self).__init__()

        in_channels = conv_size if in_channels is None else in_channels

        self.use_first_conv = use_first_conv

        if self.use_first_conv:
            self.conv1 = Conv2dModulated(in_channels, conv_size, latent_size, 3, 1, upsample=upsample)
            self.addnoise1 = AddNoise(conv_size, per_channel=False)
            self.bias_act1 = FusedLeakyReLU(conv_size, 0.2)

        self.conv2 = Conv2dModulated(conv_size, conv_size, latent_size, 3, 1)
        self.addnoise2 = AddNoise(conv_size, per_channel=False)
        self.bias_act2 = FusedLeakyReLU(conv_size, 0.2)

    def forward(self, x, w1, w2=None, noise1=None, noise2=None):

        y = x

        if self.use_first_conv:
            y = self.conv1(y, w1)
            y = self.addnoise1(y, noise1)
            y = self.bias_act1(y)

        w2 = w1 if w2 is None or not self.use_first_conv else w2
        noise2 = noise1 if not self.use_first_conv else noise2
        y = self.conv2(y, w2)
        y = self.addnoise2(y, noise2)
        y = self.bias_act2(y)
        return y


class ToRGB(nn.Module):
    def __init__(self, conv_size, latent_size, in_channels=None):
        super(ToRGB, self).__init__()

        in_channels = conv_size if in_channels is None else in_channels

        self.conv1 = Conv2dModulated(in_channels, conv_size, latent_size, 1, 0, demodulate=False)
        self.bias1 = Bias(conv_size)

    def forward(self, x, w):

        y = self.conv1(x, w)
        y = self.bias1(y)

        return y


class Generator(nn.Module):

    def __init__(self, max_res_log2, latent_size=512, fmap_base=8192, fmap_max=512,
                 base_scale_h=4, base_scale_w=4, channels=3, use_activation=False,
                 use_pn=True, label_size=0, mix_style=True, mix_prob=0.9):
        super(Generator, self).__init__()

        self.fmap_base = fmap_base
        self.fmap_decay = 1.0
        self.fmap_max = fmap_max

        self.base_scale_h = base_scale_h
        self.base_scale_w = base_scale_w

        self.nc = channels
        self.label_size = label_size
        self.latent_size = latent_size
        self.max_res_log2 = max_res_log2
        self.alpha = 1.0
        self.use_activation = use_activation
        self.use_pn = use_pn
        self.mix_style = mix_style
        self.mix_prob = mix_prob

        self.constant_tensor = nn.Parameter(torch.Tensor(1, self.num_features(1), self.base_scale_h, self.base_scale_w))
        nn.init.constant_(self.constant_tensor, 1)

        blocks = []
        to_rgbs = []
        for res_log2 in range(2, self.max_res_log2+1):
            blocks.append(self.build_block(res_log2))
            to_rgbs.append(self.build_to_rgb(res_log2))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgbs = nn.ModuleList(to_rgbs)

        self.upscale2x = self.build_upscale2x()
        self.mapping = self.build_mapping()

        self.conditional_embedding = None
        if self.label_size > 0:
            self.conditional_embedding = self.build_conditional_embedding()

    def build_mapping(self):

        layers = []
        in_units = self.latent_size
        if self.use_pn:
            layers.append(PixelNorm())
        for i in range(8):
            layers.append(nn.Linear(in_units, self.latent_size))
            in_units = self.latent_size
            layers.append(ScaledLeakyReLU(0.2))

        mapping = nn.Sequential(*layers)
        return mapping

    def build_upscale2x(self):
        upscale2x = UpsampleKernel(kernel=[1, 3, 3, 1])
        return upscale2x

    def num_features(self, res_log2):
        fmaps = int(self.fmap_base / (2.0 ** ((res_log2 - 1) * self.fmap_decay)))
        return min(fmaps, self.fmap_max)

    def build_to_rgb(self, res_log2):
        conv_size = self.num_features(res_log2)
        return ToRGB(self.nc, self.latent_size, in_channels=conv_size)

    def build_block(self, res_log2):
        conv_size = self.num_features(res_log2)
        in_channels = self.num_features(res_log2 - 1)

        if res_log2 == 2:
            net_block = StyleGeneratorBlock(conv_size, self.latent_size,
                                            use_first_conv=False, in_channels=in_channels,
                                            upsample=False)
        else:
            net_block = StyleGeneratorBlock(conv_size, self.latent_size,
                                            use_first_conv=True, in_channels=in_channels,
                                            upsample=True)
        return net_block

    def build_conditional_embedding(self):
        embedding = nn.Embedding(self.label_size, self.latent_size)
        return embedding

    def run_style_mixing(self, w):
        if self.mix_style and self.training:
            w_rev = torch.flip(w, dims=(0,))
            cur_prob = np.random.uniform(0., 1.)
            if cur_prob < self.mix_prob:
                t = np.random.randint(1, 2 * self.max_res_log2 - 2)
                w = [w] * (2*self.max_res_log2 - 2 - t) + [w_rev] * t
            else:
                w = [w] * (2*self.max_res_log2 - 2)
        else:
            w = [w] * (2*self.max_res_log2 - 2)
        return w

    def run_trunc(self, w, latent_avg, trunc_psi=0.7, trunc_cutoff=8):
        if latent_avg is not None and not self.training:
            w_trunc = []
            trunc_cutoff = len(w) if trunc_cutoff is None else trunc_cutoff
            tpsi = [trunc_psi] * trunc_cutoff
            truncation_psi = [1] * len(w)
            truncation_psi = tpsi + truncation_psi[len(tpsi):]

            for i, w_i in enumerate(w):
                w_trunc.append(w_i * truncation_psi[i] + (1 - truncation_psi[i]) * latent_avg)
            w = w_trunc
        return w

    def run_conditional(self, w, label):
        if self.label_size > 0:
            label = label.view((-1,)).detach()
            w_class = self.conditional_embedding(label)
            w = w + w_class
        return w

    def forward(self, noise, label=None, latent_avg=None, latents_only=False,
                input_is_latent=False, addnoise=None, trunc_psi=0.5, trunc_cutoff=None):

        if input_is_latent:
            w = noise
        else:
            w = self.mapping(noise)
            w = self.run_conditional(w, label)
            w = self.run_style_mixing(w)
            w = self.run_trunc(w, latent_avg, trunc_psi, trunc_cutoff)

        if latents_only:
            return w

        n = w[0].size(0)
        ct = self.constant_tensor
        ct = ct.expand(n, -1, -1, -1)

        noise1 = addnoise[0] if addnoise is not None else None
        features = []
        x = self.blocks[0](ct, w[0], noise1=noise1)
        y = self.to_rgbs[0](x, w[1])
        features.append(x)


        for res in range(3, self.max_res_log2 + 1):
            noise1 = addnoise[2*res-5] if addnoise is not None else None
            noise2 = addnoise[2*res-4] if addnoise is not None else None

            x = self.blocks[res-2](x, w[2*res-5], w[2*res-4], noise1, noise2)
            y0 = self.to_rgbs[res-2](x, w[2*res-3])
            features.append(x)
            y = self.upscale2x(y) + y0 if y is not None else y0

        return y, features


