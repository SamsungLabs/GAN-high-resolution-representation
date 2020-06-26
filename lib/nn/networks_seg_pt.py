import torch
import torch.nn as nn
import numpy as np


class DecoderResBlock(nn.Module):
    def __init__(self, conv_size, use_bn, in_c=None):
        super(DecoderResBlock, self).__init__()

        in_c = conv_size if in_c is None else in_c

        net_block = []
        net_block.append(nn.Conv2d(in_c, conv_size, 3, 1, 1, bias=True))
        if use_bn:
            net_block.append(nn.BatchNorm2d(conv_size))
        net_block.append(nn.LeakyReLU(0.2))

        net_block.append(nn.Conv2d(conv_size, conv_size, 3, 1, 1, bias=True))
        if use_bn:
            net_block.append(nn.BatchNorm2d(conv_size))
        net_block.append(nn.LeakyReLU(0.2))
        self.base_layers = nn.Sequential(*net_block)

        if conv_size != in_c:
            self.shortcut = nn.Conv2d(in_c, conv_size, 1, 1, padding=0, bias=True)
        else:
            self.shortcut = None

    def forward(self, x):
        y = self.base_layers(x)
        sc = x if self.shortcut is None else self.shortcut(x)
        return (sc + y) / np.sqrt(2)


class Decoder(nn.Module):

    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self._features = cfg['features']
        self._in_channels = cfg['in_channels']
        self._start_res = cfg['start_res']
        self._num_feats = len(self._in_channels)

        use_bn = cfg['use_bn']
        use_dropout = cfg['use_dropout']

        cvt_blocks = nn.ModuleDict()
        for i in range(self._start_res, self._num_feats):
            conv_size = self._features[i]
            in_c = self._in_channels[i]
            cvt_block = []
            cvt_block.append(nn.Conv2d(in_c, conv_size, 3, 1, 1, bias=True))
            if use_bn:
                cvt_block.append(nn.BatchNorm2d(conv_size))

            cvt_block.append(nn.LeakyReLU(0.2))
            if use_dropout:
                cvt_block.append(nn.Dropout(0.5))

            cvt_blocks[f'cvt_block_{i}'] = nn.Sequential(*cvt_block)
        self.cvt_blocks = cvt_blocks

        main_blocks = nn.ModuleDict()
        for i in range(self._start_res, self._num_feats):
            conv_size = self._features[i+1]
            in_c = self._features[i]
            in_c = 2 * in_c if i > self._start_res else in_c
            if i < self._num_feats - 1:
                net_block = []
                net_block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                net_block.append(DecoderResBlock(conv_size, use_bn, in_c))
            else:
                net_block = []
                net_block.append(nn.Conv2d(in_c, conv_size, 3, 1, 1, bias=True))

            main_blocks[f'main_block_{i}'] = nn.Sequential(*net_block)
        self.main_blocks = main_blocks


    def forward(self, inputs):

        prev = None
        pred = None

        for i in range(self._start_res, self._num_feats):
            cvt_block = self.cvt_blocks[f'cvt_block_{i}']
            input_i = inputs[i]
            if cvt_block is not None:
                input_i = cvt_block(input_i)

            if i > self._start_res:
                input_i = torch.cat([prev, input_i], dim=1)
            net_block = self.main_blocks[f'main_block_{i}']
            pred = net_block(input_i)
            prev = pred

        return pred