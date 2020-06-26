import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class Initializer(object):
    def __init__(self, local_init=True, gamma=None):
        self.local_init = local_init
        self.gamma = gamma

    def __call__(self, m):
        if getattr(m, '__initialized', False):
            return

        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                          nn.GroupNorm)):
            if m.weight is not None:
                self._init_gamma(m.weight.data)
            if m.bias is not None:
                self._init_beta(m.bias.data)
        else:
            if getattr(m, 'weight', None) is not None:
                self._init_weight(m.weight.data)
            if getattr(m, 'bias', None) is not None:
                self._init_bias(m.bias.data)

        if self.local_init:
            object.__setattr__(m, '__initialized', True)

    def _init_weight(self, data):
        nn.init.uniform_(data, -0.07, 0.07)

    def _init_bias(self, data):
        nn.init.constant_(data, 0)

    def _init_gamma(self, data):
        if self.gamma is None:
            nn.init.constant_(data, 1.0)
        else:
            nn.init.normal_(data, 1.0, self.gamma)

    def _init_beta(self, data):
        nn.init.constant_(data, 0)


class Xavier(Initializer):
    def __init__(self, rnd_type='uniform', magnitude=3.0, **kwargs):
        super().__init__(**kwargs)

        self.rnd_type = rnd_type
        self.magnitude = magnitude

    def _init_weight(self, data):
        if self.rnd_type == 'uniform':
            gain = np.sqrt(self.magnitude / 3.0)
            nn.init.xavier_uniform_(data, gain=gain)
        elif self.rnd_type == 'normal':
            gain = np.sqrt(self.magnitude)
            nn.init.xavier_normal_(data, gain=gain)
        else:
            raise NotImplementedError


class Constant(Initializer):
    def __init__(self, value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def _init_weight(self, data):
        nn.init.constant_(data, self.value)


class XavierGluon(Initializer):
    def __init__(self, rnd_type='uniform', factor_type='avg', magnitude=3,
                 nonlinearity='leaky_relu', a=0.2, **kwargs):
        super().__init__(**kwargs)

        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)
        self.nonlinearity = nonlinearity
        self.a = a

    def _init_weight(self, arr):

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(arr)
        gain = nn.init.calculate_gain(self.nonlinearity, self.a)

        if self.factor_type == 'avg':
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == 'in':
            factor = fan_in
        elif self.factor_type == 'out':
            factor = fan_out
        else:
            raise ValueError('Incorrect factor type')
        scale = gain * np.sqrt(self.magnitude / factor)

        if self.rnd_type == 'uniform':
            nn.init.uniform_(arr, -scale, scale)
        elif self.rnd_type == 'gaussian':
            nn.init.normal_(arr, 0, scale)
        else:
            raise ValueError('Unknown random type')


class Normal(Initializer):
    def __init__(self, sigma=0.01, **kwargs):
        super().__init__(**kwargs)

        self.sigma = sigma

    def _init_weight(self, data):
        nn.init.normal_(data, 0, std=self.sigma)