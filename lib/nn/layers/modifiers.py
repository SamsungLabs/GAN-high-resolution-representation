import types

import numpy as np
from torch import nn as nn
from torch.nn.functional import normalize
from torch.nn.utils import spectral_norm
import torch


def make_one_hot(ids, num_classes):
    ids = ids.view((-1,))
    one_hot = torch.zeros(ids.size(0), num_classes, device=ids.device)
    ids = one_hot.scatter_(1, ids.view(-1, 1), 1)
    return ids


def apply_wscale(net, gain=np.sqrt(2), weight_name='weight', fan_in=None):
    def apply_wscale_(block):
        wscale_ops_dim = {
            'Linear': 0, 'Conv1d': 0, 'Conv2d': 0, 'Conv3d': 0,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 1, 'ConvTranspose3d': 1,
            'Conv2dModulated': 0, 'Conv2dModulatedMod': 0,
        }

        def patch_repr(block):
            original_repr = block.extra_repr

            local_fan_in = fan_in
            local_gain = gain
            if hasattr(block, '_wscale_params'):
                _wscale_params = block._wscale_params
                local_gain = _wscale_params.get('gain', local_gain)
                local_fan_in = _wscale_params.get('fan_in', local_fan_in)

            def new_repr(self):
                repr_str = f'{original_repr()} + ({weight_name}: use wscale, gain={local_gain:.2f})'
                if local_fan_in is not None:
                    repr_str += f', fan_in=({local_fan_in})'
                return repr_str

            return types.MethodType(new_repr, block)

        def get_preprocess_method(block, dim):

            preprocess_orig = None
            if hasattr(block, f'preprocess_{weight_name}'):
                preprocess_orig = getattr(block, f'preprocess_{weight_name}')

            local_fan_in = fan_in
            local_gain = gain
            if hasattr(block, '_wscale_params'):
                _wscale_params = block._wscale_params
                local_gain = _wscale_params.get('gain', local_gain)
                local_fan_in = _wscale_params.get('fan_in', local_fan_in)

            block._fan_in = local_fan_in

            def preprocess(self, w):
                if self._fan_in is None:
                    self._fan_in = np.prod(w.shape[:dim] + w.shape[dim + 1:])
                std = local_gain / np.sqrt(self._fan_in)
                w_new = w * std
                if preprocess_orig is not None:
                    w_new = preprocess_orig(w_new)
                return w_new

            return types.MethodType(preprocess, block)

        def forward_pre_hook(module, input):
            weight = getattr(module, weight_name + '_orig')
            weight_new = getattr(block, f'preprocess_{weight_name}')(weight)
            setattr(module, weight_name, weight_new)

        block_type = type(block).__name__
        if block_type in wscale_ops_dim:
            if getattr(block, '__wscale', False):
                return block

            if not hasattr(block, weight_name + '_orig'):
                if not hasattr(block, weight_name):
                    return block
                weight = getattr(block, weight_name)
                del block._parameters[weight_name]
                block.register_parameter(weight_name + '_orig', nn.Parameter(weight.data))
                setattr(block, weight_name, weight.data)
                preprocess = get_preprocess_method(block, wscale_ops_dim[block_type])
                block.register_forward_pre_hook(forward_pre_hook)
            else:
                preprocess = get_preprocess_method(block, wscale_ops_dim[block_type])

            setattr(block, f'preprocess_{weight_name}', preprocess)

            block.extra_repr = patch_repr(block)
            block.__wscale = True

        return block

    return net.apply(apply_wscale_)


def apply_wscale2(net, weight_name='weight'):
    def apply_wscale_(block):
        wscale_ops_dim = {
            'Linear': 0, 'Conv1d': 0, 'Conv2d': 0, 'Conv3d': 0,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 1, 'ConvTranspose3d': 1,
            'Conv2dModulated': 0,
        }

        def patch_repr(block, weight_scale):
            original_repr = block.extra_repr

            def new_repr(self):
                return f'{original_repr()} + ({weight_name}: use wscale, scale={weight_scale:.4f})'

            return types.MethodType(new_repr, block)

        def get_preprocess_method(block, weight_scale):

            preprocess_orig = None
            if hasattr(block, f'preprocess_{weight_name}'):
                preprocess_orig = getattr(block, f'preprocess_{weight_name}')

            def preprocess(self, w):
                w_new = w * weight_scale
                if preprocess_orig is not None:
                    w_new = preprocess_orig(w_new)
                return w_new

            return types.MethodType(preprocess, block)

        def forward_pre_hook(module, input):
            weight = getattr(module, weight_name + '_orig')
            weight_new = getattr(block, f'preprocess_{weight_name}')(weight)
            setattr(module, weight_name, weight_new)

        block_type = type(block).__name__
        if block_type in wscale_ops_dim:
            if getattr(block, '__wscale', False):
                return block

            weight = getattr(block, weight_name)
            with torch.no_grad():
                weight_std = weight.data.std()
                nn.init.normal_(weight.data, 0, std=1.0)

            if not hasattr(block, weight_name + '_orig'):
                if not hasattr(block, weight_name):
                    return block

                del block._parameters[weight_name]
                block.register_parameter(weight_name + '_orig', nn.Parameter(weight.data))
                setattr(block, weight_name, weight.data)
                preprocess = get_preprocess_method(block, weight_std)
                block.register_forward_pre_hook(forward_pre_hook)
            else:
                preprocess = get_preprocess_method(block, weight_std)

            setattr(block, f'preprocess_{weight_name}', preprocess)

            block.extra_repr = patch_repr(block, weight_std)
            block.__wscale = True

        return block

    return net.apply(apply_wscale_)


def apply_weight_averaging(net, weight_name='weight'):
    def apply_wscale_(block):
        ops_dim = {
            'Linear': 0, 'Conv1d': 0, 'Conv2d': 0, 'Conv3d': 0,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 1, 'ConvTranspose3d': 1
        }
        kernel_dims = {
            'Linear': 0, 'Conv1d': 1, 'Conv2d': 2, 'Conv3d': 3,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 2, 'ConvTranspose3d': 3
        }

        def patch_repr(block):
            original_repr = block.extra_repr

            def new_repr(self):
                return f'{original_repr()} + ({weight_name}: use weight averaging)'

            return types.MethodType(new_repr, block)

        def get_preprocess_method(block, dim, kernel_dims):

            preprocess_orig = None
            if hasattr(block, f'preprocess_{weight_name}'):
                preprocess_orig = getattr(block, f'preprocess_{weight_name}')

            if kernel_dims == 2:
                def preprocess(self, w):
                    # pad last two dimensions with zeros on both sides (width and height), pad=1
                    w = torch.nn.functional.pad(w, [1, 1, 1, 1])
                    w_new = 0.25 * (w[:,:,1:,1:] + w[:,:,:-1,1:] + w[:,:,1:,:-1] + w[:,:,:-1,:-1])
                    if preprocess_orig is not None:
                        w_new = preprocess_orig(w_new)
                    return w_new
            elif kernel_dims == 1:
                def preprocess(self, w):
                    # pad last two dimensions with zeros on both sides (width and height), pad=1
                    w = torch.nn.functional.pad(w, [1, 1])
                    w_new = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
                    if preprocess_orig is not None:
                        w_new = preprocess_orig(w_new)
                    return w_new
            else:
                raise NotImplementedError

            return types.MethodType(preprocess, block)

        def forward_pre_hook(module, input):
            weight = getattr(module, weight_name + '_orig')
            weight_new = getattr(block, f'preprocess_{weight_name}')(weight)
            setattr(module, weight_name, weight_new)

        block_type = type(block).__name__
        if block_type in ops_dim:
            if getattr(block, '__weight_averaging', False):
                return block

            if not hasattr(block, weight_name + '_orig'):
                if not hasattr(block, weight_name):
                    return block
                weight = getattr(block, weight_name)
                del block._parameters[weight_name]
                block.register_parameter(weight_name + '_orig', nn.Parameter(weight.data))
                setattr(block, weight_name, weight.data)
                preprocess = get_preprocess_method(block, ops_dim[block_type], kernel_dims[block_type])
                block.register_forward_pre_hook(forward_pre_hook)
            else:
                preprocess = get_preprocess_method(block, ops_dim[block_type], kernel_dims[block_type])

            setattr(block, f'preprocess_{weight_name}', preprocess)

            block.extra_repr = patch_repr(block)
            block.__weight_averaging = True

        return block

    return net.apply(apply_wscale_)


def apply_lr_mult(net, lr_mult=1.0, weight_name='weight'):
    def apply_lr_mult_(block):

        wscale_ops_dim = {
            'Linear': 0, 'Conv1d': 0, 'Conv2d': 0, 'Conv3d': 0,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 1, 'ConvTranspose3d': 1,
            'Conv2dModulated': 0
        }

        def patch_repr(block, lr_mult_real):
            original_repr = block.extra_repr

            def new_repr(self):
                return f'{original_repr()} + ({weight_name}: lr_mult={lr_mult_real:.3f})'

            return types.MethodType(new_repr, block)

        def get_preprocess_method(block):

            preprocess_orig = None
            if hasattr(block, f'preprocess_{weight_name}'):
                preprocess_orig = getattr(block, f'preprocess_{weight_name}')

            local_lr_mult = lr_mult
            if hasattr(block, 'lr_mult'):
                local_lr_mult = getattr(block, 'lr_mult')

            def preprocess(self, w):
                w_new = w * local_lr_mult
                if preprocess_orig is not None:
                    w_new = preprocess_orig(w_new)
                return w_new

            return types.MethodType(preprocess, block), local_lr_mult

        def forward_pre_hook(module, input):
            weight = getattr(module, weight_name + '_orig')
            weight_new = getattr(block, f'preprocess_{weight_name}')(weight)
            setattr(module, weight_name, weight_new)

        block_type = type(block).__name__
        if block_type in wscale_ops_dim:
            if getattr(block, f'__use_lr_mult{weight_name}', False):
                return block

            if not hasattr(block, weight_name + '_orig'):
                if not hasattr(block, weight_name):
                    return block
                weight = getattr(block, weight_name)
                del block._parameters[weight_name]
                block.register_parameter(weight_name + '_orig', nn.Parameter(weight.data))
                setattr(block, weight_name, weight.data)
                preprocess, lr_mult_real = get_preprocess_method(block)
                block.register_forward_pre_hook(forward_pre_hook)
            else:
                preprocess, lr_mult_real = get_preprocess_method(block)

            setattr(block, f'preprocess_{weight_name}', preprocess)
            setattr(block, f'lr_mult_{weight_name}', lr_mult_real)

            block.extra_repr = patch_repr(block, lr_mult_real)
            setattr(block, f'__use_lr_mult{weight_name}', True)

        return block

    return net.apply(apply_lr_mult_)


def apply_spectral_norm(net, weight_name='weight'):
    def apply_sn_(block):
        wscale_ops_dim = {
            'Linear': 0, 'Conv1d': 0, 'Conv2d': 0, 'Conv3d': 0,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 1, 'ConvTranspose3d': 1
        }

        def patch_repr(block):
            original_repr = block.extra_repr

            def new_repr(self):
                return f'{original_repr()} + ({weight_name}: use spectral norm)'

            return types.MethodType(new_repr, block)

        block_type = type(block).__name__
        if block_type in wscale_ops_dim:
            block = spectral_norm(block, name=weight_name, dim=wscale_ops_dim[block_type])

            block.extra_repr = patch_repr(block)
            block.__spectral_norm = True

        return block

    return net.apply(apply_sn_)


def apply_spectral_norm_diff(net, weight_name='weight'):
    def apply_sn_(block):
        wscale_ops_dim = {
            'Linear': 0, 'Conv1d': 0, 'Conv2d': 0, 'Conv3d': 0,
            'ConvTranspose1d': 1, 'ConvTranspose2d': 1, 'ConvTranspose3d': 1
        }

        def patch_repr(block):
            original_repr = block.extra_repr

            def new_repr(self):
                return f'{original_repr()} + ({weight_name}: use diff spectral norm)'

            return types.MethodType(new_repr, block)

        def get_preprocess_method(block, dim):

            preprocess_orig = None
            if hasattr(block, f'preprocess_{weight_name}'):
                preprocess_orig = getattr(block, f'preprocess_{weight_name}')

            def preprocess(self, weight):

                u = getattr(block, weight_name + '_u')
                v = getattr(block, weight_name + '_v')

                weight_mat = weight
                if dim != 0:
                    weight_mat = weight_mat.permute(dim, *[d for d in range(weight_mat.dim()) if d != dim])
                height = weight_mat.size(0)
                weight_mat = weight_mat.reshape(height, -1)

                v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=1e-12)
                u = normalize(torch.mv(weight_mat, v), dim=0, eps=1e-12)

                sigma = torch.dot(u, torch.mv(weight_mat, v))
                w_new = weight / sigma

                if preprocess_orig is not None:
                    w_new = preprocess_orig(w_new)
                return w_new

            return types.MethodType(preprocess, block)

        def forward_pre_hook(module, input):
            weight = getattr(module, weight_name + '_orig')
            weight_new = getattr(block, f'preprocess_{weight_name}')(weight)
            setattr(module, weight_name, weight_new)

        block_type = type(block).__name__
        if block_type in wscale_ops_dim:
            if getattr(block, '__spectral_norm', False):
                return block

            if not hasattr(block, weight_name + '_orig'):
                if not hasattr(block, weight_name):
                    return block
                weight = getattr(block, weight_name)
                del block._parameters[weight_name]
                block.register_parameter(weight_name + '_orig', nn.Parameter(weight.data))
                setattr(block, weight_name, weight.data)
                block.register_forward_pre_hook(forward_pre_hook)
                weight = weight.data
            else:
                weight = getattr(block, weight_name)

            weight_mat = weight
            dim = wscale_ops_dim[block_type]
            if dim != 0:
                weight_mat = weight_mat.permute(dim, *[d for d in range(weight_mat.dim()) if d != dim])
            height = weight_mat.size(0)
            weight_mat = weight_mat.reshape(height, -1)

            h, w = weight_mat.size()
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=1e-12)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=1e-12)
            block.register_parameter(weight_name + '_u', nn.Parameter(u))
            block.register_parameter(weight_name + '_v', nn.Parameter(v))

            preprocess = get_preprocess_method(block, wscale_ops_dim[block_type])

            setattr(block, f'preprocess_{weight_name}', preprocess)

            block.extra_repr = patch_repr(block)
            block.__spectral_norm = True

        return block

    return net.apply(apply_sn_)
