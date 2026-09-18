# Copyright 2020 The Google Research Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
# Modified: only the primitives needed by NCSN++ / DDPM++ are retained.
import math
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(config):
    name = config.model.nonlinearity.lower()
    acts = {'elu': nn.ELU, 'relu': nn.ReLU, 'lrelu': lambda: nn.LeakyReLU(0.2), 'swish': nn.SiLU}
    if name not in acts:
        raise ValueError(f'Unknown activation: {name}')
    return acts[name]()


def variance_scaling(scale, mode, distribution, in_axis=1, out_axis=0,
                     dtype=torch.float32, device='cpu'):
    def init(shape, dtype=dtype, device=device):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in, fan_out = shape[in_axis] * receptive_field_size, shape[out_axis] * receptive_field_size
        denominators = {'fan_in': fan_in, 'fan_out': fan_out, 'fan_avg': (fan_in + fan_out) / 2}
        if mode not in denominators:
            raise ValueError(f'Invalid mode: {mode}')
        variance = scale / denominators[mode]
        if distribution == 'normal':
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        if distribution == 'uniform':
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        raise ValueError(f'Invalid distribution: {distribution}')
    return init


def default_init(scale=1.):
    return variance_scaling(1e-10 if scale == 0 else scale, 'fan_avg', 'uniform')


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def contract_inner(x, y):
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]
    out_chars = x_chars[:-1] + y_chars[1:]
    return torch.einsum(f"{''.join(x_chars)},{''.join(y_chars)}->{''.join(out_chars)}", x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)
