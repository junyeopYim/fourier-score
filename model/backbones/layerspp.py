# Copyright 2020 The Google Research Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
# NCSN++ layers, retained with formatting changes and a keyword-only bug fix.
from . import layers, up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        if self.method == 'sum':
            return h + y
        raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(min(channels // 4, 32), channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q, k, v = self.NIN_0(h), self.NIN_1(h), self.NIN_2(h)
        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        return (x + h) / np.sqrt(2.) if self.skip_rescale else x + h


class Upsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if with_conv:
            if fir:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch, kernel=3, up=True,
                    resample_kernel=fir_kernel, use_bias=True, kernel_init=default_init())
            else:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        self.fir, self.with_conv, self.fir_kernel, self.out_ch = fir, with_conv, fir_kernel, out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            # Original third positional argument was scale_factor, not mode.
            h = F.interpolate(x, (H * 2, W * 2), mode='nearest')
            return self.Conv_0(h) if self.with_conv else h
        return self.Conv2d_0(x) if self.with_conv else up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)


class Downsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if with_conv:
            if fir:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch, kernel=3, down=True,
                    resample_kernel=fir_kernel, use_bias=True, kernel_init=default_init())
            else:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        self.fir, self.fir_kernel, self.with_conv, self.out_ch = fir, fir_kernel, with_conv, out_ch

    def forward(self, x):
        if not self.fir:
            return self.Conv_0(F.pad(x, (0, 1, 0, 1))) if self.with_conv else F.avg_pool2d(x, 2, stride=2)
        return self.Conv2d_0(x) if self.with_conv else up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)


class ResnetBlockDDPMpp(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
                 dropout=0.1, skip_rescale=False, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(min(in_ch // 4, 32), in_ch, eps=1e-6)
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(min(out_ch // 4, 32), out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.skip_rescale, self.act = skip_rescale, act
        self.out_ch, self.conv_shortcut = out_ch, conv_shortcut

    def forward(self, x, temb=None):
        h = self.Conv_0(self.act(self.GroupNorm_0(x)))
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.Conv_1(self.Dropout_0(self.act(self.GroupNorm_1(h))))
        if x.shape[1] != self.out_ch:
            x = self.Conv_2(x) if self.conv_shortcut else self.NIN_0(x)
        return (x + h) / np.sqrt(2.) if self.skip_rescale else x + h


class ResnetBlockBigGANpp(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1), skip_rescale=True, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(min(in_ch // 4, 32), in_ch, eps=1e-6)
        self.up, self.down, self.fir, self.fir_kernel = up, down, fir, fir_kernel
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(min(out_ch // 4, 32), out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.skip_rescale, self.act, self.in_ch, self.out_ch = skip_rescale, act, in_ch, out_ch

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.Conv_1(self.Dropout_0(self.act(self.GroupNorm_1(h))))
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        return (x + h) / np.sqrt(2.) if self.skip_rescale else x + h
