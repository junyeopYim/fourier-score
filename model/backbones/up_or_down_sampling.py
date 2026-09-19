"""Native PyTorch FIR resampling (score_sde_pytorch / StyleGAN2 lineage).

Modified 2026-09-19: zero insertion uses only 4D padding for MPS.
Same filters, padding, convolutions and trainable parameters as the source.
No CUDA extension, no silent CPU fallback. See docs/PROVENANCE.md.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    B, C, H, W = input.shape
    out = input.reshape(B * C, H, W, 1)
    out = F.pad(out, (0, up_x - 1)).reshape(B, C, H, W * up_x)
    out = out.permute(0, 1, 3, 2).reshape(B * C, W * up_x, H, 1)
    out = F.pad(out, (0, up_y - 1)).reshape(B, C, W * up_x, H * up_y).permute(0, 1, 3, 2)
    out = F.pad(out, (max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)))
    out = out[:, :, max(-pad_y0, 0):out.shape[2] - max(-pad_y1, 0),
                    max(-pad_x0, 0):out.shape[3] - max(-pad_x1, 0)]
    out = F.conv2d(out.reshape(B * C, 1, out.shape[2], out.shape[3]),
                   kernel.flip((0, 1)).reshape(1, 1, *kernel.shape))
    out = out[:, :, ::down_y, ::down_x]
    return out.reshape(B, C, out.shape[2], out.shape[3])


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(x, kernel.to(x), up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k = k / np.sum(k)
    assert k.ndim == 2 and k.shape[0] == k.shape[1]
    return k


def naive_upsample_2d(x, factor=2):
    _, C, H, W = x.shape
    x = x.reshape(-1, C, H, 1, W, 1).repeat(1, 1, 1, factor, 1, factor)
    return x.reshape(-1, C, H * factor, W * factor)


def naive_downsample_2d(x, factor=2):
    _, C, H, W = x.shape
    return x.reshape(-1, C, H // factor, factor, W // factor, factor).mean(dim=(3, 5))


def upsample_2d(x, k=None, factor=2, gain=1):
    k = _setup_kernel([1] * factor if k is None else k) * (gain * factor ** 2)
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), up=factor,
                    pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
    k = _setup_kernel([1] * factor if k is None else k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2))


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    _, _, convH, convW = w.shape
    assert convW == convH
    k = _setup_kernel([1] * factor if k is None else k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=[factor, factor], padding=0)


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    outC, inC, convH, convW = w.shape
    assert convH == convW
    k = _setup_kernel([1] * factor if k is None else k) * (gain * factor ** 2)
    p = (k.shape[0] - factor) - (convW - 1)
    groups = x.shape[1] // inC
    w = w.reshape(groups, -1, inC, convH, convW)
    w = torch.flip(w, [-2, -1]).permute(0, 2, 1, 3, 4).reshape(groups * inC, -1, convH, convW)
    x = F.conv_transpose2d(x, w, stride=(factor, factor), groups=groups)
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                    pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
                 resample_kernel=(1, 3, 3, 1), use_bias=True, kernel_init=None):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        self.up, self.down = up, down
        self.resample_kernel, self.kernel, self.use_bias = resample_kernel, kernel, use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)
        return x + self.bias.reshape(1, -1, 1, 1) if self.use_bias else x
