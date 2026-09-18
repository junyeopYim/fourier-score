"""Native PyTorch resampling, adapted from score_sde_pytorch and StyleGAN2.

No import-time CUDA compilation. See NOTICE and docs/MIGRATION.md.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0),
              max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)


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
    # Fixes an unused upstream branch: negative tensor slicing and 4D stride.
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
