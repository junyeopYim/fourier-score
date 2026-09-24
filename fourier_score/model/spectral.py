"""FFT and portable real-DFT implementations of the same spectral filter."""

import math
import torch
from torch import nn


def conjugate_symmetrize(power: torch.Tensor) -> torch.Tensor:
    h, w = power.shape[-2:]
    iy = (-torch.arange(h, device=power.device)) % h
    ix = (-torch.arange(w, device=power.device)) % w
    return 0.5 * (power + power.index_select(-2, iy).index_select(-1, ix))


class SpectralFilter(nn.Module):
    """Apply a real, conjugate-symmetric multiplier without severing autograd.

    auto: FFT on CPU/CUDA, separable real DFT matmul on MPS.
    matmul: no complex tensors and no hidden CPU transfers in forward/backward.
    cpu: explicitly transfer the FFT branch to CPU, preserving gradients.
    """

    def __init__(self, height: int, width: int, backend: str = "auto"):
        super().__init__()
        if backend not in ("auto", "fft", "matmul", "cpu"):
            raise ValueError(backend)
        self.backend = backend
        self.height = height
        self.width = width
        # Compute trigonometry in float64 once on CPU, then store FP32.
        for axis, size in (("h", height), ("w", width)):
            t = torch.arange(size, dtype=torch.float64)
            angle = (2 * math.pi / size) * t[:, None] * t[None, :]
            self.register_buffer(
                "cos_" + axis, (angle.cos() / math.sqrt(size)).float(), persistent=False
            )
            self.register_buffer(
                "sin_" + axis, (angle.sin() / math.sqrt(size)).float(), persistent=False
            )

    def resolved_backend(self, device):
        return (
            ("matmul" if device.type == "mps" else "fft")
            if self.backend == "auto"
            else self.backend
        )

    def forward(self, x: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError("Spectral shape mismatch")
        method = self.resolved_backend(x.device)
        if method in ("fft", "cpu"):
            work = x.to("cpu") if method == "cpu" else x
            # .to() is differentiable; intentionally never detach here.
            weight = multiplier.to(work.device, dtype=work.dtype)
            out = torch.fft.ifft2(
                torch.fft.fft2(work, norm="ortho") * weight, norm="ortho"
            ).real
            return out.to(x.device)
        ch, sh = self.cos_h.to(x), self.sin_h.to(x)
        cw, sw = self.cos_w.to(x), self.sin_w.to(x)
        # (Ch-iSh) x (Cw-iSw)^T, then the real part of its inverse.
        hr = ch @ x
        hi = -(sh @ x)
        real = (hr @ cw.T + hi @ sw.T) * multiplier
        imag = (hi @ cw.T - hr @ sw.T) * multiplier
        ar = ch.T @ real - sh.T @ imag
        ai = ch.T @ imag + sh.T @ real
        return ar @ cw - ai @ sw
