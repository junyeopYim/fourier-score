"""Fixed Gaussian references from score_sde_pytorch/mnist_compare/core.py.

VE only: references return a score, not epsilon or sigma * score.
Full orthonormal FFT power is a covariance eigenvalue; no factor 1/2.
Original implementation: junyeopYim/score_sde_pytorch @ 534c75478bf3.
"""
from __future__ import annotations
from typing import Iterable
import math
import torch
from torch import nn


def to_model_space(images: torch.Tensor, dataset: str = 'mnist',
                   centered: bool | None = None) -> torch.Tensor:
    dataset = dataset.lower()
    if dataset == 'mnist':
        if images.ndim != 3 or tuple(images.shape[-2:]) != (28, 28):
            raise ValueError(f"Expected [B,28,28], got {tuple(images.shape)}")
        x = images.to(torch.float32).unsqueeze(1) / 255.0
        x = torch.nn.functional.pad(x, (2, 2, 2, 2))
    elif dataset == 'cifar10':
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError(f"Expected [B,3,32,32], got {tuple(images.shape)}")
        x = images.to(torch.float32) / 255.0
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    if centered is None:
        centered = dataset == 'mnist'
    return 2.0 * x - 1.0 if centered else x


@torch.no_grad()
def estimate_stats(batches: Iterable[torch.Tensor], floor: float = 1e-4) -> dict:
    """Training-only float64 sufficient statistics; one spectrum per channel."""
    if floor <= 0:
        raise ValueError("floor must be positive")
    n = 0
    sum_x = sum_power = sum_x2 = None
    for batch in batches:
        x = batch.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 4 or min(x.shape) < 1 or not torch.isfinite(x).all():
            raise ValueError("Expected nonempty finite BCHW batches")
        if sum_x is None:
            sum_x = torch.zeros_like(x[0])
            sum_power = torch.zeros_like(x[0])
            sum_x2 = torch.zeros((), dtype=torch.float64)
        elif x.shape[1:] != sum_x.shape:
            raise ValueError("All samples must have the same shape")
        n += len(x)
        sum_x += x.sum(0)
        sum_power += torch.fft.fft2(x, norm="ortho").abs().square().sum(0)
        sum_x2 += x.square().sum()
    if n < 2:
        raise ValueError("At least two training samples are required")
    mean = sum_x / n
    power = (sum_power / n - torch.fft.fft2(mean, norm="ortho").abs().square()).clamp_min(0)
    h, w = mean.shape[-2:]
    iy, ix = (-torch.arange(h)) % h, (-torch.arange(w)) % w
    power = 0.5 * (power + power.index_select(-2, iy).index_select(-1, ix))
    isotropic = ((sum_x2 / n - mean.square().sum()) / mean.numel()).clamp_min(floor)
    return {"mean": mean.float(), "power": power.clamp_min(floor).float(),
            "isotropic_variance": isotropic.float(), "n_train": n, "floor": floor}


class GaussianReference(nn.Module):
    """s_G(x,sigma)=-(Sigma+sigma^2 I)^(-1)(x-m)."""
    def __init__(self, stats: dict, mode: str = "spectral"):
        super().__init__()
        if mode not in {"spectral", "isotropic"}:
            raise ValueError(mode)
        self.mode = mode
        self.register_buffer("mean", stats["mean"].detach().clone())
        self.register_buffer("power", stats["power"].detach().clone())
        self.register_buffer("isotropic_variance", torch.as_tensor(stats["isotropic_variance"]).clone())
        if self.mean.ndim != 3 or min(self.mean.shape) < 1 or self.mean.shape != self.power.shape:
            raise ValueError("mean and power must both be [C,H,W]")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.power).all():
            raise ValueError("Non-finite Gaussian statistics")
        if torch.any(self.power <= 0) or self.isotropic_variance <= 0:
            raise ValueError("Covariance must be positive definite after flooring")
        h, w = self.power.shape[-2:]
        iy = (-torch.arange(h, device=self.power.device)) % h
        ix = (-torch.arange(w, device=self.power.device)) % w
        if not torch.allclose(self.power, self.power.index_select(-2, iy).index_select(-1, ix), atol=1e-6):
            raise ValueError("Power must have conjugate-frequency symmetry")

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != self.mean.shape or sigma.shape != (x.shape[0],):
            raise ValueError("Expected x=[B,C,H,W], sigma=[B]")
        var = sigma[:, None, None, None].square()
        centered = x - self.mean[None]
        if self.mode == "isotropic":
            return -centered / (self.isotropic_variance + var)
        fx = torch.fft.fft2(centered, norm="ortho")
        return -torch.fft.ifft2(fx / (self.power[None] + var), norm="ortho").real


class ResidualScore(nn.Module):
    """Backbone output is already divided by sigma in continuous VE."""
    def __init__(self, backbone: nn.Module, reference: GaussianReference | None):
        super().__init__()
        self.backbone = backbone
        self.reference = reference

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        residual = self.backbone(x, sigma)
        return residual if self.reference is None else residual + self.reference(x, sigma)


REFERENCE_MODES = (
    "baseline", "spectral", "isotropic", "spectral_linear", "isotropic_linear",
    "spectral_mmse", "isotropic_mmse",
)


class LinearTimeGaussianReference(GaussianReference):
    """lambda(t)=t for the geometric VE schedule, t=0 low / t=1 high noise."""
    def __init__(self, stats: dict, mode: str, *, sigma_min: float, sigma_max: float):
        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)
        if not (math.isfinite(sigma_min) and math.isfinite(sigma_max)
                and 0.0 < sigma_min < sigma_max):
            raise ValueError("Expected finite 0 < sigma_min < sigma_max")
        super().__init__(stats, mode)
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_range = math.log(sigma_max) - math.log(sigma_min)

    def noise_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        t = (torch.log(sigma) - self.log_sigma_min) / self.log_sigma_range
        return t.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        score_g = super().forward(x, sigma)
        lam = self.noise_weight(sigma)
        return lam[:, None, None, None] * score_g


class MMSEGaussianReference(GaussianReference):
    """a_G = sum_k P_k*sigma^2/(P_k+sigma^2) / sum_k P_k.

    FULL FFT spectrum; one scalar per image, not a per-frequency gate.
    No endpoint normalization, trainable parameters or detached score.
    """
    def noise_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim != 1 or sigma.numel() == 0 or not sigma.is_floating_point():
            raise ValueError("Expected a nonempty floating sigma tensor of shape [B]")
        if sigma.device != self.power.device:
            raise ValueError("sigma and Gaussian reference must be on the same device")
        dtype = (torch.float64 if torch.float64 in (sigma.dtype, self.power.dtype)
                 else torch.float32)
        variance = sigma.to(dtype=dtype).square()
        if self.mode == "isotropic":
            prior_variance = self.isotropic_variance.to(dtype=dtype)
            return (variance / (prior_variance + variance)).clamp(0.0, 1.0)
        power = self.power.to(dtype=dtype).reshape(-1)
        total_power = power.sum()
        sigma_batch = max(1, (1 << 20) // power.numel())
        weights = []
        for part in variance.split(sigma_batch):
            v = part[:, None]
            posterior_variance = power[None, :] * (v / (power[None, :] + v))
            weights.append(posterior_variance.sum(dim=1) / total_power)
        return torch.cat(weights).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        score_g = super().forward(x, sigma)
        weight = self.noise_weight(sigma).to(dtype=score_g.dtype)
        return weight[:, None, None, None] * score_g


def make_reference(stats: dict, mode: str, *, sigma_min: float,
                   sigma_max: float) -> GaussianReference | None:
    if mode not in REFERENCE_MODES:
        raise ValueError(f"Unknown reference mode: {mode}")
    if mode == "baseline":
        return None
    if mode in ("spectral", "isotropic"):
        return GaussianReference(stats, mode)
    if mode in ("spectral_mmse", "isotropic_mmse"):
        return MMSEGaussianReference(stats, mode.removesuffix("_mmse"))
    return LinearTimeGaussianReference(stats, mode.removesuffix("_linear"),
                                       sigma_min=sigma_min, sigma_max=sigma_max)
