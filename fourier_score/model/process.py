"""Forward processes; all objectives share the exact same process object."""

from __future__ import annotations
import math
from dataclasses import dataclass
import torch


@dataclass
class NoiseLevel:
    alpha: torch.Tensor
    sigma: torch.Tensor
    coordinate: torch.Tensor  # continuous t in VE, integer index in DDPM

    def slice(self, start, end):
        return NoiseLevel(
            self.alpha[start:end], self.sigma[start:end], self.coordinate[start:end]
        )


class NoiseProcess:
    def __init__(self, config):
        self.config = dict(config)
        self.kind = config["type"]
        self.N = config["num_scales"]
        self.sigma_min = config["sigma_min"]
        self.sigma_max = config["sigma_max"]
        self.t_min = config["t_min"]
        self.log_ratio = math.log(self.sigma_max / self.sigma_min)
        self.betas = torch.linspace(
            config["beta_start"], config["beta_end"], self.N, dtype=torch.float64
        ).float()
        self.alpha_bars = (1 - self.betas.double()).cumprod(0).float()

    def level(self, coordinate, device=None):
        c = coordinate.to(device=device or coordinate.device)
        if self.kind == "ve":
            s = self.sigma_min * torch.exp(c.float() * self.log_ratio)
            return NoiseLevel(torch.ones_like(s), s, c.float())
        index = c.long()
        ab = self.alpha_bars.to(index.device)[index]
        return NoiseLevel(ab.sqrt(), (1 - ab).sqrt(), index)

    def sample(self, batch_size, device, generator):
        # Dedicated CPU generator makes t/noise independent of spectral kernels.
        if self.kind == "ve":
            c = (
                torch.rand(batch_size, generator=generator) * (1 - self.t_min)
                + self.t_min
            )
        else:
            c = torch.randint(self.N, (batch_size,), generator=generator)
        return self.level(c, device)

    def condition(self, level, embedding):
        if embedding == "fourier":
            return level.sigma
        if self.kind == "ddpm":
            return level.coordinate.float()
        # Original discrete VE label direction: high sigma -> label zero.
        return ((1 - level.coordinate) * (self.N - 1)).clamp(0, self.N - 1)

    def perturb(self, x, level, noise):
        return (
            level.alpha[:, None, None, None] * x
            + level.sigma[:, None, None, None] * noise
        )

    def bin_coordinate(self, level):
        return (
            level.coordinate.float() / (self.N - 1)
            if self.kind == "ddpm"
            else level.coordinate
        )
