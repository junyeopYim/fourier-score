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


@torch.no_grad()
def sample_batch(model, cfg, batch_size, device, generator):
    p = model.process
    d = cfg["data_loader"]["args"]
    opt = cfg["sampling"]
    shape = (batch_size, d["channels"], d["image_size"], d["image_size"])

    def noise():
        return torch.randn(shape, generator=generator, device="cpu").to(device)

    x = noise() * (p.sigma_max if p.kind == "ve" else 1.0)
    nfe = 0

    def score(x, level):
        nonlocal nfe
        nfe += 1
        return model(x, level)

    def ve_level(sigma):
        coord = torch.full(
            (batch_size,),
            math.log(float(sigma) / p.sigma_min) / p.log_ratio,
            device=device,
        )
        return p.level(coord)

    if p.kind == "ddpm":
        if opt["method"] != "ddpm" or opt["steps"] != p.N:
            raise ValueError("DDPM requires its full trained grid")
        for i in range(p.N - 1, -1, -1):
            lev = p.level(torch.full((batch_size,), i, device=device, dtype=torch.long))
            sc = score(x, lev)
            alpha = lev.alpha[:, None, None, None]
            sigma = lev.sigma[:, None, None, None]
            x0 = (x + sigma.square() * sc) / alpha
            if opt["clip_denoised"]:
                x0 = x0.clamp(-1 if d["centered"] else 0, 1)
            beta = float(p.betas[i])
            ab = float(p.alpha_bars[i])
            prev = float(p.alpha_bars[i - 1]) if i else 1.0
            mean = (beta * math.sqrt(prev) / (1 - ab)) * x0 + (
                (1 - prev) * math.sqrt(1 - beta) / (1 - ab)
            ) * x
            x = (
                mean + math.sqrt(max(0.0, beta * (1 - prev) / (1 - ab))) * noise()
                if i
                else mean
            )
    else:
        lo = p.sigma_min * math.exp(p.t_min * p.log_ratio)
        sigmas = torch.exp(
            torch.linspace(
                math.log(p.sigma_max), math.log(lo), opt["steps"], dtype=torch.float64
            )
        ).tolist()
        if opt["method"] == "heun":
            for hi, low in zip(sigmas[:-1], sigmas[1:]):
                deriv = -hi * score(x, ve_level(hi))
                proposal = x + (low - hi) * deriv
                next_deriv = -low * score(proposal, ve_level(low))
                x = x + (low - hi) * 0.5 * (deriv + next_deriv)
            if opt["denoise"]:
                x = x + lo * lo * score(x, ve_level(lo))
        elif opt["method"] == "pc":
            mean = x
            for i, hi in enumerate(sigmas):
                lev = ve_level(hi)
                for _ in range(opt["corrector_steps"]):
                    grad = score(x, lev)
                    z = noise()
                    gn = grad.flatten(1).norm(dim=1).mean()
                    zn = z.flatten(1).norm(dim=1).mean()
                    if not torch.isfinite(gn) or gn <= 0:
                        raise FloatingPointError("Invalid Langevin gradient norm")
                    step = 2 * (opt["snr"] * zn / gn).square()
                    x = x + step * grad + (2 * step).sqrt() * z
                low = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0
                dv = hi * hi - low * low
                mean = x + dv * score(x, lev)
                x = mean + math.sqrt(dv) * noise()
            if opt["denoise"]:
                x = mean
        else:
            raise ValueError("Unknown sampler")
    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "Nonfinite samples. Check model and sampler settings."
        )
    return x, nfe
