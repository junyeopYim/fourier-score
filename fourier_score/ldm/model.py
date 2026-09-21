"""One epsilon interface for native LDM training, DDPM and DDIM sampling."""

import copy
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from fourier_score.method import GAUSSIAN_OBJECTIVES, FourierGaussian

from .first_stage import read_state
from .upstream.openaimodel import UNetModel
from .upstream.util import (
    make_beta_schedule,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
)


class Schedule(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.num_timesteps = spec["timesteps"]
        b = make_beta_schedule(
            "linear", self.num_timesteps, spec["linear_start"], spec["linear_end"]
        )
        a = 1.0 - b
        ab = np.cumprod(a)
        prev = np.append(1.0, ab[:-1])
        v = spec["v_posterior"]
        pv = (1 - v) * b * (1 - prev) / (1 - ab) + v * b
        arrays = {
            "betas": b,
            "alphas_cumprod": ab,
            "alphas_cumprod_prev": prev,
            "sqrt_alphas_cumprod": np.sqrt(ab),
            "sqrt_one_minus_alphas_cumprod": np.sqrt(1 - ab),
            "sqrt_recip_alphas_cumprod": np.sqrt(1 / ab),
            "sqrt_recipm1_alphas_cumprod": np.sqrt(1 / ab - 1),
            "posterior_variance": pv,
            "posterior_log_variance_clipped": np.log(np.maximum(pv, 1e-20)),
            "posterior_mean_coef1": b * np.sqrt(prev) / (1 - ab),
            "posterior_mean_coef2": (1 - prev) * np.sqrt(a) / (1 - ab),
        }
        for name, value in arrays.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        weights = self.betas.square() / (
            2
            * self.posterior_variance
            * torch.tensor(a, dtype=torch.float32)
            * (1 - self.alphas_cumprod)
        )
        weights[0] = weights[1]
        self.register_buffer("lvlb_weights", weights)

    def coefficients(self, t):
        return self.sqrt_alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t]

    def q_sample(self, z, t, noise):
        a, s = self.coefficients(t)
        return a[:, None, None, None] * z + s[:, None, None, None] * noise

    def predict_start(self, z, t, eps):
        return (
            self.sqrt_recip_alphas_cumprod[t, None, None, None] * z
            - self.sqrt_recipm1_alphas_cumprod[t, None, None, None] * eps
        )


class Denoiser(nn.Module):
    def __init__(
        self,
        spec,
        parameterization="epsilon",
        stats=None,
        spectral_backend="auto",
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.parameterization = parameterization
        args = copy.deepcopy(spec["unet"])
        if gradient_checkpointing:
            args["use_checkpoint"] = True
        self.diffusion_model = UNetModel(**args)
        self.schedule = Schedule(spec)
        self.reference = None
        if parameterization in GAUSSIAN_OBJECTIVES:
            if stats is None:
                raise ValueError(
                    "Gaussian parameterizations require training-latent statistics"
                )
            self.reference = FourierGaussian(
                stats, spectral_backend, **GAUSSIAN_OBJECTIVES[parameterization]
            )
        elif parameterization != "epsilon":
            raise ValueError("Unknown LDM parameterization")

    def forward(self, z_t, t):
        raw = self.diffusion_model(z_t, t)
        if self.reference is None:
            return raw
        alpha, sigma = self.schedule.coefficients(t)
        return -self.reference.scaled_score(raw, z_t, alpha, sigma)

    def loss(self, clean, t, noise):
        eps = self(self.schedule.q_sample(clean, t, noise), t)
        residual = eps - noise
        simple = (
            (residual.abs() if self.spec["loss_type"] == "l1" else residual.square())
            .flatten(1)
            .mean(1)
        )
        logvar = self.spec["logvar_init"]
        loss = self.spec["l_simple_weight"] * (simple / np.exp(logvar) + logvar).mean()
        loss = (
            loss
            + self.spec["original_elbo_weight"]
            * (self.schedule.lvlb_weights[t] * simple).mean()
        )
        return loss


@contextmanager
def ema_scope(model, ema):
    ema.store(model.parameters())
    ema.copy_to(model)
    try:
        yield
    finally:
        ema.restore(model.parameters())
        ema.collected_params = []


def load_public_denoiser(path, spec, weights="ema", device="cpu"):
    if weights not in ("ema", "raw"):
        raise ValueError("Public weights must be ema or raw")
    state = read_state(path)
    model = Denoiser(spec)
    tensors = {}
    for key in model.diffusion_model.state_dict():
        source = (
            "model.diffusion_model." + key
            if weights == "raw"
            else "model_ema." + ("diffusion_model." + key).replace(".", "")
        )
        if source not in state:
            raise ValueError(f"Missing public {weights} tensor: {source}")
        tensors[key] = state[source]
    model.diffusion_model.load_state_dict(tensors, strict=True)
    for name, value in model.schedule.named_buffers():
        if name in state and not torch.allclose(
            value, state[name], rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"Checkpoint/config diffusion schedule mismatch: {name}")
    return model.to(device).eval()


@torch.no_grad()
def sample_latents(model, batch_size, options, device, generator, initial_noise=None):
    shape = (batch_size, *model.spec["latent_shape"])

    def noise():
        return torch.randn(shape, generator=generator).to(device)

    z = noise() if initial_noise is None else initial_noise.to(device).clone()
    if tuple(z.shape) != shape:
        raise ValueError("Initial latent shape mismatch")
    schedule = model.schedule
    n = schedule.num_timesteps
    calls = 0
    if options["method"] == "ddim":
        steps = options["steps"]
        if not 0 < steps < n or n % steps:
            raise ValueError(
                "Native DDIM requires steps dividing the trained timestep count, with steps < timesteps"
            )
        times = make_ddim_timesteps("uniform", steps, n, verbose=False)
        sigma, alpha, previous = make_ddim_sampling_parameters(
            schedule.alphas_cumprod.cpu().numpy(), times, options["eta"], verbose=False
        )
        for j in reversed(range(len(times))):
            t = torch.full(
                (batch_size,), int(times[j]), device=device, dtype=torch.long
            )
            eps = model(z, t)
            a, prev, s = [
                torch.as_tensor(v, device=device, dtype=z.dtype)
                for v in (alpha[j], previous[j], sigma[j])
            ]
            # Native DDIM recomputes this from the float32 cumulative alpha.
            predicted = (z - (1 - a).sqrt() * eps) / a.sqrt()
            direction = (1 - prev - s.square()).clamp_min(0).sqrt() * eps
            z = prev.sqrt() * predicted + direction + s * noise()
            calls += 1
    elif options["method"] == "ddpm":
        if options["steps"] != n:
            raise ValueError("DDPM sampling uses the full trained grid")
        for i in reversed(range(n)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            predicted = schedule.predict_start(z, t, model(z, t))
            mean = (
                schedule.posterior_mean_coef1[i] * predicted
                + schedule.posterior_mean_coef2[i] * z
            )
            # Latent predictions are unbounded: never clamp to RGB [-1, 1].
            z = (
                mean
                + (0 if i == 0 else 1)
                * torch.exp(0.5 * schedule.posterior_log_variance_clipped[i])
                * noise()
            )
            calls += 1
    else:
        raise ValueError("Unknown sampler")
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite sampled latents")
    return z, calls
