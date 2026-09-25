"""The normalized objective, with or without a noise gate, must match inverse-variance-weighted DSM."""

from types import SimpleNamespace

import pytest
import torch

from fourier_score.model.loss import training_loss
from fourier_score.model.process import NoiseLevel, NoiseProcess
from fourier_score.model.reference import FourierGaussian
from fourier_score.model.spectral import conjugate_symmetrize

GATE_MODES = ["log_sigma", "log_sigma_plateau", "spectral_cap", "linear_sigma", "tanh_sigma",
              "linear_log_sigma", "bounded_log_sigmoid"]


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
@pytest.mark.parametrize("reduction", ["mean", "half_sum"])
def test_normalized_loss_matches_weighted_dsm_value_and_gradient(cfg, covariance, reduction):
    gen = torch.Generator().manual_seed(11)
    mean = torch.randn(3, 7, 9, generator=gen, dtype=torch.float64)
    power = conjugate_symmetrize(torch.rand(mean.shape, generator=gen) + 0.01)
    power *= torch.tensor([0.1, 1.0, 10.0])[:, None, None]
    ref = FourierGaussian({"mean": mean, "power": power}, covariance=covariance).double()
    clean = torch.randn(3, 3, 7, 9, generator=gen, dtype=torch.float64)
    noise = torch.randn(clean.shape, generator=gen, dtype=torch.float64)
    raw = torch.randn(clean.shape, generator=gen, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([0.01, 0.5, 50.0], dtype=torch.float64)
    level = NoiseLevel(torch.ones_like(sigma), sigma, torch.zeros_like(sigma))
    process = NoiseProcess(cfg["process"])
    model = SimpleNamespace(
        process=process, reference=ref, embedding="fourier",
        backbone=lambda y, condition: raw,
    )
    actual = training_loss(
        model, clean, level, noise, reduction, objective="normalized_residual"
    )

    # Independent route through the existing score adapter and spectral DSM.
    y = process.perturb(clean, level, noise)
    residual = ref.scaled_score(raw, y, level.alpha, sigma) + noise
    spectrum = torch.fft.fft2(residual, norm="ortho")
    inverse_b2 = 1 + sigma[:, None, None, None].square() / ref.power[None]
    errors = (spectrum.abs().square() * inverse_b2).flatten(1)
    expected = (errors.mean(1) if reduction == "mean" else 0.5 * errors.sum(1)).mean()
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=2e-9)
    actual_grad = torch.autograd.grad(actual, raw)[0]
    expected_grad = torch.autograd.grad(expected, raw)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=2e-9)


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
@pytest.mark.parametrize("mode", GATE_MODES)
def test_gated_loss_matches_weighted_dsm_and_gradient(cfg, covariance, mode):
    rng = torch.Generator().manual_seed(91)
    stats = dict(mean=torch.randn(2, 5, 7, generator=rng),
                 power=conjugate_symmetrize(torch.rand(2, 5, 7, generator=rng) + .03))
    x, noise, raw = [torch.randn(4, 2, 5, 7, generator=rng, dtype=torch.float64) for _ in range(3)]
    raw.requires_grad_()
    sigma = torch.tensor([.01, .5, 1., 50.], dtype=torch.float64)
    level = NoiseLevel(torch.ones_like(sigma), sigma, torch.zeros_like(sigma))
    ref = FourierGaussian(stats, covariance=covariance,
                          gate=dict(mode=mode, sigma_switch=.7, sigma_lo=.25, sigma_hi=.75)).double()
    model = SimpleNamespace(process=NoiseProcess(cfg["process"]), reference=ref,
                            embedding="fourier", backbone=lambda y, condition: raw)
    loss = training_loss(model, x, level, noise, "mean", objective="normalized_residual")
    y = model.process.perturb(x, level, noise)
    error = ref.scaled_score(raw, y, level.alpha, level.sigma) + noise
    s = level.sigma[:, None, None, None]
    # Independent rational gate and variance derivation, without adapter helpers.
    g = s.pow(4) / (s.pow(4) + .7 ** 4)
    if mode == "log_sigma_plateau":
        z = (torch.log(s / .25) / torch.log(torch.tensor(3., dtype=s.dtype))).clamp(0, 1)
        g = g + z.square() * (3 - 2*z) * (1 - g)
    elif mode == "spectral_cap":
        z = (torch.log(s / .25) / torch.log(torch.tensor(3., dtype=s.dtype))).clamp(0, 1)
        ramp = z.square() * (3 - 2*z)
        g = 1 - (1-g) / (1 + ramp * (1-g).square() * s.square() / (.5**2 * ref.power[None])).sqrt()
    elif mode == "linear_sigma":
        g = (s / .7 - .5).clamp(0, 1)
    elif mode == "tanh_sigma":
        g = .5 * (1 + torch.tanh(2 * (s / .7 - 1)))
    elif mode in ("linear_log_sigma", "bounded_log_sigmoid"):
        z = (torch.log(s/.25) / torch.log(torch.tensor(3., dtype=s.dtype))).clamp(0, 1)
        g = z
        if mode == "bounded_log_sigmoid":
            center = torch.log(torch.tensor(.7/.25, dtype=s.dtype)) / torch.log(torch.tensor(3., dtype=s.dtype))
            a, b = (z*(1-center)).pow(4), ((1-z)*center).pow(4)
            g = a/(a+b)
    total = ref.power[None] + s.square()
    variance = (g.square() * s.square() * ref.power[None]
                + (ref.power[None] + (1-g)*s.square()).square()) / total.square()
    errors = (torch.fft.fft2(error, norm="ortho").abs().square() / variance).flatten(1)
    expected = errors.mean(1).mean()
    torch.testing.assert_close(loss, expected, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(torch.autograd.grad(loss, raw)[0], torch.autograd.grad(expected, raw)[0],
                               atol=2e-10, rtol=2e-10)
