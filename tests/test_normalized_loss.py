"""The normalized objective must match inverse-b^2-weighted DSM."""

import copy
from types import SimpleNamespace

import pytest
import torch

from fourier_score.config import experiment_name, validate
from fourier_score.model.loss import training_loss
from fourier_score.model.process import NoiseLevel, NoiseProcess
from fourier_score.model.reference import FourierGaussian
from fourier_score.model.spectral import conjugate_symmetrize
from fourier_score.trainer.checkpoints import resume_signature


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


def test_scalar_target_matches_flat_spectrum_without_fft(monkeypatch):
    gen = torch.Generator().manual_seed(12)
    mean = torch.randn(3, 7, 9, generator=gen)
    power = conjugate_symmetrize(torch.rand(mean.shape, generator=gen) + 0.1)
    power *= torch.tensor([1.0, 2.0, 4.0])[:, None, None]
    scalar = FourierGaussian({"mean": mean, "power": power}, covariance="scalar")
    flat = power.mean((-2, -1), keepdim=True).expand_as(power)
    fourier = FourierGaussian({"mean": mean, "power": flat})
    clean = torch.randn(3, 3, 7, 9, generator=gen)
    noise = torch.randn(clean.shape, generator=gen)
    sigma = torch.tensor([0.01, 1.0, 50.0])

    def fail_fft(*args, **kwargs):
        raise AssertionError("The scalar target must use pointwise operations")

    monkeypatch.setattr(scalar.filter, "forward", fail_fft)
    torch.testing.assert_close(
        scalar.normalized_target(clean, noise, sigma),
        fourier.normalized_target(clean, noise, sigma),
        atol=2e-6, rtol=2e-5,
    )


@pytest.mark.parametrize("backend", ["matmul", "cpu"])
def test_normalized_target_backends(backend):
    gen = torch.Generator().manual_seed(13)
    mean = torch.randn(3, 7, 9, generator=gen)
    power = conjugate_symmetrize(torch.rand(mean.shape, generator=gen) + 0.01)
    stats = {"mean": mean, "power": power}
    clean = torch.randn(3, 3, 7, 9, generator=gen)
    noise = torch.randn(clean.shape, generator=gen)
    sigma = torch.tensor([0.01, 1.0, 50.0])
    expected = FourierGaussian(stats, "fft").normalized_target(clean, noise, sigma)
    actual = FourierGaussian(stats, backend).normalized_target(clean, noise, sigma)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
def test_target_is_stable_at_small_power_and_large_sigma(covariance):
    gen = torch.Generator().manual_seed(14)
    ref = FourierGaussian(
        {"mean": torch.zeros(1, 7, 9), "power": torch.full((1, 7, 9), 1e-12)},
        covariance=covariance,
    )
    clean = 1e-6 * torch.randn(3, 1, 7, 9, generator=gen)
    noise = torch.randn(clean.shape, generator=gen)
    sigma = torch.tensor([0.01, 50.0, 1e6])
    actual = ref.normalized_target(clean, noise, sigma)
    s = sigma.double()[:, None, None, None]
    power = ref.power.double()[None]
    target_hat = (
        s * torch.fft.fft2(clean.double(), norm="ortho")
        - power * torch.fft.fft2(noise.double(), norm="ortho")
    ) / (power * (power + s.square())).sqrt()
    expected = torch.fft.ifft2(target_hat, norm="ortho").real
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("parameterization,process", [
    ("score", "ve"), ("diffusion", "ve"),
    ("scalar_gaussian", "ddpm"), ("fourier_gaussian", "ddpm"),
])
def test_normalized_objective_rejects_unsupported_combinations(cfg, parameterization, process):
    cfg["loss"].update(type=parameterization, objective="normalized_residual")
    cfg["process"]["type"] = process
    with pytest.raises(ValueError, match="normalized_residual requires VE"):
        validate(cfg)


@pytest.mark.parametrize("name", ["auto", "pilot", "pilot_{parameterization}_s{seed}"])
def test_objective_defaults_names_and_resume_signature(cfg, name):
    cfg["name"] = name
    legacy = copy.deepcopy(cfg)
    del legacy["loss"]["objective"]
    assert validate(legacy)["loss"]["objective"] == "dsm"
    assert resume_signature(legacy) == resume_signature(cfg)
    assert experiment_name(legacy) == experiment_name(cfg)
    normalized = copy.deepcopy(cfg)
    normalized["loss"]["objective"] = "normalized_residual"
    assert experiment_name(normalized) == experiment_name(cfg) + "_normalized_residual"
    assert resume_signature(normalized) != resume_signature(cfg)
