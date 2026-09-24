"""Endpoint, weighted-DSM, population-moment and checkpoint contracts for gates."""

import copy
from types import SimpleNamespace

import pytest
import torch

from fourier_score.checkpoints import resume_signature
from fourier_score.config import experiment_name, validate
from fourier_score.diffusion import NoiseLevel, NoiseProcess
from fourier_score.loss import training_loss
from fourier_score.method import FourierGaussian
from fourier_score.spectral import conjugate_symmetrize


def inputs():
    rng = torch.Generator().manual_seed(91)
    stats = dict(mean=torch.randn(2, 5, 7, generator=rng),
                 power=conjugate_symmetrize(torch.rand(2, 5, 7, generator=rng) + .03))
    x, noise, raw = [torch.randn(4, 2, 5, 7, generator=rng, dtype=torch.float64) for _ in range(3)]
    sigma = torch.tensor([.01, .5, 1., 50.], dtype=torch.float64)
    level = NoiseLevel(torch.ones_like(sigma), sigma, torch.zeros_like(sigma))
    return stats, x, noise, raw.requires_grad_(), level


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
@pytest.mark.parametrize("value", [0., 1.])
def test_gate_endpoints(covariance, value):
    stats, x, noise, raw, level = inputs()
    ref = FourierGaussian(stats, covariance=covariance, gate=dict(mode="constant", value=value)).double()
    original = FourierGaussian(stats, covariance=covariance).double()
    y = x + level.sigma[:, None, None, None] * noise
    target = ref.normalized_target(x, noise, level.sigma)
    scaled = ref.scaled_score(raw, y, level.alpha, level.sigma)
    expected_target = -noise if value == 0 else original.normalized_target(x, noise, level.sigma)
    expected_scaled = raw if value == 0 else original.scaled_score(raw, y, level.alpha, level.sigma)
    torch.testing.assert_close(target, expected_target, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(scaled, expected_scaled, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
@pytest.mark.parametrize("reduction", ["mean", "half_sum"])
@pytest.mark.parametrize("backend", ["fft", "matmul"])
def test_gated_loss_matches_weighted_dsm_and_gradient(cfg, covariance, reduction, backend):
    stats, x, noise, raw, level = inputs()
    ref = FourierGaussian(stats, backend, covariance=covariance,
                          gate=dict(mode="log_sigma", sigma_switch=.7)).double()
    model = SimpleNamespace(process=NoiseProcess(cfg["process"]), reference=ref,
                            embedding="fourier", backbone=lambda y, condition: raw)
    loss = training_loss(model, x, level, noise, reduction, objective="normalized_residual")
    y = model.process.perturb(x, level, noise)
    error = ref.scaled_score(raw, y, level.alpha, level.sigma) + noise
    s = level.sigma[:, None, None, None]
    # Independent rational gate and variance derivation, without adapter helpers.
    g = s.pow(4) / (s.pow(4) + .7 ** 4)
    total = ref.power[None] + s.square()
    variance = (g.square() * s.square() * ref.power[None]
                + (ref.power[None] + (1-g)*s.square()).square()) / total.square()
    errors = (torch.fft.fft2(error, norm="ortho").abs().square() / variance).flatten(1)
    expected = (errors.mean(1) if reduction == "mean" else .5 * errors.sum(1)).mean()
    # The portable DFT stores its basis in FP32; inverse-scale weighting at
    # sigma=50 amplifies that fixed basis error even after .double().
    tol = 5e-6 if backend == "matmul" else 2e-10
    torch.testing.assert_close(loss, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(torch.autograd.grad(loss, raw)[0], torch.autograd.grad(expected, raw)[0],
                               atol=tol, rtol=tol)


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
def test_nongaussian_target_second_moment(covariance):
    # Finite population with exact covariance: signed scaled basis vectors.
    d = 16
    power = conjugate_symmetrize(torch.arange(1, d+1).reshape(1, 4, 4).float()) / d
    ref = FourierGaussian(dict(mean=torch.zeros_like(power), power=power), covariance=covariance,
                          gate=dict(mode="log_sigma", sigma_switch=.5)).double()
    basis = torch.eye(d, dtype=torch.float64).reshape(d, 1, 4, 4) * d ** .5
    population = torch.cat([basis, -basis])
    clean = torch.fft.ifft2(torch.fft.fft2(population, norm="ortho") * power.double().sqrt(), norm="ortho").real
    # Independent zero-mean unit-covariance noise has the same second-moment identity.
    x = clean.repeat_interleave(2*d, 0)
    noise = population.repeat(2*d, 1, 1, 1)
    sigma = torch.full((len(x),), .7, dtype=torch.float64)
    target = ref.normalized_target(x, noise, sigma)
    moment = torch.fft.fft2(target, norm="ortho").abs().square().mean(0)
    if covariance == "scalar":
        moment = moment.mean((-2, -1))  # Scalar matches channel-average power only.
    torch.testing.assert_close(moment, torch.ones_like(moment), atol=2e-12, rtol=2e-12)


def test_flat_spectrum_gated_scalar_matches_fourier_without_fft(monkeypatch):
    stats, x, noise, raw, level = inputs()
    stats["power"] = stats["power"].mean((-2, -1), keepdim=True).expand_as(stats["power"])
    gate = dict(mode="log_sigma", sigma_switch=1.5)
    scalar = FourierGaussian(stats, covariance="scalar", gate=gate).double()
    fourier = FourierGaussian(stats, gate=gate).double()
    monkeypatch.setattr(scalar.filter, "forward", lambda *args: pytest.fail("Scalar used FFT"))
    y = x + level.sigma[:, None, None, None] * noise
    torch.testing.assert_close(scalar.normalized_target(x, noise, level.sigma),
                               fourier.normalized_target(x, noise, level.sigma), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(scalar.scaled_score(raw, y, level.alpha, level.sigma),
                               fourier.scaled_score(raw, y, level.alpha, level.sigma), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
def test_gated_target_extreme_power_noise(covariance):
    ref = FourierGaussian(dict(mean=torch.zeros(1, 4, 4), power=torch.full((1, 4, 4), 1e-12)),
                          covariance=covariance, gate=dict(mode="log_sigma")).float()
    rng = torch.Generator().manual_seed(99)
    x = 1e-6 * torch.randn(3, 1, 4, 4, generator=rng)
    noise = torch.randn(x.shape, generator=rng)
    sigma = torch.tensor([.01, 50., 1e6])
    actual = ref.normalized_target(x, noise, sigma)
    expected = copy.deepcopy(ref).double().normalized_target(x.double(), noise.double(), sigma.double())
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-5)


def test_gate_config_names_and_legacy_signatures(cfg):
    legacy = copy.deepcopy(cfg)
    del legacy["fourier"]["gate"]
    assert validate(legacy) == cfg
    assert resume_signature(legacy) == resume_signature(cfg)
    original_name = experiment_name(cfg)
    cfg["fourier"]["gate"]["mode"] = "log_sigma"
    validate(cfg)
    assert experiment_name(cfg) == original_name + "_gate_s1_p4"
    assert resume_signature(cfg) != resume_signature(legacy)
    changed = copy.deepcopy(cfg)
    changed["fourier"]["gate"]["sigma_switch"] = .5
    assert experiment_name(changed) != experiment_name(cfg)
    assert resume_signature(changed) != resume_signature(cfg)


@pytest.mark.parametrize("parameterization,process", [("score", "ve"), ("fourier_gaussian", "ddpm")])
def test_gate_rejects_unsupported_models(cfg, parameterization, process):
    cfg["loss"]["type"] = parameterization
    cfg["process"]["type"] = process
    cfg["fourier"]["gate"]["mode"] = "log_sigma"
    with pytest.raises(ValueError, match="Gaussian gate requires VE"):
        validate(cfg)


@pytest.mark.parametrize("gate", [dict(mode="learned"), dict(sigma_switch=0), dict(sharpness=-1),
                                dict(value=1.1), dict(value=float("nan")), dict(sharpness=True)])
def test_invalid_gate(gate, stats):
    with pytest.raises(ValueError, match="gate"):
        FourierGaussian(stats, gate=gate)
