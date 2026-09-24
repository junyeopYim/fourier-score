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
@pytest.mark.parametrize("mode", ["log_sigma", "log_sigma_plateau", "spectral_cap", "linear_sigma", "tanh_sigma"])
def test_gated_loss_matches_weighted_dsm_and_gradient(cfg, covariance, reduction, backend, mode):
    stats, x, noise, raw, level = inputs()
    ref = FourierGaussian(stats, backend, covariance=covariance,
                          gate=dict(mode=mode, sigma_switch=.7, sigma_lo=.25, sigma_hi=.75)).double()
    model = SimpleNamespace(process=NoiseProcess(cfg["process"]), reference=ref,
                            embedding="fourier", backbone=lambda y, condition: raw)
    loss = training_loss(model, x, level, noise, reduction, objective="normalized_residual")
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
@pytest.mark.parametrize("mode", ["log_sigma", "log_sigma_plateau", "spectral_cap", "linear_sigma", "tanh_sigma"])
def test_nongaussian_target_second_moment(covariance, mode):
    # Finite population with exact covariance: signed scaled basis vectors.
    d = 16
    power = conjugate_symmetrize(torch.arange(1, d+1).reshape(1, 4, 4).float()) / d
    ref = FourierGaussian(dict(mean=torch.zeros_like(power), power=power), covariance=covariance,
                          gate=dict(mode=mode, sigma_switch=.5, sigma_lo=.6, sigma_hi=.8)).double()
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


@pytest.mark.parametrize("mode", ["log_sigma", "log_sigma_plateau", "spectral_cap", "linear_sigma", "tanh_sigma"])
def test_flat_spectrum_gated_scalar_matches_fourier_without_fft(monkeypatch, mode):
    stats, x, noise, raw, level = inputs()
    stats["power"] = stats["power"].mean((-2, -1), keepdim=True).expand_as(stats["power"])
    gate = dict(mode=mode, sigma_switch=1.5, sigma_lo=.25, sigma_hi=.75)
    scalar = FourierGaussian(stats, covariance="scalar", gate=gate).double()
    fourier = FourierGaussian(stats, gate=gate).double()
    monkeypatch.setattr(scalar.filter, "forward", lambda *args: pytest.fail("Scalar used FFT"))
    y = x + level.sigma[:, None, None, None] * noise
    torch.testing.assert_close(scalar.normalized_target(x, noise, level.sigma),
                               fourier.normalized_target(x, noise, level.sigma), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(scalar.scaled_score(raw, y, level.alpha, level.sigma),
                               fourier.scaled_score(raw, y, level.alpha, level.sigma), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
@pytest.mark.parametrize("mode", ["log_sigma", "log_sigma_plateau", "spectral_cap", "linear_sigma", "tanh_sigma"])
def test_gated_target_extreme_power_noise(covariance, mode):
    ref = FourierGaussian(dict(mean=torch.zeros(1, 4, 4), power=torch.full((1, 4, 4), 1e-12)),
                          covariance=covariance, gate=dict(mode=mode)).float()
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


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_plateau_preserves_low_adapter_and_recovers_high_adapter(covariance, dtype):
    stats, x, noise, raw, _ = inputs()
    x, noise, raw = [t.to(dtype) for t in (x, noise, raw)]
    sigma = torch.tensor([.1, .8, 1., 50.], dtype=dtype)
    old = FourierGaussian(stats, covariance=covariance,
                          gate=dict(mode="log_sigma", sigma_switch=1.5)).to(dtype)
    plateau = FourierGaussian(stats, covariance=covariance,
                              gate=dict(mode="log_sigma_plateau", sigma_switch=1.5)).to(dtype)
    original = FourierGaussian(stats, covariance=covariance).to(dtype)
    g, remaining = plateau._gate_weights(sigma)
    torch.testing.assert_close(g[:2], old.gate_value(sigma[:2]), atol=0, rtol=0)
    assert torch.equal(g[2:], torch.ones_like(g[2:]))
    assert torch.equal(remaining[2:], torch.zeros_like(remaining[2:]))
    y = x + sigma[:, None, None, None] * noise
    for a, b in [(plateau.normalized_target(x, noise, sigma),
                  torch.cat([old.normalized_target(x[:2], noise[:2], sigma[:2]),
                             original.normalized_target(x[2:], noise[2:], sigma[2:])])),
                 (plateau.scaled_score(raw, y, torch.ones_like(sigma), sigma),
                  torch.cat([old.scaled_score(raw[:2], y[:2], torch.ones(2, dtype=dtype), sigma[:2]),
                             original.scaled_score(raw[2:], y[2:], torch.ones(2, dtype=dtype), sigma[2:])]))]:
        torch.testing.assert_close(a[:2], b[:2], atol=0, rtol=0)
        tol = 3e-6 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(a[2:], b[2:], atol=tol, rtol=tol)


def test_plateau_is_monotone_and_has_continuous_endpoint_slopes(stats):
    ref = FourierGaussian(stats, gate=dict(mode="log_sigma_plateau", sigma_switch=1.5)).double()
    sigma = torch.cat([torch.linspace(.1, 2., 200, dtype=torch.float64),
                       torch.tensor([.8-1e-7, .8, .8+1e-7, 1.-1e-7, 1., 1.+1e-7], dtype=torch.float64)])
    sigma.requires_grad_()
    g, complement = ref._gate_weights(sigma)
    assert (g[1:200] >= g[:199]).all()
    torch.testing.assert_close(g + complement, torch.ones_like(g), atol=3e-16, rtol=3e-16)
    derivative = torch.autograd.grad(g.sum(), sigma)[0]
    torch.testing.assert_close(derivative[-6:-3], derivative[-5].expand(3), atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(derivative[-3:], torch.zeros(3, dtype=sigma.dtype), atol=2e-5, rtol=0)


def test_plateau_config_and_pre_plateau_active_gate_signature(cfg):
    cfg["fourier"]["gate"]["mode"] = "log_sigma"
    old = copy.deepcopy(cfg)
    del old["fourier"]["gate"]["sigma_lo"], old["fourier"]["gate"]["sigma_hi"]
    assert validate(old) == cfg
    assert resume_signature(old) == resume_signature(cfg)
    cfg["fourier"]["gate"].update(mode="log_sigma_plateau", sigma_switch=1.5)
    validate(cfg)
    assert experiment_name(cfg).endswith("_gate_s1p5_p4_plateau_lo0p8_hi1")
    changed = copy.deepcopy(cfg)
    changed["fourier"]["gate"]["sigma_hi"] = 1.1
    assert experiment_name(changed) != experiment_name(cfg)
    assert resume_signature(changed) != resume_signature(cfg)


@pytest.mark.parametrize("covariance", ["scalar", "fourier"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_cap_preserves_low_noise_and_bounds_high_scale(covariance, dtype):
    stats, x, noise, raw, _ = inputs()
    x, noise, raw = [t.to(dtype) for t in (x, noise, raw)]
    sigma = torch.tensor([.1, 1., 2., 50.], dtype=dtype)
    gate = dict(mode="spectral_cap", sigma_switch=1.5, sigma_lo=1., sigma_hi=2., delta=.5)
    ref = FourierGaussian(stats, covariance=covariance, gate=gate).to(dtype)
    old = FourierGaussian(stats, covariance=covariance,
                          gate=dict(mode="log_sigma", sigma_switch=1.5)).to(dtype)
    g, q, c, total = ref._coefficients(sigma, ref.power[None])
    old_g, old_q = old._gate_weights(sigma)
    torch.testing.assert_close(g[:2], old_g[:2].expand_as(g[:2]), atol=0, rtol=0)
    torch.testing.assert_close(q[:2], old_q[:2].expand_as(q[:2]), atol=0, rtol=0)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    b = (ref.power[None] / total).sqrt()
    assert ((c/b)[2:] <= (1+.5**2)**.5 + tolerance).all()
    torch.testing.assert_close(g+q, torch.ones_like(g), atol=tolerance, rtol=tolerance)
    assert not list(ref.parameters())
    y = x + sigma[:, None, None, None] * noise
    target = ref.normalized_target(x, noise, sigma)
    torch.testing.assert_close(target[:2], old.normalized_target(x[:2], noise[:2], sigma[:2]),
                               atol=0, rtol=0)
    scaled = ref.scaled_score(raw, y, torch.ones_like(sigma), sigma)
    torch.testing.assert_close(scaled[:2], old.scaled_score(raw[:2], y[:2], torch.ones(2, dtype=dtype), sigma[:2]),
                               atol=tolerance, rtol=tolerance)
    if covariance == "fourier":
        # A varying Fourier multiplier must never be applied as a pixel mask.
        expected = torch.fft.ifft2(-sigma[:,None,None,None] * g / total
                                  * torch.fft.fft2(y-ref.mean[None], norm="ortho")
                                  + c * torch.fft.fft2(raw, norm="ortho"), norm="ortho").real
        torch.testing.assert_close(scaled, expected, atol=tolerance, rtol=tolerance)
        assert g[2].max() > g[2].min()


@pytest.mark.parametrize("mode", ["constant", "log_sigma", "log_sigma_plateau"])
def test_spectral_delta_preserves_legacy_signatures(cfg, mode):
    cfg["fourier"]["gate"]["mode"] = mode
    legacy = copy.deepcopy(cfg)
    del legacy["fourier"]["gate"]["delta"]
    assert validate(legacy) == cfg
    assert resume_signature(legacy) == resume_signature(cfg)
    cfg["fourier"]["gate"].update(mode="spectral_cap", sigma_switch=1.5, sigma_lo=1., sigma_hi=2.)
    assert experiment_name(cfg).endswith("_gate_s1p5_p4_cap_lo1_hi2_d0p5")
    for key, value in (("delta", .25), ("sigma_hi", 2.5)):
        changed = copy.deepcopy(cfg)
        changed["fourier"]["gate"][key] = value
        assert resume_signature(changed) != resume_signature(cfg)
        assert experiment_name(changed) != experiment_name(cfg)


@pytest.mark.parametrize("sharpness", [2., 4., 8.])
@pytest.mark.parametrize("mode", ["linear_sigma", "tanh_sigma"])
def test_sigma_gate_center_slope_and_shape(stats, mode, sharpness):
    ref = FourierGaussian(stats, gate=dict(mode=mode, sigma_switch=1.5, sharpness=sharpness)).double()
    sigma = torch.linspace(.1, 3., 200, dtype=torch.float64)
    g, q = ref._gate_weights(sigma)
    assert (g[1:] >= g[:-1]).all() and ((g >= 0) & (g <= 1)).all()
    torch.testing.assert_close(g+q, torch.ones_like(g), rtol=3e-16, atol=3e-16)
    center = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    value = ref.gate_value(center)
    torch.testing.assert_close(value, torch.full_like(value, .5), atol=0, rtol=0)
    slope = torch.autograd.grad(value.sum(), center)[0]
    torch.testing.assert_close(slope, torch.full_like(center, sharpness/(4*1.5)), atol=1e-15, rtol=1e-15)
    if mode == "tanh_sigma":
        expected = .5 * (1+torch.tanh((sharpness/2)*(sigma/1.5-1)))
        torch.testing.assert_close(g.flatten(), expected, atol=3e-16, rtol=3e-16)
        # log-sigma tanh is an algebraic identity, not a distinct comparator.
        old = FourierGaussian(stats, gate=dict(mode="log_sigma", sigma_switch=1.5, sharpness=sharpness)).double()
        log_tanh = .5 * (1+torch.tanh((sharpness/2)*torch.log(sigma/1.5)))
        torch.testing.assert_close(old.gate_value(sigma).flatten(), log_tanh, atol=3e-16, rtol=3e-16)
        assert not torch.allclose(g, old.gate_value(sigma))
        _, high_q = ref._gate_weights(torch.tensor([20.], dtype=torch.float64))
        assert high_q.item() > 0  # Keep the tail after the gate rounds to one.
    else:
        assert (g[sigma <= 1.5*(1-2/sharpness)] == 0).all()
        assert (g[sigma >= 1.5*(1+2/sharpness)] == 1).all()


@pytest.mark.parametrize("mode", ["linear_sigma", "tanh_sigma"])
def test_sigma_gate_names_and_resume_contract(cfg, mode):
    old = copy.deepcopy(cfg)
    old["fourier"]["gate"].update(mode="log_sigma", sigma_switch=1.5)
    cfg["fourier"]["gate"].update(mode=mode, sigma_switch=1.5)
    validate(cfg)
    assert experiment_name(cfg).endswith("_gate_s1p5_p4_"+mode)
    assert resume_signature(cfg) != resume_signature(old)
    for key in ("sigma_switch", "sharpness"):
        changed = copy.deepcopy(cfg)
        changed["fourier"]["gate"][key] *= 2
        assert experiment_name(changed) != experiment_name(cfg)
        assert resume_signature(changed) != resume_signature(cfg)
    ignored = copy.deepcopy(cfg)
    ignored["fourier"]["gate"].update(sigma_lo=.1, sigma_hi=3., delta=.25)
    assert resume_signature(ignored) == resume_signature(cfg)


@pytest.mark.parametrize("parameterization,process", [("score", "ve"), ("fourier_gaussian", "ddpm")])
def test_gate_rejects_unsupported_models(cfg, parameterization, process):
    cfg["loss"]["type"] = parameterization
    cfg["process"]["type"] = process
    cfg["fourier"]["gate"]["mode"] = "log_sigma"
    with pytest.raises(ValueError, match="Gaussian gate requires VE"):
        validate(cfg)


@pytest.mark.parametrize("gate", [dict(mode="learned"), dict(sigma_switch=0), dict(sharpness=-1),
                                dict(value=1.1), dict(value=float("nan")), dict(sharpness=True),
                                dict(sigma_lo=0), dict(sigma_lo=1), dict(sigma_hi=.5),
                                dict(sigma_hi=float("inf")), dict(delta=0), dict(delta=-.5),
                                dict(delta=float("nan"))])
def test_invalid_gate(gate, stats):
    with pytest.raises(ValueError, match="gate"):
        FourierGaussian(stats, gate=gate)
