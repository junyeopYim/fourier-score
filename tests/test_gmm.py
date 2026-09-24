"""Independent oracle, paired resume, and validation-only model selection."""

from dataclasses import asdict, replace
import math

import pytest
import torch

from fourier_score.gmm import (
    GMMConfig, MatchedMomentFamily, bank_seed, gated_arm, make_bank,
    make_model, select_gate, train_arm,
)


def small_config():
    return GMMConfig(preset="test", image_size=4, width=16, depth=2, batch_size=8,
                     steps=4, eval_every=2, n_noise_levels=3, val_per_noise=8,
                     test_per_noise=8, device="cpu", cpu_threads=1)


def test_gmm_oracle_matches_dense_log_density_gradient():
    cfg = small_config()
    family = MatchedMomentFamily(cfg, "gmm", 1.)
    rng = torch.Generator().manual_seed(45)
    y = torch.randn(3, 1, 4, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([.1, .7, 3.], dtype=torch.float64)
    covariance = family.within_fraction * family.covariance + sigma[:, None, None].square() * torch.eye(16)
    delta = y.flatten(1)[:, None] - family.means[None]
    solved = torch.linalg.solve(covariance[:, None], delta[..., None]).squeeze(-1)
    log_components = -.5 * ((delta * solved).sum(-1) + torch.linalg.slogdet(covariance)[1][:, None]
                            + 16*math.log(2*math.pi)) - math.log(len(family.means))
    log_density = torch.logsumexp(log_components, 1)
    expected = torch.autograd.grad(log_density.sum(), y)[0]
    actual_density, actual = family.log_prob_and_score(y, torch.ones_like(sigma), sigma)
    torch.testing.assert_close(actual_density, log_density, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_bank_version_changes_evaluation_only():
    cfg = small_config()
    original = MatchedMomentFamily(cfg, "gmm", 1.)
    changed = MatchedMomentFamily(replace(cfg, bank_version="independent-gated-test"), "gmm", 1.)
    assert bank_seed(original, "test") != bank_seed(changed, "test")
    assert bank_seed(changed, "test") != bank_seed(changed, "validation")
    assert make_bank(original, "test").fingerprint != make_bank(changed, "test").fingerprint
    torch.testing.assert_close(original.sample_cpu(8, torch.Generator().manual_seed(42)),
                               changed.sample_cpu(8, torch.Generator().manual_seed(42)), atol=0, rtol=0)


def test_gmm_gate_resume_matches_uninterrupted(tmp_path):
    cfg = small_config()
    family = MatchedMomentFamily(cfg, "gmm", 1.)
    arm = gated_arm("fourier", .5)
    validation = make_bank(family, "validation")
    provenance = {"test": "resume"}
    full = train_arm(family, arm, 42, validation, tmp_path / "full", provenance)
    paused = train_arm(family, arm, 42, validation, tmp_path / "split", provenance, stop_after=1)
    assert not paused["completed"]
    resumed = train_arm(family, arm, 42, validation, tmp_path / "split", provenance)
    assert full["final_ema_backbone_sha256"] == resumed["final_ema_backbone_sha256"]
    assert full["final_data_rng_sha256"] == resumed["final_data_rng_sha256"]
    assert full["validation"][-1]["score_error"] == resumed["validation"][-1]["score_error"]
    assert "test" not in full
    with pytest.raises(ValueError, match="mismatch"):
        train_arm(family, arm, 42, validation, tmp_path / "split", {"test": "different"})


def test_gate_selection_uses_validation_and_one_shared_switch():
    results = []
    for switch in (.5, 1.):
        for covariance in ("scalar", "fourier"):
            arm = gated_arm(covariance, switch)
            results.append(dict(arm=asdict(arm), completed=True, step=4, spectrum_lambda=1., seed=42,
                                validation=[dict(split="validation", step=4, score_error=2-switch)],
                                test=dict(score_error=switch)))
    selection = select_gate(results, [.5, 1.])
    assert selection["sigma_switch"] == 1.
    assert selection["candidates"][0]["n_runs"] == 2
    with pytest.raises(ValueError, match="Unpaired"):
        select_gate(results[:-1], [.5, 1.])


def test_toy_backbone_unchanged_between_arms():
    family = MatchedMomentFamily(small_config(), "gmm", 1.)
    baseline = make_model(family, "score", 42)
    gated = make_model(family, gated_arm("fourier"), 42)
    for name, tensor in baseline.backbone.state_dict().items():
        torch.testing.assert_close(tensor, gated.backbone.state_dict()[name], atol=0, rtol=0)
    assert gated.embedding == "fourier"
    assert gated.loss_objective == "normalized_residual"
