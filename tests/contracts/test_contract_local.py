"""Golden contract: local. Values recorded from the epoch-0 tree.

Needs FOURIER_GOLDEN_ROOT (the original checkout with real checkpoints, MNIST,
statistics and latent caches; read-only).  Skipped when it is unset (CI).
"""

import warnings

import pytest

import golden_probes as gp
from golden_probes import local

pytestmark = pytest.mark.golden_local

KEYS = gp.load_group("local")


def require_golden_root(ctx):
    if ctx.golden_root is None:
        pytest.skip("local-only golden; set FOURIER_GOLDEN_ROOT to the original checkout")


@pytest.mark.parametrize("key", KEYS)
def test_golden(key, golden_ctx):
    require_golden_root(golden_ctx)
    if key == local.STORED_CONFIGS_KEY:
        report = local.check_stored_configs(golden_ctx)
        skipped = report["golden_only"] + report["current_only"]
        if skipped:
            warnings.warn(
                f"stored_configs compared {len(report['compared'])} runs; skipped "
                f"{len(report['golden_only'])} golden-only {report['golden_only']} and "
                f"{len(report['current_only'])} new {report['current_only']}",
                stacklevel=1,
            )
        return
    gp.check("local", key, golden_ctx)


def test_recorded_pairing_matches_live_mnist_runs(golden_ctx):
    """The golden itself must hold the live MNIST seed-0 hashes (known-good record)."""
    require_golden_root(golden_ctx)
    recorded = gp.load_golden("local")["probes"]["local.mnist_pairing"]
    assert recorded["exact"]["same_initial_hash_across_arms"] is True
    assert set(recorded["exact"]["architecture_sha256"].values()) == {local.MNIST_ARCHITECTURE_SHA256}
    for arm in local.MNIST_ARMS:
        assert recorded["numeric"][arm]["initial_backbone_sha256"] == local.MNIST_INITIAL_BACKBONE_SHA256


def test_recorded_checkpoints_warn_and_verify(golden_ctx):
    """Every recorded real checkpoint loaded strictly with the expected audit facts."""
    require_golden_root(golden_ctx)
    recorded = gp.load_golden("local")["probes"]["local.real_checkpoints"]["exact"]
    for rel in (*local.IMAGE_CHECKPOINTS, local.LDM_CHECKPOINT):
        assert recorded[rel]["source_hash_warning"] is True, rel
    assert recorded["saved/mnist_ve_score_s0/last.pt"]["resume_signature_matches"] is True
    gmm = recorded[local.GMM_CHECKPOINT]
    assert gmm["file_sha256_matches_audit"] and gmm["final_ema_matches_audit"]
    identity = gp.load_golden("local")["probes"]["local.ldm_cache_identity"]["exact"]
    assert identity["identity_equal"] and identity["cache_unchanged"]
