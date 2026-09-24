"""Preserve the approved design and keep new training separate from controls."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from fourier_score.gmm import GMMConfig, log_gate_arm, shaped_gate_arm
from experiments.gmm.pipeline import reuse_directory
from experiments.gmm.registry import EXPERIMENTS

NEW_MODES = EXPERIMENTS["log_gates"].new_modes
CFG = GMMConfig(preset="test")


def test_log_experiment_defaults_match_prior_design_artifact():
    proposal = Path(__file__).resolve().parents[1] / "assets/log_gate_design/design.json"
    expected = json.loads(proposal.read_text())["arms"]
    actual = [asdict(log_gate_arm("fourier", mode)) for mode in NEW_MODES]
    assert actual == expected


@pytest.mark.parametrize("mode", NEW_MODES)
def test_new_log_arms_never_reuse_control_weights(tmp_path, mode):
    arm = log_gate_arm("fourier", mode)
    checkpoint = tmp_path / "gmm_lambda1" / f"{arm.name}_seed42" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    assert reuse_directory([tmp_path], CFG, 1., 42, arm, new_modes=NEW_MODES) is None


def test_prior_sigma_gate_is_a_control_and_requires_one_checkpoint(tmp_path):
    arm = shaped_gate_arm("fourier", "linear_sigma")
    roots = [tmp_path / "first", tmp_path / "second"]
    relative = Path("gmm_lambda1") / f"{arm.name}_seed42" / "checkpoint.pt"
    with pytest.raises(ValueError, match="found 0"):
        reuse_directory(roots, CFG, 1., 42, arm, new_modes=NEW_MODES)
    first = roots[0] / relative
    first.parent.mkdir(parents=True)
    first.touch()
    assert reuse_directory(roots, CFG, 1., 42, arm, new_modes=NEW_MODES) == roots[0]
    second = roots[1] / relative
    second.parent.mkdir(parents=True)
    second.touch()
    with pytest.raises(ValueError, match="found 2"):
        reuse_directory(roots, CFG, 1., 42, arm, new_modes=NEW_MODES)
