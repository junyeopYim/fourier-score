import json
from pathlib import Path

import pytest
import torch

from fourier_score.checkpoints import FORMAT
from fourier_score.config import experiment_name, load_config
from fourier_score.utils import source_hash
from scripts.run_mnist_remaining import ARMS, completed_checkpoint, plan_run, run_config


def config(tmp_path, name="fourier_normalized"):
    arm = next(a for a in ARMS if a.name == name)
    return run_config(load_config("configs/mnist.json"), arm, 0, tmp_path / "new", 100000, "cpu")


def save_checkpoint(folder, cfg, step=100000, source=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.resolved.json").write_text(json.dumps(cfg))
    path = folder / "last.pt"
    torch.save(dict(format=FORMAT, kind="training", step=step, config=cfg,
                    source_sha256=source if source is not None else source_hash()), path)
    return path


def test_suite_contains_all_nineteen_paired_arms(tmp_path):
    assert len(ARMS) == len({a.name for a in ARMS}) == 19
    assert sum(a.objective == "normalized_residual" for a in ARMS) == 16
    configs = [run_config(load_config("configs/mnist.json"), a, 0, tmp_path, 100000, "cpu") for a in ARMS]
    assert len({experiment_name(c) for c in configs}) == 19
    for c in configs:
        assert c["arch"] == configs[0]["arch"]
        assert c["optimizer"] == configs[0]["optimizer"]
        assert c["data_loader"] == configs[0]["data_loader"]


def test_completed_control_is_reused_even_with_old_source(tmp_path):
    cfg = config(tmp_path, "score")
    folder = tmp_path / "old" / "custom_name"
    checkpoint = save_checkpoint(folder, cfg, source="historical training source")
    plan = plan_run(cfg, [folder.parent], 100000)
    assert plan["action"] == "reuse"
    assert plan["checkpoint"] == checkpoint


def test_logs_and_wrong_budget_do_not_count_as_completed(tmp_path):
    cfg = config(tmp_path)
    folder = tmp_path / "old" / "partial"
    save_checkpoint(folder, cfg, step=30000)
    (folder / "metrics.jsonl").write_text('{"step":100000}\n')
    assert completed_checkpoint(folder, cfg, 100000) is None
    assert plan_run(cfg, [folder.parent], 100000)["action"] == "train"


def test_unrelated_configurations_in_search_root_are_ignored(tmp_path):
    cfg = config(tmp_path)
    unrelated = tmp_path / "old" / "latent_run"
    unrelated.mkdir(parents=True)
    (unrelated / "config.resolved.json").write_text('{"model":{"type":"latent"}}')
    assert plan_run(cfg, [unrelated.parent], 100000)["action"] == "train"


@pytest.mark.parametrize("field,value", [("seed", 1), ("batch_size", 64), ("gate", "log_sigma")])
def test_incompatible_control_is_not_reused(tmp_path, field, value):
    cfg = config(tmp_path)
    other = json.loads(json.dumps(cfg))
    if field == "batch_size":
        other["data_loader"]["args"][field] = value
    elif field == "gate":
        other["fourier"]["gate"]["mode"] = value
    else:
        other[field] = value
    folder = tmp_path / "old" / "mismatch"
    save_checkpoint(folder, other)
    assert plan_run(cfg, [folder.parent], 100000)["action"] == "train"


def test_current_partial_run_resumes_but_changed_source_does_not(tmp_path):
    cfg = config(tmp_path)
    folder = Path(cfg["trainer"]["save_dir"]) / experiment_name(cfg)
    checkpoint = save_checkpoint(folder, cfg, step=10000)
    plan = plan_run(cfg, [], 100000)
    assert plan["action"] == "resume" and plan["checkpoint"] == checkpoint
    save_checkpoint(folder, cfg, step=10000, source="different")
    with pytest.raises(ValueError, match="Cannot resume"):
        plan_run(cfg, [], 100000)


def test_existing_incomplete_output_is_preserved(tmp_path):
    cfg = config(tmp_path)
    folder = Path(cfg["trainer"]["save_dir"]) / experiment_name(cfg)
    folder.mkdir(parents=True)
    old_log = folder / "metrics.jsonl"
    old_log.write_text("keep partial history\n")
    with pytest.raises(ValueError, match="without last.pt"):
        plan_run(cfg, [], 100000)
    assert old_log.read_text() == "keep partial history\n"
