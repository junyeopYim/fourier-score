"""Public recipes must retain controlled comparisons and working evaluation."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from evaluate import main as evaluate_main
from fourier_score.config import experiment_name, load_config
from fourier_score.evaluation import evaluate_images
from fourier_score.images import write_png
from fourier_score.method import COMPARISON_OBJECTIVES
from fourier_score.training import Trainer
from scripts.run_comparison import comparison_runs


def test_parameterization_alias_preserves_v1_protocol(tmp_path):
    old = load_config("configs/smoke.json", ["loss.type=scalar_gaussian"])
    new = load_config("configs/smoke.json", ["parameterization=scalar_gaussian"])
    assert old == new
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"parameterization": "scalar_gaussian"}))
    assert load_config(path)["loss"]["type"] == "scalar_gaussian"
    path.write_text(
        json.dumps({"parameterization": "score", "loss": {"type": "diffusion"}})
    )
    with pytest.raises(ValueError, match="disagree"):
        load_config(path)


def test_long_cifar_protocols_change_only_budget_and_name():
    short = load_config("configs/cifar10_950k.json")
    long = load_config("configs/cifar10_1m3.json")
    assert short["trainer"]["iterations"] == 950000
    assert long["trainer"]["iterations"] == 1300000
    assert short["data_loader"]["args"]["validation_size"] == 0
    assert experiment_name(short) != experiment_name(long)
    long["name"] = short["name"]
    long["trainer"]["iterations"] = short["trainer"]["iterations"]
    assert short == long


def test_cifar_names_separate_budgets_splits_arms_and_seeds():
    outputs = set()
    for config in (
        "cifar10",
        "cifar10_full",
        "cifar10_paper950k",
        "cifar10_ablation",
        "cifar10_950k",
        "cifar10_1m3",
        "cifar10_ddpm",
    ):
        path = f"configs/{config}.json"
        runs = comparison_runs(path, seeds=[42, 43])
        assert len(runs) == 2 * len(COMPARISON_OBJECTIVES)
        for run in runs:
            cfg = run["config"]
            direct = load_config(
                path, [f"parameterization={cfg['loss']['type']}", f"seed={cfg['seed']}"]
            )
            assert experiment_name(cfg) == experiment_name(direct)
            assert run["output"] not in outputs
            outputs.add(run["output"])


@pytest.mark.parametrize(
    "name",
    [
        "../bad",
        "{seed}",
        "{parameterization}",
        "{seed.__class__}_{parameterization}",
        "{seed:03d}_{parameterization}",
        "{seed!r}_{parameterization}",
    ],
)
def test_run_templates_reject_ambiguous_or_unsafe_names(name):
    with pytest.raises(ValueError):
        load_config("configs/smoke.json", [f"name={name}"])


def test_train_dry_run_accepts_public_flags_without_loading_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "-c",
            "configs/cifar10_950k.json",
            "--parameterization",
            "score",
            "--download",
            "--dry-run",
            "--set",
            f"data_loader.args.root={tmp_path}/data",
            "--set",
            f"trainer.save_dir={tmp_path}/runs",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cfg = json.loads(result.stdout)
    assert cfg["loss"]["type"] == "score"
    assert cfg["data_loader"]["args"]["download"] is True
    assert experiment_name(cfg) == "cifar10_950k_full_score_s42"
    assert not list(tmp_path.iterdir())


def test_evaluate_entry_point_loads_ema_and_preserves_checkpoint(cfg, tmp_path, capsys):
    trainer = Trainer(cfg)
    trainer.train()
    checkpoint = trainer.out / "last.pt"
    before = checkpoint.read_bytes()
    output = tmp_path / "evaluation.json"
    capsys.readouterr()
    evaluate_main(["dsm", "-r", str(checkpoint), "-o", str(output), "--device", "cpu"])
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["weights"] == "EMA"
    assert report["step"] == 3 and report["split"] == "validation"
    assert report["dsm_pixel_mean"] >= 0
    assert checkpoint.read_bytes() == before


def test_fid_adapter_preserves_input_roles_and_provenance(tmp_path, monkeypatch):
    real = tmp_path / "real"
    generated = tmp_path / "generated"
    for folder in (real, generated):
        folder.mkdir()
        for index in range(2):
            write_png(
                torch.full((8, 8, 3), index, dtype=torch.uint8).numpy(),
                folder / f"{index}.png",
            )
    calls = []

    def calculate_metrics(**kwargs):
        calls.append(kwargs)
        return {"frechet_inception_distance": 1.0}

    monkeypatch.setitem(
        sys.modules,
        "torch_fidelity",
        SimpleNamespace(calculate_metrics=calculate_metrics),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-version")
    result = evaluate_images(real, generated, tmp_path / "fid.json")
    assert calls[0]["input1"] == str(generated.resolve())
    assert calls[0]["input2"] == str(real.resolve())
    assert calls[0]["fid"] and calls[0]["isc"] and not calls[0]["cuda"]
    assert result["real"]["count"] == result["generated"]["count"] == 2
    assert result["version"] == "test-version"
    assert json.loads((tmp_path / "fid.json").read_text()) == result
