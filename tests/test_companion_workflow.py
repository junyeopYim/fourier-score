"""Public recipes must retain controlled comparisons and working evaluation."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from fourier_score.config import experiment_name, load_config
from fourier_score.evaluation import evaluate_images
from fourier_score.images import write_png


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
