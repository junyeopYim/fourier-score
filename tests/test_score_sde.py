"""One small download → official EMA → public inference regression."""

import hashlib
import io

import pytest
import torch

from fourier_score.checkpoints import load_inference
from fourier_score.evaluation import evaluate_checkpoint
from fourier_score.model import build_model
from scripts import download_score_sde as download
from scripts import import_score_sde as importer


def test_verified_official_ema_roundtrip(cfg, tmp_path, monkeypatch, http_server):
    cfg["loss"]["type"] = "score"
    source = build_model(cfg).backbone
    parameters = dict(source.named_parameters())
    expected = {
        name: value.detach().clone() + (0.001 if value.requires_grad else 0)
        for name, value in parameters.items()
    }
    state = {
        # Real public releases omit this unused deterministic buffer.
        "model": {
            "module." + name: value
            for name, value in source.state_dict().items()
            if name != "sigmas"
        },
        "ema": {
            "shadow_params": [
                expected[name] for name, value in parameters.items() if value.requires_grad
            ]
        },
        "step": 12,
    }
    payload = io.BytesIO()
    torch.save(state, payload)
    checksum = hashlib.sha256(payload.getvalue()).hexdigest()
    model_name = "cifar10_ncsnpp_continuous"
    plan = download.download_plan(model_name, tmp_path)
    with http_server({"/checkpoint": payload.getvalue(), "/config": b"fixture config"}) as (base, calls):
        plan["checkpoint_url"] = base + "/checkpoint"
        plan["config_sources"] = {"upstream/config.py": base + "/config"}
        monkeypatch.setattr(download, "download_plan", lambda *args: plan)
        monkeypatch.setattr(importer, "download_plan", lambda *args: plan)
        monkeypatch.setattr(importer, "load_config", lambda *args: cfg)
        bundle = download.download_model(model_name, tmp_path, checksum)
        assert len(calls) == 2

    # Completed bundles and exports must work without the source server.
    assert download.download_model(model_name, tmp_path, checksum) == bundle
    target = importer.convert_model(model_name, tmp_path)
    assert importer.convert_model(model_name, tmp_path) == target
    model, _, _, converted = load_inference(target, device="cpu")
    assert converted["kind"] == "ema" and converted["stats"] is None
    assert converted["step"] == 12
    assert converted["pretrained_source"]["checkpoint_sha256"] == checksum
    for name, parameter in model.backbone.named_parameters():
        assert torch.equal(parameter, expected[name])
    assert torch.equal(model.backbone.sigmas, source.sigmas)
    level = model.process.level(torch.tensor([0.5]))
    y = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        # The upstream score divides the network output by sigma exactly once.
        raw = model.backbone(y, level.sigma)
        torch.testing.assert_close(model(y, level), raw / level.sigma[:, None, None, None])
    report = evaluate_checkpoint(target, tmp_path / "dsm.json", device="cpu")
    assert report["pretrained_source"] == converted["pretrained_source"]
    assert not (tmp_path / "stats").exists()

    # Reject ambiguous/incomplete EMA mapping and changed original bytes.
    bad = {**state, "ema": {"shadow_params": state["ema"]["shadow_params"][:-1]}}
    with pytest.raises(ValueError, match="EMA parameter count"):
        importer.load_official_ema(bad, cfg)
    bad = {**state, "model": dict(reversed(list(state["model"].items())))}
    with pytest.raises(ValueError, match="parameter order"):
        importer.load_official_ema(bad, cfg)
    (bundle / plan["checkpoint_name"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Downloaded file changed"):
        download.download_model(model_name, tmp_path)
