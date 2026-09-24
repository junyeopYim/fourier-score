"""Native latent contracts and paired training, without downloading weights."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("einops")
yaml = pytest.importorskip("yaml")
pytest.importorskip("cv2")

from fourier_score.ldm.config import (
    PAPER_TRAINING,
    defaults,
    load_config,
    load_spec,
    lr_multiplier,
)
from fourier_score.ldm.data import (
    draw_latents,
    image_splits,
    open_cache,
    prepare_cache,
)
from fourier_score.ldm.evaluation import evaluate_latents, generate
from fourier_score.ldm.first_stage import (
    FrozenFirstStage,
    file_sha256,
    load_first_stage,
)
from fourier_score.ldm.model import (
    Denoiser,
    Schedule,
    load_public_denoiser,
    sample_latents,
)
from fourier_score.ldm.training import Trainer, load_trained
from fourier_score.ldm.upstream.ema import LitEma
from fourier_score.utils import load_checkpoint


def test_pinned_computational_sources_and_training_yamls():
    root = Path("fourier_score/ldm/upstream")
    manifest = json.loads((root / "PROVENANCE.json").read_text())
    for name, item in manifest["files"].items():
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["local_sha256"]
        original = (
            data.decode()
            .replace(
                "from .util import", "from ldm.modules.diffusionmodules.util import"
            )
            .replace("from .attention import", "from ldm.modules.attention import")
            .replace(
                "from .factory import instantiate_from_config",
                "from ldm.util import instantiate_from_config",
            )
        )
        assert hashlib.sha256(original.encode()).hexdigest() == item["upstream_sha256"]
    for name, item in manifest["configs"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == item["sha256"]


@pytest.fixture(params=["AutoencoderKL", "VQModelInterface"])
def latent_experiment(tmp_path, request):
    cfg = defaults("ffhq")
    source = yaml.safe_load(Path("configs/ldm/upstream/ffhq.yaml").read_text())
    p = source["model"]["params"]
    kl = request.param == "AutoencoderKL"
    p.update(image_size=4, channels=2, timesteps=10, scale_by_std=kl)
    p["unet_config"]["params"] = {
        "image_size": 4,
        "in_channels": 2,
        "out_channels": 2,
        "model_channels": 32,
        "attention_resolutions": [1, 2],
        "num_res_blocks": 1,
        "channel_mult": [1, 2],
        "num_heads": 4,
    }
    p["first_stage_config"] = {
        "target": "ldm.models.autoencoder." + request.param,
        "params": {
            "embed_dim": 2,
            "ddconfig": {
                "double_z": kl,
                "z_channels": 2,
                "resolution": 8,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 32,
                "ch_mult": [1, 2],
                "num_res_blocks": 1,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
            "lossconfig": {"target": "torch.nn.Identity"},
        },
    }
    if not kl:
        p["first_stage_config"]["params"]["n_embed"] = 16
    source["data"]["params"]["train"]["params"]["size"] = 8
    spec_path = tmp_path / "native.yaml"
    spec_path.write_text(yaml.safe_dump(source))
    cfg.update(
        upstream_config=str(spec_path),
        device="cpu",
        name="tiny_{parameterization}_s{seed}",
    )
    cfg["sampling"].update(
        steps=5, num_samples=2, batch_size=2, decode_batch_size=1, eta=0.0
    )
    cfg["evaluation"].update(batch_size=2, max_images=2, noise_bins=2, frequency_bins=2)
    cfg["training"].update(
        iterations=2,
        batch_size=2,
        microbatch_size=1,
        save_dir=str(tmp_path / "runs"),
        save_every=1,
        snapshot_every=0,
        eval_every=1,
        log_every=1,
        console="quiet",
        lr=1e-3,
    )
    cfg["backend"]["cpu_threads"] = 2
    cfg["cache"].update(dir=str(tmp_path / "cache"), batch_size=2)
    root = tmp_path / "images"
    root.mkdir()
    rng = np.random.default_rng(17)
    for i in range(6):
        Image.fromarray(rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)).save(
            root / f"{i}.png"
        )
    for split, ids in (("train", range(4)), ("validation", range(4, 6))):
        listing = tmp_path / f"{split}.txt"
        listing.write_text("".join(f"{i}.png\n" for i in ids))
        cfg["data"][split + "_list"] = str(listing)
    cfg["data"].update(root=str(root), random_flip=True)
    spec = load_spec(cfg)
    torch.manual_seed(123)
    stage = FrozenFirstStage(spec)
    checkpoint = tmp_path / "public.ckpt"
    state = {"first_stage_model." + k: v for k, v in stage.state_dict().items()}
    if kl:
        state["scale_factor"] = torch.tensor(0.7)
    torch.save({"state_dict": state}, checkpoint)
    cfg["first_stage"]["checkpoint"] = str(checkpoint)
    return cfg, spec


def test_paper_presets_and_l1_are_preserved():
    for model, (batch, lr, updates) in PAPER_TRAINING.items():
        cfg = load_config(f"configs/ldm/{model}.json")
        spec = load_spec(cfg)
        assert (
            cfg["training"]["batch_size"],
            cfg["training"]["lr"],
            cfg["training"]["iterations"],
        ) == (batch, lr, updates)
        assert spec["timesteps"] == 1000
        assert spec["loss_type"] == ("l1" if model == "lsun_churches" else "l2")
        assert lr_multiplier(spec, 0) == (1e-6 if model == "lsun_churches" else 1)
    assert (
        load_spec(load_config("configs/ldm/lsun_churches_l2.json"))["loss_type"] == "l2"
    )
    with pytest.raises(ValueError, match="Unknown"):
        load_config("configs/ldm/ffhq.json", ["training.batch_szie=3"])


@pytest.mark.parametrize("latent_experiment", ["AutoencoderKL"], indirect=True)
def test_native_beta_and_ddim_endpoint(latent_experiment):
    cfg, spec = latent_experiment
    schedule = Schedule(spec)
    expected = (
        np.linspace(
            np.sqrt(spec["linear_start"]),
            np.sqrt(spec["linear_end"]),
            spec["timesteps"],
        )
        ** 2
    )
    np.testing.assert_allclose(schedule.betas.numpy(), expected, rtol=1e-7)
    assert not np.allclose(
        expected,
        np.linspace(spec["linear_start"], spec["linear_end"], spec["timesteps"]),
    )
    model = Denoiser(spec).eval()  # native zero-initialized output convolution
    initial = torch.full((1, *spec["latent_shape"]), 10.0)
    output, calls = sample_latents(
        model, 1, cfg["sampling"], "cpu", torch.Generator().manual_seed(9), initial
    )
    # Native uniform DDIM visits 1,3,5,7,9; terminal previous alpha is alpha[0], not 1.
    expected_output = (
        initial * (schedule.alphas_cumprod[0] / schedule.alphas_cumprod[9]).sqrt()
    )
    torch.testing.assert_close(output, expected_output)
    assert output.min() > 1 and calls == 5


def test_frozen_loader_cache_and_train_only_moments(latent_experiment):
    cfg, spec = latent_experiment
    stage, first = load_first_stage(cfg["first_stage"]["checkpoint"], spec)
    stage.train(True)
    assert not stage.training and not any(p.requires_grad for p in stage.parameters())
    manifest = prepare_cache(cfg, spec, torch.device("cpu"), progress=lambda _: None)
    assert open_cache(cfg, spec)["identity"] == manifest["identity"]
    cached = np.load(Path(cfg["cache"]["dir"]) / "train.npy")
    ds = image_splits(cfg, spec)["train"]
    torch.testing.assert_close(
        torch.tensor(cached[0, 1]),
        stage.encode_parameters(ds[0][None].flip(-1))[0],
        atol=2e-6,
        rtol=2e-5,
    )
    assert not np.allclose(cached[0, 1], cached[0, 0, :, :, ::-1])
    params = torch.tensor(cached).flatten(0, 1).double()
    if first["kind"] == "AutoencoderKL":
        means, logvar = params.chunk(2, 1)
        within = (logvar.exp() * first["scale_factor"] ** 2).mean((0, 2, 3))[
            :, None, None
        ]
    else:
        means, within = params, 0
    means = means * first["scale_factor"]
    mean = means.mean(0)
    power = torch.fft.fft2(means - mean, norm="ortho").abs().square().mean(0) + within
    stats = load_checkpoint(Path(cfg["cache"]["dir"]) / "stats.pt")
    torch.testing.assert_close(stats["mean"].double(), mean)
    torch.testing.assert_close(
        stats["power"].double(), power.clamp_min(cfg["cache"]["power_floor"])
    )
    assert stats["n_effective"] == 8  # four TRAIN images, both encoded views
    generator = torch.Generator().manual_seed(12)
    batch = torch.tensor(cached[:2])
    if first["kind"] == "AutoencoderKL":
        assert not torch.equal(
            draw_latents(batch, first, generator), draw_latents(batch, first, generator)
        )
    changed = copy.deepcopy(cfg)
    changed["data"]["random_flip"] = False
    with pytest.raises(ValueError, match="mismatch"):
        open_cache(changed, spec)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_first_stage(
            cfg["first_stage"]["checkpoint"], spec, expected_sha256="wrong"
        )
    cache_file = Path(cfg["cache"]["dir"]) / "validation.npy"
    with cache_file.open("r+b") as f:
        f.seek(-1, 2)
        value = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="Corrupt"):
        open_cache(cfg, spec)


@pytest.mark.parametrize("latent_experiment", ["AutoencoderKL"], indirect=True)
def test_all_arms_initialize_the_same_and_sampling_uses_adapter(latent_experiment):
    cfg, spec = latent_experiment
    prepare_cache(cfg, spec, torch.device("cpu"), progress=lambda _: None)
    hashes = []
    initial = torch.randn(1, *spec["latent_shape"])
    outputs = {}
    for arm in (
        "epsilon",
        "scalar_gaussian",
        "fourier_gaussian",
    ):
        c = copy.deepcopy(cfg)
        c["parameterization"] = arm
        trainer = Trainer(c, spec)
        hashes.append(trainer.initial_hash)
        outputs[arm], _ = sample_latents(
            trainer.model.eval(),
            1,
            cfg["sampling"],
            "cpu",
            torch.Generator().manual_seed(1),
            initial,
        )
        values = trainer.train_step(trainer.stream.next_batch())
        assert np.isfinite(values["loss"]) and values["grad_norm"] > 0
    assert len(set(hashes)) == 1
    assert not torch.allclose(outputs["epsilon"], outputs["fourier_gaussian"])


def test_training_resume_and_ema_decode_end_to_end(latent_experiment, tmp_path):
    cfg, spec = latent_experiment
    cfg["parameterization"] = "fourier_gaussian"
    before = file_sha256(cfg["first_stage"]["checkpoint"])
    prepare_cache(cfg, spec, torch.device("cpu"), progress=lambda _: None)
    full_cfg = copy.deepcopy(cfg)
    full_cfg["name"] = "uninterrupted"
    full = load_checkpoint(Trainer(full_cfg, spec).train())
    first_cfg = copy.deepcopy(cfg)
    first_cfg["training"]["iterations"] = 1
    first = load_checkpoint(Trainer(first_cfg, spec).train())
    output = Trainer(cfg, spec, first).train()
    resumed = load_checkpoint(output)
    for group in ("model", "ema"):
        for key in full[group]:
            assert torch.equal(full[group][key], resumed[group][key]), (group, key)
    assert torch.equal(full["generator"], resumed["generator"])
    assert full["stream"] == resumed["stream"]
    assert before == file_sha256(cfg["first_stage"]["checkpoint"])
    model, resolved, device, state = load_trained(output)
    stage, _ = load_first_stage(
        cfg["first_stage"]["checkpoint"], spec, expected_sha256=before
    )
    metrics = evaluate_latents(model, resolved, state["cache"], device)
    assert metrics["n_images"] == 2 and np.isfinite(metrics["epsilon_mse"])
    report = generate(model, stage, resolved, tmp_path / "samples", {"test": True})
    assert report["complete"] and report["nfe_per_sample"] == 5
    assert len(list((tmp_path / "samples/png").glob("*.png"))) == 2
    assert np.load(tmp_path / "samples/samples_00000.npz")["samples"].shape == (
        2,
        8,
        8,
        3,
    )
    with pytest.raises(ValueError, match="cannot change"):
        load_trained(output, ["parameterization=epsilon"])


@pytest.mark.parametrize("latent_experiment", ["AutoencoderKL"], indirect=True)
@pytest.mark.parametrize("source_change", ["edited", "missing"])
def test_checkpoint_keeps_startup_source_when_checkout_changes(
    latent_experiment, monkeypatch, source_change
):
    cfg, spec = latent_experiment
    prepare_cache(cfg, spec, torch.device("cpu"), progress=lambda _: None)
    trainer = Trainer(cfg, spec)
    startup_hash = trainer.env["source_sha256"]
    trainer.train_step(trainer.stream.next_batch())

    def changed_source_hash():
        if source_change == "missing":
            raise FileNotFoundError("Source file moved while training was running")
        return "0" * 64

    monkeypatch.setattr("fourier_score.ldm.training.source_hash", changed_source_hash)
    try:
        trainer.save(snapshot=True)
    finally:
        trainer.console.close()
    checkpoint = load_checkpoint(trainer.out / "last.pt")
    snapshot = load_checkpoint(trainer.out / f"ema_{trainer.step:09d}.pt")
    for state in (checkpoint, snapshot):
        assert state["step"] == 1
        assert state["source_sha256"] == startup_hash
        assert state["source_sha256"] == state["environment"]["source_sha256"]

    # Freezing save-time provenance must not bypass the resume source guard.
    if source_change == "edited":
        with pytest.raises(ValueError, match="same source revision"):
            Trainer(cfg, spec, checkpoint)


@pytest.mark.parametrize("latent_experiment", ["AutoencoderKL"], indirect=True)
def test_public_ema_selection_and_strict_native_keys(latent_experiment, tmp_path):
    _cfg, spec = latent_experiment
    model = Denoiser(spec)
    ema = LitEma(model)
    with torch.no_grad():
        model.diffusion_model.out[-1].weight.fill_(0.2)
    public = {
        "model.diffusion_model." + k: v
        for k, v in model.diffusion_model.state_dict().items()
    }
    public.update({"model_ema." + k: v for k, v in ema.state_dict().items()})
    public.update(dict(model.schedule.named_buffers()))
    path = tmp_path / "denoiser.ckpt"
    torch.save({"state_dict": public}, path)
    raw = load_public_denoiser(path, spec, "raw")
    averaged = load_public_denoiser(path, spec, "ema")
    assert raw.diffusion_model.out[-1].weight.mean() > 0.19
    assert averaged.diffusion_model.out[-1].weight.count_nonzero() == 0
    del public["model.diffusion_model.out.2.weight"]
    torch.save({"state_dict": public}, path)
    with pytest.raises(ValueError, match="Missing public"):
        load_public_denoiser(path, spec, "raw")


def test_cli_dry_run_has_no_data_or_output_side_effects(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "ldm.py",
            "compare",
            "-c",
            "configs/ldm/lsun_churches_l2.json",
            "--seeds",
            "7",
            "--dry-run",
            "--set",
            f"cache.dir={tmp_path}/cache",
            "--set",
            f"training.save_dir={tmp_path}/runs",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    jobs = json.loads(result.stdout)
    assert len(jobs) == 3
    assert all("l2" in job["run_dir"] for job in jobs)
    assert not (tmp_path / "cache").exists() and not (tmp_path / "runs").exists()


def test_ldm_fid_small_reconstruction_set_avoids_undefined_inception_score(
    tmp_path, monkeypatch
):
    import importlib.metadata
    import sys
    from types import SimpleNamespace

    from fourier_score.ldm.cli import main

    images = tmp_path / "images"
    images.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), color=(i * 30, 10, 50)).save(images / f"{i}.png")

    def metrics(**kwargs):
        result = {"frechet_inception_distance": 0.0}
        if kwargs["isc"]:
            # torch-fidelity's default ten IS splits cannot partition two images.
            result["inception_score_mean"] = float("nan")
        return result

    monkeypatch.setitem(
        sys.modules, "torch_fidelity", SimpleNamespace(calculate_metrics=metrics)
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test")
    output = tmp_path / "fid.json"
    main(["fid", "--real", str(images), "--generated", str(images), "-o", str(output)])
    report = json.loads(output.read_text())
    assert report["metrics"] == {"frechet_inception_distance": 0.0}
    assert report["real"]["count"] == 2
