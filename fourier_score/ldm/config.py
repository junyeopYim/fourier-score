"""Pinned LDM architectures and explicit, paper-derived training budgets."""

import copy
import hashlib
import json
import math
from pathlib import Path

from fourier_score.config import deep_merge
from fourier_score.method import GAUSSIAN_OBJECTIVES
from fourier_score.utils import ROOT

REVISION = "a506df5756472e2ebaf9078affdde2c4f1502cd4"
PAPER = "https://arxiv.org/html/2112.10752#A5.T12"
PARAMETERIZATIONS = ("epsilon", *GAUSSIAN_OBJECTIVES)
# Table 12: effective batch, actual optimizer learning rate, optimizer updates.
PAPER_TRAINING = {
    "ffhq": (42, 8.4e-5, 635_000),
    "celebahq": (48, 9.6e-5, 410_000),
    "lsun_churches": (96, 5e-5, 500_000),
    "lsun_bedrooms": (48, 9.6e-5, 1_900_000),
}


def defaults(model):
    if model not in PAPER_TRAINING:
        raise ValueError(f"Unknown LDM model: {model}")
    batch, lr, steps = PAPER_TRAINING[model]
    lsun = model.startswith("lsun_")
    data_root = {
        "ffhq": "data/ffhq",
        "celebahq": "data/celebahq",
        "lsun_churches": "data/lsun/churches",
        "lsun_bedrooms": "data/lsun/bedrooms",
    }[model]
    lists = {
        "ffhq": ("data/ffhqtrain.txt", "data/ffhqvalidation.txt"),
        "celebahq": ("data/celebahqtrain.txt", "data/celebahqvalidation.txt"),
        "lsun_churches": (
            "data/lsun/church_outdoor_train.txt",
            "data/lsun/church_outdoor_val.txt",
        ),
        "lsun_bedrooms": ("data/lsun/bedrooms_train.txt", "data/lsun/bedrooms_val.txt"),
    }[model]
    return {
        "schema_version": "fourier-ldm-v1",
        "model": model,
        "name": "ldm_{model}_{protocol}_{parameterization}_s{seed}",
        "protocol": "upstream",
        "parameterization": "epsilon",
        "seed": 42,
        "device": "auto",
        "upstream_config": None,
        "first_stage": {"checkpoint": f"pretrained/ldm/{model}/model.ckpt"},
        "data": {
            "root": data_root,
            "train_list": lists[0],
            "validation_list": lists[1],
            "preprocessing": "lsun_bicubic" if lsun else "faces_cv2",
            "random_flip": lsun,
        },
        "cache": {
            "dir": f"data/ldm_cache/{model}",
            "batch_size": 4,
            "num_workers": 0,
            "power_floor": 1e-4,
        },
        "training": {
            "iterations": steps,
            "batch_size": batch,
            "microbatch_size": 1,
            "lr": lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "ema_decay": 0.9999,
            "grad_clip": None,
            "gradient_checkpointing": False,
            "save_dir": "saved",
            "save_every": 10000,
            "snapshot_every": 50000,
            "eval_every": 10000,
            "log_every": 50,
            "console": "human",
        },
        "backend": {
            "precision": "fp32",
            "tf32": False,
            "cpu_threads": 4,
            "spectral_transform": "auto",
        },
        "sampling": {
            "method": "ddim",
            "steps": 500 if model == "celebahq" else 200,
            "eta": 1.0,
            "batch_size": 4,
            "decode_batch_size": 1,
            "num_samples": 50000,
            "seed": 17002,
        },
        "evaluation": {
            "batch_size": 4,
            "max_images": 1024,
            "seed": 17001,
            "noise_bins": 10,
            "frequency_bins": 6,
        },
    }


def override(cfg, entries):
    cfg = copy.deepcopy(cfg)
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected key=value: {entry}")
        path, raw = entry.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        obj = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in obj or not isinstance(obj[part], dict):
                raise ValueError(f"Unknown LDM configuration key: {path}")
            obj = obj[part]
        if parts[-1] not in obj:
            raise ValueError(f"Unknown LDM configuration key: {path}")
        obj[parts[-1]] = value
    return cfg


def validate(cfg):
    cfg = deep_merge(defaults(cfg["model"]), cfg)
    if cfg["schema_version"] != "fourier-ldm-v1":
        raise ValueError("Not an LDM experiment config")
    if cfg["protocol"] not in ("upstream", "l2"):
        raise ValueError("protocol must be upstream or l2 (a labeled loss-only change)")
    if cfg["parameterization"] not in PARAMETERIZATIONS:
        raise ValueError(f"parameterization must be one of {PARAMETERIZATIONS}")
    if cfg["data"]["preprocessing"] not in ("faces_cv2", "lsun_bicubic"):
        raise ValueError("Unknown preprocessing")
    if cfg["training"]["console"] not in ("human", "quiet", "json"):
        raise ValueError("Invalid console mode")
    if cfg["backend"]["precision"] != "fp32":
        raise ValueError("LDM currently requires fp32; mixed precision is not implicit")
    if cfg["backend"]["spectral_transform"] not in ("auto", "fft", "matmul", "cpu"):
        raise ValueError("Invalid spectral transform")

    def walk(ref, obj, path=""):
        for key, value in obj.items():
            expected = ref[key]
            label = f"{path}.{key}"
            if isinstance(expected, dict):
                walk(expected, value, label)
            elif expected is not None:
                valid = type(value) is type(expected)
                if type(expected) is float:
                    valid = type(value) in (int, float) and math.isfinite(value)
                if not valid:
                    raise ValueError(f"Invalid type/value: {label}")

    walk(defaults(cfg["model"]), cfg)
    for section, names in {
        "cache": ("batch_size",),
        "backend": ("cpu_threads",),
        "training": (
            "batch_size",
            "microbatch_size",
            "iterations",
            "save_every",
            "log_every",
        ),
        "sampling": ("steps", "batch_size", "decode_batch_size", "num_samples"),
        "evaluation": ("batch_size", "max_images", "noise_bins"),
    }.items():
        for key in names:
            if cfg[section][key] < 1:
                raise ValueError(f"{section}.{key} must be positive")
    for value in (
        cfg["seed"],
        cfg["cache"]["num_workers"],
        cfg["evaluation"]["seed"],
        cfg["sampling"]["seed"],
        cfg["training"]["eval_every"],
        cfg["training"]["snapshot_every"],
        cfg["evaluation"]["frequency_bins"],
    ):
        if value < 0:
            raise ValueError("Seeds, counts and intervals must be nonnegative")
    t = cfg["training"]
    if t["lr"] <= 0 or t["eps"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if len(t["betas"]) != 2 or any(
        type(b) not in (int, float) or not 0 <= b < 1 for b in t["betas"]
    ):
        raise ValueError("Invalid AdamW betas")
    if not 0 <= t["ema_decay"] < 1 or cfg["cache"]["power_floor"] <= 0:
        raise ValueError("Invalid EMA decay or power floor")
    if t["grad_clip"] is not None and (
        type(t["grad_clip"]) not in (float, int)
        or not math.isfinite(t["grad_clip"])
        or t["grad_clip"] <= 0
    ):
        raise ValueError("grad_clip must be null or positive")
    if (
        cfg["sampling"]["method"] not in ("ddim", "ddpm")
        or not 0 <= cfg["sampling"]["eta"] <= 1
    ):
        raise ValueError("Invalid sampler or eta")
    for section, keys in {
        "data": ("root", "train_list", "validation_list"),
        "cache": ("dir",),
        "first_stage": ("checkpoint",),
    }.items():
        if any(not cfg[section][k] for k in keys):
            raise ValueError(f"Empty {section} path")
    if cfg["upstream_config"] is not None and not isinstance(
        cfg["upstream_config"], str
    ):
        raise ValueError("upstream_config must be null or a YAML path")
    experiment_name(cfg)
    return cfg


def experiment_name(cfg):
    try:
        name = cfg["name"].format(
            **{k: cfg[k] for k in ("model", "protocol", "parameterization", "seed")}
        )
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError("Invalid LDM run name template") from error
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("Run name must be a single directory name")
    return name


def load_config(path, overrides=()):
    def read(p, seen=()):
        p = Path(p).resolve()
        if p in seen:
            raise ValueError("Configuration inheritance cycle")
        obj = json.loads(p.read_text())
        parent = obj.pop("extends", None)
        if parent:
            base = read(p.parent / parent, (*seen, p))
            return deep_merge(base, obj)
        return deep_merge(defaults(obj.get("model", "ffhq")), obj)

    return validate(override(read(path), overrides))


def load_spec(cfg):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Install LDM dependencies: uv sync --locked --extra ldm"
        ) from error
    path = Path(
        cfg["upstream_config"] or ROOT / "configs/ldm/upstream" / f"{cfg['model']}.yaml"
    )
    raw = path.read_bytes()
    original = yaml.safe_load(raw)
    p = original["model"]["params"]
    if (
        original["model"]["target"] != "ldm.models.diffusion.ddpm.LatentDiffusion"
        or p["cond_stage_config"] != "__is_unconditional__"
    ):
        raise ValueError("Only unconditional CompVis LatentDiffusion is supported")
    first = p["first_stage_config"]
    kind = first["target"].rsplit(".", 1)[-1]
    if kind not in ("AutoencoderKL", "VQModelInterface"):
        raise ValueError("Unsupported first stage")
    if (
        p["unet_config"]["target"]
        != "ldm.modules.diffusionmodules.openaimodel.UNetModel"
    ):
        raise ValueError("Unsupported denoiser")
    if p.get("parameterization", "eps") != "eps" or p.get("learn_logvar", False):
        raise ValueError("Only epsilon prediction with fixed log variance is supported")
    if p.get("beta_schedule", "linear") != "linear":
        raise ValueError(
            "Expected the CompVis linear (square-root interpolated) schedule"
        )
    if first["params"].get("remap") is not None:
        raise ValueError("Remapped VQ codebooks are unsupported")
    spec = {
        "model": cfg["model"],
        "upstream_revision": REVISION,
        "upstream_config_sha256": hashlib.sha256(raw).hexdigest(),
        "custom_upstream_config": cfg["upstream_config"] is not None,
        "unet": copy.deepcopy(p["unet_config"]["params"]),
        "first_stage": {"kind": kind, "params": copy.deepcopy(first["params"])},
        "image_size": original["data"]["params"]["train"]["params"]["size"],
        "latent_shape": [p["channels"], p["image_size"], p["image_size"]],
        "timesteps": p["timesteps"],
        "linear_start": p["linear_start"],
        "linear_end": p["linear_end"],
        "scale_by_std": p.get("scale_by_std", False),
        "scale_factor": p.get("scale_factor", 1.0),
        "upstream_loss": p.get("loss_type", "l2"),
        "loss_type": p.get("loss_type", "l2")
        if cfg["protocol"] == "upstream"
        else "l2",
        "logvar_init": p.get("logvar_init", 0.0),
        "l_simple_weight": p.get("l_simple_weight", 1.0),
        "original_elbo_weight": p.get("original_elbo_weight", 0.0),
        "v_posterior": p.get("v_posterior", 0.0),
        "scheduler": copy.deepcopy(p.get("scheduler_config")),
        "paper_training": dict(
            zip(("batch_size", "lr", "iterations"), PAPER_TRAINING[cfg["model"]])
        ),
    }
    if spec["loss_type"] not in ("l1", "l2"):
        raise ValueError("Unsupported native loss")
    if (
        spec["unet"].get("use_spatial_transformer", False)
        or spec["unet"].get("num_classes") is not None
    ):
        raise ValueError("Conditioned U-Nets are unsupported")
    if (
        spec["scheduler"]
        and spec["scheduler"]["target"] != "ldm.lr_scheduler.LambdaLinearScheduler"
    ):
        raise ValueError("Unsupported learning rate schedule")
    s, n = cfg["sampling"], spec["timesteps"]
    if s["method"] == "ddpm" and s["steps"] != n:
        raise ValueError("DDPM sampling requires all trained timesteps")
    if s["method"] == "ddim" and (s["steps"] >= n or n % s["steps"]):
        raise ValueError(
            "Native DDIM steps must divide timesteps and be smaller (e.g. 50, 100, 200, 250, 500)"
        )
    return spec


def lr_multiplier(spec, step):
    if not spec["scheduler"]:
        return 1.0
    p = spec["scheduler"]["params"]
    for warm, length, start, maximum, minimum in zip(
        p["warm_up_steps"], p["cycle_lengths"], p["f_start"], p["f_max"], p["f_min"]
    ):
        if step <= length:
            return (
                (maximum - start) * step / warm + start
                if step < warm
                else minimum + (maximum - minimum) * (length - step) / length
            )
        step -= length
    raise ValueError("Training exceeds the configured LR schedule")
