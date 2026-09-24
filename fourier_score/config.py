"""One strict JSON schema, inherited presets, and shared CLI overrides.

No implicit architecture / process changes when loss.type changes.
Paths in `extends` are relative to their declaring JSON file.
Data / output paths are relative to the current working directory.
"""

from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from string import Formatter
from fourier_score.method import GAUSSIAN_OBJECTIVES, OBJECTIVES, validate_gate, gate_suffix

ROOT = Path(__file__).resolve().parents[1]


def normalize_parameterization(obj: dict) -> dict:
    """Expose the mathematical name while retaining the v1 checkpoint schema."""
    obj = copy.deepcopy(obj)
    if "parameterization" in obj:
        value = obj.pop("parameterization")
        loss = obj.setdefault("loss", {})
        if not isinstance(loss, dict):
            raise ValueError("loss must be an object")
        if "type" in loss and loss["type"] != value:
            raise ValueError("parameterization and loss.type disagree")
        loss["type"] = value
    return obj


def deep_merge(base: dict, patch: dict, prefix: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in out:
            raise ValueError(f"Unknown configuration key: {path}")
        if isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            out[key] = deep_merge(out[key], value, path)
        else:
            out[key] = value
    return out


def _read(path: Path, stack: tuple = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Configuration must be a JSON object")
    obj = normalize_parameterization(obj)
    parent = obj.pop("extends", None)
    if parent is None:
        if path == (ROOT / "configs/base.json").resolve():
            return obj
        defaults = json.loads((ROOT / "configs/base.json").read_text())
    else:
        if not isinstance(parent, str):
            raise ValueError("extends must be one JSON filename")
        defaults = _read(path.parent / parent, stack + (path,))
    return deep_merge(defaults, obj)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"Expected key=value, got {entry!r}")
        path, raw = entry.split("=", 1)
        if path == "parameterization":
            path = "loss.type"
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parts = path.split(".")
        parent = cfg
        for key in parts[:-1]:
            if key not in parent or not isinstance(parent[key], dict):
                raise ValueError(f"Unknown configuration key: {path}")
            parent = parent[key]
        if parts[-1] not in parent:
            raise ValueError(f"Unknown configuration key: {path}")
        parent[parts[-1]] = value
    return cfg


def validate(cfg: dict) -> dict:
    defaults = json.loads((ROOT / "configs/base.json").read_text())
    cfg = deep_merge(defaults, normalize_parameterization(cfg))

    def shape(ref, obj, prefix=""):
        for k, v in ref.items():
            w = obj[k]
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                shape(v, w, p)
            elif v is not None:
                good = (
                    type(w) is type(v)
                    if not isinstance(v, float)
                    else type(w) in (int, float)
                )
                if not good:
                    raise ValueError(f"{p}: invalid type {type(w).__name__}")
            if isinstance(w, float) and not math.isfinite(w):
                raise ValueError(f"{p} must be finite")

    shape(defaults, cfg)

    def choice(path, options):
        x = cfg
        for p in path.split("."):
            x = x[p]
        if x not in options:
            raise ValueError(f"{path} must be one of {options}; got {x!r}")

    choice("loss.type", OBJECTIVES)
    choice("loss.objective", ("dsm", "normalized_residual"))
    choice("loss.reduction", ("mean", "half_sum"))
    choice("trainer.console", ("human", "json", "quiet"))
    choice("arch.type", ("NCSNpp",))
    choice("data_loader.type", ("ImageDataLoader",))
    choice(
        "data_loader.args.dataset", ("mnist", "cifar10", "image_folder", "synthetic")
    )
    choice("process.type", ("ve", "ddpm"))
    gate = validate_gate(cfg["fourier"]["gate"])
    if gate["mode"] != "none" and (
        cfg["process"]["type"] != "ve" or cfg["loss"]["type"] not in GAUSSIAN_OBJECTIVES
    ):
        raise ValueError("Gaussian gate requires VE with scalar_gaussian or fourier_gaussian")
    if cfg["loss"]["objective"] == "normalized_residual" and (
        cfg["process"]["type"] != "ve"
        or cfg["loss"]["type"] not in GAUSSIAN_OBJECTIVES
    ):
        raise ValueError(
            "loss.objective=normalized_residual requires VE with "
            "scalar_gaussian or fourier_gaussian"
        )
    choice("optimizer.type", ("Adam",))
    choice("backend.precision", ("fp32",))
    choice("backend.spectral_transform", ("auto", "fft", "matmul", "cpu"))
    a = cfg["arch"]["args"]
    d = cfg["data_loader"]["args"]
    p = cfg["process"]
    s = cfg["sampling"]
    t = cfg["trainer"]
    for key, vals in {
        "resblock_type": ("biggan", "ddpm"),
        "embedding_type": ("fourier", "positional"),
        "progressive": ("none", "output_skip", "residual"),
        "progressive_input": ("none", "input_skip", "residual"),
        "progressive_combine": ("cat", "sum"),
        "nonlinearity": ("swish", "elu", "relu", "lrelu"),
    }.items():
        choice("arch.args." + key, vals)
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    if not (
        cfg["device"] in ("auto", "cpu", "mps", "cuda")
        or cfg["device"].startswith("cuda:")
    ):
        raise ValueError("Invalid device")
    if cfg["seed"] < 0 or d["split_seed"] < 0:
        raise ValueError("Seeds must be nonnegative")
    if a["nf"] < 4 or a["nf"] % 4:
        raise ValueError("nf must be a positive multiple of 4")
    if not a["ch_mult"] or any(type(x) != int or x < 1 for x in a["ch_mult"]):
        raise ValueError("Invalid ch_mult")
    if a["num_res_blocks"] < 1 or not 0 <= a["dropout"] < 1 or a["init_scale"] < 0:
        raise ValueError("Invalid architecture values")
    if a["fourier_scale"] <= 0:
        raise ValueError("fourier_scale must be positive")
    if (
        not a["fir_kernel"]
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0
            for x in a["fir_kernel"]
        )
        or sum(a["fir_kernel"]) <= 0
    ):
        raise ValueError("Invalid fir_kernel")
    if d["image_size"] < 4 or d["image_size"] % 2 ** (len(a["ch_mult"]) - 1):
        raise ValueError("image_size incompatible with ch_mult depth")
    resolutions = {d["image_size"] // 2**i for i in range(len(a["ch_mult"]))}
    if any(type(x) != int or x not in resolutions for x in a["attn_resolutions"]):
        raise ValueError("attn_resolutions must be actual resolution levels")
    if d["channels"] not in (1, 3):
        raise ValueError("Only 1 or 3 image channels are supported")
    if d["dataset"] == "mnist" and (d["channels"] != 1 or d["image_size"] != 32):
        raise ValueError("MNIST uses MNIST32-pad, channels=1")
    if d["dataset"] == "cifar10" and (d["channels"] != 3 or d["image_size"] != 32):
        raise ValueError("CIFAR10 requires 3x32x32")
    if (
        d["batch_size"] < 1
        or d["num_workers"] < 0
        or d["validation_size"] < 0
        or d["synthetic_size"] < 4
    ):
        raise ValueError("Invalid data settings")
    if d["crop_size"] is not None and (
        type(d["crop_size"]) != int or d["crop_size"] < 1
    ):
        raise ValueError("crop_size must be null or positive integer")
    if (
        not 0 < p["sigma_min"] < p["sigma_max"]
        or p["num_scales"] < 2
        or not 0 < p["t_min"] < 1
    ):
        raise ValueError("Invalid noise schedule")
    if not 0 < p["beta_start"] <= p["beta_end"] < 1:
        raise ValueError("Invalid DDPM betas")
    if s["method"] not in (("pc", "heun") if p["type"] == "ve" else ("ddpm",)):
        raise ValueError("VE: pc/heun; DDPM: ddpm. Set sampling.method explicitly.")
    if p["type"] == "ddpm" and s["steps"] != p["num_scales"]:
        raise ValueError("DDPM ancestral sampling must use all trained steps")
    for key in ("steps", "batch_size", "num_samples"):
        if s[key] < 1:
            raise ValueError(f"sampling.{key} must be positive")
    if s["steps"] < 2 or s["corrector_steps"] < 0 or s["snr"] <= 0:
        raise ValueError("Invalid sampler")
    if s["clip_denoised"] and p["type"] != "ddpm":
        raise ValueError("clip_denoised is a DDPM-only option")
    for key in (
        "iterations",
        "save_every",
        "snapshot_every",
        "log_every",
        "eval_every",
    ):
        if t[key] < 1:
            raise ValueError(f"trainer.{key} must be positive")
    if t["progress_every_seconds"] <= 0:
        raise ValueError("trainer.progress_every_seconds must be positive")
    if t["warmup"] < 0 or not 0 < t["ema_decay"] < 1 or t["grad_clip"] <= 0:
        raise ValueError("Invalid optimizer lifecycle")
    if t["microbatch_size"] is not None and (
        type(t["microbatch_size"]) != int
        or not 1 <= t["microbatch_size"] <= d["batch_size"]
    ):
        raise ValueError("Invalid microbatch_size")
    o = cfg["optimizer"]["args"]
    if (
        o["lr"] <= 0
        or o["eps"] <= 0
        or o["weight_decay"] < 0
        or len(o["betas"]) != 2
        or any(type(x) not in (int, float) or not 0 <= x < 1 for x in o["betas"])
    ):
        raise ValueError("Invalid Adam settings")
    if (
        cfg["fourier"]["power_floor"] <= 0
        or cfg["fourier"]["stats_batch_size"] < 1
        or cfg["backend"]["cpu_threads"] < 1
    ):
        raise ValueError("Invalid Fourier/backend settings")
    if any(
        cfg["evaluation"][k] < 1 for k in ("batch_size", "max_images", "noise_bins")
    ):
        raise ValueError("Invalid evaluation settings")
    if cfg["evaluation"]["frequency_bins"] < 0:
        raise ValueError("evaluation.frequency_bins must be nonnegative")
    if cfg["name"] != "auto":
        fields = set()
        for _, field, spec, conversion in Formatter().parse(cfg["name"]):
            if field is not None:
                if field not in ("parameterization", "seed") or spec or conversion:
                    raise ValueError("name only supports {parameterization} and {seed}")
                fields.add(field)
        if fields and fields != {"parameterization", "seed"}:
            raise ValueError(
                "name templates must include both {parameterization} and {seed}"
            )
        name = experiment_name(cfg)
        if not name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in name
        ):
            raise ValueError("name must resolve to a simple experiment slug")
    return cfg


def load_config(path="config.json", overrides=()) -> dict:
    return validate(apply_overrides(_read(Path(path)), list(overrides)))


def experiment_name(cfg):
    if cfg["name"] != "auto":
        name = cfg["name"].format(
            parameterization=cfg["loss"]["type"], seed=cfg["seed"]
        )
    else:
        name = f"{cfg['data_loader']['args']['dataset']}_{cfg['process']['type']}_{cfg['loss']['type']}_s{cfg['seed']}"
    objective = cfg["loss"].get("objective", "dsm")
    name = name if objective == "dsm" else f"{name}_{objective}"
    return name + gate_suffix(cfg["fourier"].get("gate"))


def add_config_args(parser: argparse.ArgumentParser):
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda[:index]/mps")
    parser.add_argument(
        "--parameterization",
        choices=OBJECTIVES,
        help="Output parameterization; select loss.objective separately with --set",
    )
    return parser


def from_args(args):
    changes = list(args.set)
    if args.device is not None:
        changes.append("device=" + args.device)
    if args.parameterization is not None:
        changes.append("parameterization=" + args.parameterization)
    return load_config(args.config, changes)
