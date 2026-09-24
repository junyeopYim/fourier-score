"""Load EMA from either last.pt or an EMA-only snapshot."""

import copy
import warnings
import torch
from fourier_score.config import validate, apply_overrides
from fourier_score.model import build_model
from fourier_score.utils import load_checkpoint, configure_runtime, source_hash


FORMAT = "fourier-image-template-v1"


def resume_signature(cfg):
    c = copy.deepcopy(cfg)
    # Missing objective in v1 checkpoints means DSM; retain their signatures.
    if c["loss"].get("objective", "dsm") == "dsm":
        c["loss"].pop("objective", None)
    # Adding disabled-gate defaults must preserve pre-gate v1 signatures.
    if c["fourier"].get("gate", {}).get("mode", "none") == "none":
        c["fourier"].pop("gate", None)
    else:
        gate = c["fourier"]["gate"]
        if gate["mode"] not in ("log_sigma_plateau", "spectral_cap", "linear_log_sigma", "bounded_log_sigmoid"):
            gate.pop("sigma_lo", None)
            gate.pop("sigma_hi", None)
        if gate["mode"] == "linear_log_sigma":
            gate.pop("sigma_switch", None)
            gate.pop("sharpness", None)
        # Optional defaults must not change older active-gate signatures.
        if gate["mode"] != "spectral_cap":
            gate.pop("delta", None)
    for k in ("name", "device", "evaluation", "sampling"):
        c.pop(k)
    for k in (
        "iterations",
        "save_dir",
        "save_every",
        "snapshot_every",
        "log_every",
        "eval_every",
        "tensorboard",
    ):
        c["trainer"].pop(k)
    for k in ("console", "progress_every_seconds"):
        c["trainer"].pop(k, None)
    for k in ("root", "download", "num_workers"):
        c["data_loader"]["args"].pop(k)
    c["fourier"].pop("cache_dir")
    c["fourier"].pop("stats_batch_size")
    return c


def load_inference(path, overrides=(), device=None):
    ckpt = load_checkpoint(path)
    if ckpt.get("format") != FORMAT:
        raise ValueError("Not a fourier-image-template v1 checkpoint")
    changes = list(overrides)
    if device is not None:
        changes.append("device=" + device)
    for change in changes:
        key = change.split("=", 1)[0]
        if not (
            key == "device"
            or key.startswith(("sampling.", "evaluation.", "backend."))
            or key
            in (
                "data_loader.args.root",
                "data_loader.args.download",
                "data_loader.args.num_workers",
            )
        ):
            raise ValueError(
                f"Inference cannot alter the trained model/process/loss/statistics: {key}"
            )
    # Supply new optional evaluation defaults before applying overrides to old snapshots.
    cfg = validate(apply_overrides(validate(copy.deepcopy(ckpt["config"])), changes))
    dev = configure_runtime(cfg)
    model = build_model(cfg, ckpt["stats"], dev)
    model.load_state_dict(ckpt["model"], strict=True)
    if ckpt["kind"] == "training":
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n, t in ckpt["ema"]["shadow"].items():
                params[n].copy_(t.to(params[n]))
    elif ckpt["kind"] != "ema":
        raise ValueError("Unknown checkpoint kind")
    if ckpt["source_sha256"] != source_hash():
        warnings.warn(
            "Inference source differs from checkpoint; results are a new evaluation protocol",
            stacklevel=2,
        )
    model.eval()
    return model, cfg, dev, ckpt
