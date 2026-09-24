"""Device, reproducibility and safe local artifact utilities."""

from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
import numpy as np
import torch
from fourier_score.provenance import ROOT, source_hash


def resolve_device(name="auto"):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS requested but unavailable. Run scripts/doctor.py on an MPS-capable Mac."
        )
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError("CUDA device index out of range")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def configure_runtime(cfg):
    device = resolve_device(cfg["device"])
    torch.set_num_threads(cfg["backend"]["cpu_threads"])
    torch.set_default_dtype(torch.float32)
    tf32 = cfg["backend"]["tf32"]
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32" if tf32 else "ieee"
        torch.backends.cudnn.conv.fp32_precision = "tf32" if tf32 else "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    return device


def seed_all(seed, device=None):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device is not None and device.type == "mps":
        torch.mps.manual_seed(seed)


def capture_rng(device=None):
    n = np.random.get_state()
    out = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": n[0],
            "keys": n[1].tolist(),
            "position": int(n[2]),
            "has_gauss": int(n[3]),
            "cached": float(n[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    if device is not None and device.type == "mps":
        out["mps"] = torch.mps.get_rng_state().cpu()
    return out


def restore_rng(state, device=None):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state(
        (
            n["algorithm"],
            np.asarray(n["keys"], dtype=np.uint32),
            n["position"],
            n["has_gauss"],
            n["cached"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    if "mps" in state and device is not None and device.type == "mps":
        torch.mps.set_rng_state(state["mps"].cpu())


@contextlib.contextmanager
def isolated_rng(device=None):
    state = capture_rng(device)
    try:
        yield
    finally:
        restore_rng(state, device)


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path):
    # All our saved objects are tensors or plain Python primitives.
    return torch.load(path, map_location="cpu", weights_only=True)


def json_write(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def environment(device, cfg):
    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "tf32": cfg["backend"]["tf32"],
        "precision": "fp32",
        "amp": False,
        "spectral_transform_config": cfg["backend"]["spectral_transform"],
        "mps_fallback_environment": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK", "unset"
        ),
        "source_sha256": source_hash(),
    }
    if device.type == "cuda":
        out["device_name"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        out["device_name"] = "Apple MPS"
    else:
        out["device_name"] = platform.processor() or "CPU"
    return out
