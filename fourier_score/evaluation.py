"""Checkpoint DSM evaluation and FID/IS of image folders."""

import torch
from fourier_score.model.metric import evaluate_dsm


def evaluate_checkpoint(checkpoint, output, overrides=(), device=None):
    """Evaluate the checkpoint's EMA with its original data and statistics."""
    from fourier_score.data_loader.data_loaders import build_data
    from fourier_score.data_loader.statistics import prepare_stats
    from fourier_score.trainer.checkpoints import load_inference
    from fourier_score.utils import environment, json_write

    model, cfg, device, state = load_inference(checkpoint, overrides, device)
    bundle = build_data(cfg)
    if state["stats"] is not None:
        prepare_stats(cfg, bundle, state["stats"])
    result = evaluate_dsm(model, cfg, bundle.validation, device)
    result.update(
        step=state["step"],
        split=bundle.metadata["eval_split"],
        weights="EMA",
        training_wall_seconds=state.get("training_wall_seconds"),
        environment=environment(device, cfg),
    )
    if "pretrained_source" in state:
        result["pretrained_source"] = state["pretrained_source"]
    json_write(result, output)
    return result


def folder_manifest(path):
    """Record which image files were evaluated, including their content hash."""
    import hashlib
    from pathlib import Path

    path = Path(path)
    files = sorted(
        p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if len(files) < 2:
        raise ValueError(f"At least two images required: {path}")
    digest = hashlib.sha256()
    for image in files:
        digest.update(image.name.encode())
        digest.update(image.read_bytes())
    return {
        "path": str(path.resolve()),
        "count": len(files),
        "sha256": digest.hexdigest(),
    }


def evaluate_images(real, generated, output, device="cpu", batch_size=64, *, inception_score=True):
    """Compute FID/IS; the optional Inception dependency is loaded only here."""
    import importlib.metadata
    from fourier_score.utils import json_write, resolve_device

    device = resolve_device(device)
    if device.type == "mps":
        raise ValueError("Use --device cpu or cuda for torch-fidelity")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    real_manifest = folder_manifest(real)
    generated_manifest = folder_manifest(generated)
    try:
        import torch_fidelity
    except ImportError as error:
        raise RuntimeError(
            "Install optional metrics: uv sync --locked --extra metrics"
        ) from error
    values = torch_fidelity.calculate_metrics(
        input1=generated_manifest["path"],
        input2=real_manifest["path"],
        cuda=device.type == "cuda",
        fid=True,
        isc=inception_score,
        kid=False,
        prc=False,
        batch_size=batch_size,
        verbose=True,
    )
    result = {
        "metrics": {key: float(value) for key, value in values.items()},
        "implementation": "torch-fidelity",
        "version": importlib.metadata.version("torch-fidelity"),
        "real": real_manifest,
        "generated": generated_manifest,
        "device": str(device),
        "batch_size": batch_size,
    }
    json_write(result, output)
    return result
