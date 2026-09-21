"""Common DSM and optional noise-by-frequency diagnostics, no FID claims."""

import math
import torch
from torch.utils.data import DataLoader, Subset
from fourier_score.loss import noise_residual
from fourier_score.utils import isolated_rng


def evaluate_checkpoint(checkpoint, output, overrides=(), device=None):
    """Evaluate the checkpoint's EMA with its original data and statistics."""
    from fourier_score.checkpoints import load_inference
    from fourier_score.data import build_data, prepare_stats
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
        "warning": "Not directly interchangeable with original score-SDE TF-Hub/TF-GAN FID",
    }
    json_write(result, output)
    return result


def radial_frequency_bands(height, width, bins):
    """Full FFT coordinates, equal radial bands in cycles/pixel, DC included."""
    if bins < 1:
        raise ValueError("At least one frequency band is required")
    fy = torch.fft.fftfreq(height, dtype=torch.float64)
    fx = torch.fft.fftfreq(width, dtype=torch.float64)
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    maximum = math.sqrt(0.5)
    ids = (radius / maximum * bins).long().clamp_max(bins - 1).flatten()
    return (
        ids,
        torch.bincount(ids, minlength=bins),
        torch.linspace(0, maximum, bins + 1, dtype=torch.float64).tolist(),
    )


def frequency_band_sums(residual, band_ids, bins):
    """Per-image spectral energy, summed over modes and averaged over channels.

    Diagnostic-only CPU FFT also works when training/evaluating on MPS.
    Divide by each band's mode count to obtain its mean squared residual.
    """
    spectrum = torch.fft.fft2(
        residual.detach().to(device="cpu", dtype=torch.float64), norm="ortho"
    )
    energy = spectrum.abs().square().mean(1).flatten(1)
    result = torch.zeros(len(residual), bins, dtype=torch.float64)
    return result.scatter_add_(1, band_ids[None].expand(len(residual), -1), energy)


@torch.no_grad()
def evaluate_dsm(model, cfg, dataset, device, progress=None):
    opt = cfg["evaluation"]
    n = min(len(dataset), opt["max_images"])
    if n < 1:
        raise ValueError("Empty evaluation set")
    bins = opt["noise_bins"]
    sums = torch.zeros(bins, dtype=torch.float64)
    counts = torch.zeros(bins, dtype=torch.long)
    frequency_bins = opt.get("frequency_bins", 0)
    spectral_sums = torch.zeros(bins, frequency_bins, dtype=torch.float64)
    band_ids = mode_counts = band_edges = None
    total = 0.0
    sq = 0.0
    seen = 0
    old = model.training
    try:
        model.eval()
        with isolated_rng(device):
            gen = torch.Generator().manual_seed(opt["seed"])
            loader = DataLoader(
                Subset(dataset, range(n)),
                batch_size=opt["batch_size"],
                shuffle=False,
                num_workers=0,
                generator=torch.Generator().manual_seed(opt["seed"] + 1),
            )
            for clean in loader:
                lev = model.process.sample(len(clean), device, gen)
                noise = torch.randn(clean.shape, generator=gen).to(device)
                residual = noise_residual(model, clean.to(device), lev, noise)
                values = residual.square().flatten(1).mean(1).cpu().double()
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite evaluation loss")
                which = (
                    (model.process.bin_coordinate(lev).cpu() * bins)
                    .long()
                    .clamp(0, bins - 1)
                )
                sums.scatter_add_(0, which, values)
                counts.scatter_add_(0, which, torch.ones_like(which))
                if frequency_bins:
                    if band_ids is None:
                        band_ids, mode_counts, band_edges = radial_frequency_bands(
                            *residual.shape[-2:], frequency_bins
                        )
                    spectral_sums.index_add_(
                        0,
                        which,
                        frequency_band_sums(residual, band_ids, frequency_bins),
                    )
                total += values.sum().item()
                sq += values.square().sum().item()
                seen += len(values)
                if progress is not None:
                    progress(seen, n)
    finally:
        model.train(old)
    mean = total / seen
    var = max(0.0, (sq - seen * mean * mean) / max(1, seen - 1))
    result = {
        "dsm_pixel_mean": mean,
        "standard_error": (var / seen) ** 0.5,
        "n_images": seen,
        "eval_seed": opt["seed"],
        "eval_batch_size": opt["batch_size"],
        "bin_axis": "linear diffusion step"
        if model.process.kind == "ddpm"
        else "linear t = logarithmic sigma",
        "dsm_by_noise": [
            {
                "bin": i,
                "count": int(counts[i]),
                "mean": float(sums[i] / counts[i]) if counts[i] else None,
            }
            for i in range(bins)
        ],
    }
    if frequency_bins:
        result["frequency_bands"] = {
            "edges": band_edges,
            "units": "radial cycles/pixel",
            "modes_per_channel": mode_counts.tolist(),
            "transform": "full orthonormal FFT; includes DC and both conjugate partners",
        }
        result["dsm_by_frequency"] = [
            float(spectral_sums[:, j].sum() / (seen * mode_counts[j]))
            if mode_counts[j]
            else None
            for j in range(frequency_bins)
        ]
        result["dsm_by_noise_frequency"] = [
            {
                "noise_bin": i,
                "count": int(counts[i]),
                "mean": [
                    float(spectral_sums[i, j] / (counts[i] * mode_counts[j]))
                    if counts[i] and mode_counts[j]
                    else None
                    for j in range(frequency_bins)
                ],
            }
            for i in range(bins)
        ]
    return result
