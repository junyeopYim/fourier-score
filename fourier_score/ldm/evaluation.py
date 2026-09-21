"""Latent diagnostics and decoded RGB artifacts using the shared epsilon path."""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fourier_score.evaluation import frequency_band_sums, radial_frequency_bands
from fourier_score.images import preview_grid, write_png
from fourier_score.utils import environment, isolated_rng, json_write

from .data import LatentDataset, draw_latents
from .model import sample_latents
from .training import synchronize


@torch.no_grad()
def evaluate_latents(model, cfg, cache, device):
    opt = cfg["evaluation"]
    ds = LatentDataset(cfg["cache"]["dir"], "validation", cache)
    n = min(len(ds), opt["max_images"])
    if n < 1:
        raise ValueError("Empty validation data")
    total, square, native_total = 0.0, 0.0, 0.0
    bins, fbins = opt["noise_bins"], opt["frequency_bins"]
    sums = torch.zeros(bins, dtype=torch.float64)
    counts = torch.zeros(bins, dtype=torch.long)
    spectral = torch.zeros(bins, fbins, dtype=torch.float64)
    if fbins:
        band_ids, modes, edges = radial_frequency_bands(
            *model.spec["latent_shape"][1:], fbins
        )
    old = model.training
    try:
        model.eval()
        with isolated_rng(device):
            generator = torch.Generator().manual_seed(opt["seed"])
            loader = DataLoader(
                Subset(ds, range(n)),
                batch_size=opt["batch_size"],
                shuffle=False,
                generator=torch.Generator().manual_seed(opt["seed"] + 1),
            )
            for batch in loader:
                clean = draw_latents(batch, cache["first_stage"], generator).to(device)
                t = torch.randint(
                    model.spec["timesteps"], (len(clean),), generator=generator
                ).to(device)
                noise = torch.randn(clean.shape, generator=generator).to(device)
                residual = model(model.schedule.q_sample(clean, t, noise), t) - noise
                values = residual.square().flatten(1).mean(1).cpu().double()
                native = (
                    residual.abs()
                    if model.spec["loss_type"] == "l1"
                    else residual.square()
                )
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite latent evaluation loss")
                native_total += native.flatten(1).mean(1).sum().item()
                which = (t.cpu() * bins // model.spec["timesteps"]).clamp_max(bins - 1)
                sums.scatter_add_(0, which, values)
                counts.scatter_add_(0, which, torch.ones_like(which))
                if fbins:
                    spectral.index_add_(
                        0, which, frequency_band_sums(residual, band_ids, fbins)
                    )
                total += values.sum().item()
                square += values.square().sum().item()
    finally:
        model.train(old)
    mean = total / n
    result = {
        "epsilon_mse": mean,
        "standard_error": (max(0, (square - n * mean**2) / max(n - 1, 1)) / n) ** 0.5,
        "native_simple_loss": native_total / n,
        "native_loss_type": model.spec["loss_type"],
        "n_images": n,
        "split": "validation",
        "seed": opt["seed"],
        "noise_bins": [
            {
                "count": int(counts[i]),
                "mse": float(sums[i] / counts[i]) if counts[i] else None,
            }
            for i in range(bins)
        ],
    }
    if fbins:
        result["frequency"] = {
            "units": "radial cycles/latent-pixel",
            "edges": edges,
            "modes": modes.tolist(),
            "mse": [
                float(spectral[:, j].sum() / (n * modes[j])) if modes[j] else None
                for j in range(fbins)
            ],
            "noise_frequency_mse": [
                [
                    float(spectral[i, j] / (counts[i] * modes[j]))
                    if counts[i] and modes[j]
                    else None
                    for j in range(fbins)
                ]
                for i in range(bins)
            ],
        }
    return result


def rgb_uint8(images):
    if not torch.isfinite(images).all():
        raise FloatingPointError("Nonfinite decoded image")
    return (
        (((images + 1) / 2).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    )


def empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.no_grad()
def generate(model, stage, cfg, output, provenance):
    out = empty_output(output)
    (out / "png").mkdir()
    device = next(model.parameters()).device
    options = cfg["sampling"]
    metadata = {
        "complete": False,
        "config": cfg,
        "spec": model.spec,
        "environment": environment(device, cfg),
        "provenance": provenance,
        "quantization": "clamp decoded RGB to [0,1], multiply by 255, truncate to uint8",
        "sampler": options,
        "latent_clipping": False,
        "first_stage_scale": stage.scale_factor,
    }
    json_write(metadata, out / "settings.json")
    generator = torch.Generator().manual_seed(options["seed"])
    model.eval()
    preview, done, calls, shard = [], 0, 0, 0
    synchronize(device)
    started = time.perf_counter()
    with isolated_rng(device):
        while done < options["num_samples"]:
            size = min(options["batch_size"], options["num_samples"] - done)
            latent, nfe = sample_latents(model, size, options, device, generator)
            decoded = [
                rgb_uint8(stage.decode(block))
                for block in latent.split(options["decode_batch_size"])
            ]
            arrays = np.concatenate(decoded)
            for j, arr in enumerate(arrays):
                write_png(arr, out / "png" / f"{done + j:07d}.png")
            np.savez_compressed(out / f"samples_{shard:05d}.npz", samples=arrays)
            if len(preview) < 64:
                preview.extend(arrays[: 64 - len(preview)])
            done, calls, shard = done + len(arrays), calls + nfe, shard + 1
            print(
                f"[sample] {done}/{options['num_samples']} | {nfe} NFE/sample",
                flush=True,
            )
    synchronize(device)
    preview_grid(np.stack(preview), out / "preview.png")
    metadata.update(
        complete=True,
        num_samples=done,
        batched_denoiser_calls=calls,
        nfe_per_sample=options["steps"],
        wall_seconds=time.perf_counter() - started,
    )
    json_write(metadata, out / "settings.json")
    return metadata


@torch.no_grad()
def reconstruct(stage, dataset, cfg, output, provenance):
    out = empty_output(output)
    for name in ("real", "reconstruction"):
        (out / name).mkdir()
    limit = min(len(dataset), cfg["evaluation"]["max_images"])
    generator = torch.Generator().manual_seed(cfg["evaluation"]["seed"])
    loader = DataLoader(
        Subset(dataset, range(limit)),
        batch_size=cfg["sampling"]["decode_batch_size"],
        generator=torch.Generator().manual_seed(0),
    )
    device = next(stage.parameters()).device
    psnr, done, preview = [], 0, []
    json_write({"complete": False, "provenance": provenance}, out / "settings.json")
    for images in loader:
        images = images.to(device)
        decoded = stage.decode(stage.encode(images, generator))
        mse = ((decoded.clamp(-1, 1) - images) / 2).square().flatten(1).mean(1)
        psnr.extend((-10 * mse.clamp_min(1e-12).log10()).cpu().tolist())
        real = (
            ((images + 1) * 127.5)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        rec = rgb_uint8(decoded)
        for a, b in zip(real, rec):
            write_png(a, out / "real" / f"{done:07d}.png")
            write_png(b, out / "reconstruction" / f"{done:07d}.png")
            if len(preview) < 64:
                preview.extend((a, b))
            done += 1
    preview_grid(np.stack(preview), out / "preview.png")
    result = {
        "complete": True,
        "n_images": done,
        "psnr_db_mean": float(np.mean(psnr)),
        "posterior": "sample"
        if stage.kind == "AutoencoderKL"
        else "continuous_prequantization",
        "provenance": provenance,
        "config": cfg,
    }
    json_write(result, out / "settings.json")
    return result
