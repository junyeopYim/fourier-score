"""Generate EMA PNGs + sharded uint8 NHWC NPZ + protocol metadata."""

import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from fourier_score.checkpoints import load_inference
from fourier_score.images import write_png, preview_grid
from fourier_score.utils import json_write, environment
from fourier_score.diffusion import sample_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    for name in ("num_samples", "batch_size", "steps"):
        value = getattr(args, name)
        if value is not None:
            args.set.append(f"sampling.{name}={value}")
    model, cfg, device, ckpt = load_inference(args.resume, args.set, args.device)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be empty: {out}")
    (out / "png").mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(args.resume, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    settings = {
        "complete": False,
        "checkpoint_sha256": h.hexdigest(),
        "checkpoint_step": ckpt["step"],
        "training_wall_seconds": ckpt.get("training_wall_seconds"),
        "weights": "EMA",
        "config": cfg,
        "environment": environment(device, cfg),
        "npz_format": "uint8 NHWC, key=samples",
        "png_dir": "png",
    }
    json_write(settings, out / "settings.json")
    g = torch.Generator().manual_seed(cfg["sampling"]["seed"])
    n = cfg["sampling"]["num_samples"]
    batch = cfg["sampling"]["batch_size"]
    done = 0
    shard = 0
    nfe = 0
    preview = []
    clipped = 0
    elements = 0
    start = time.perf_counter()
    while done < n:
        # Always use the FULL effective batch, even in the final round: PC's
        # batch-mean Langevin norm must not silently change with the remainder.
        x, calls = sample_batch(model, cfg, batch, device, g)
        keep = min(batch, n - done)
        x = x[:keep]
        unit = (x + 1) / 2 if cfg["data_loader"]["args"]["centered"] else x
        clipped += int(((unit < 0) | (unit > 1)).sum())
        elements += unit.numel()
        arrays = (
            (unit.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        )
        np.savez_compressed(out / f"samples_{shard:05d}.npz", samples=arrays)
        for j, arr in enumerate(arrays):
            write_png(arr, out / "png" / f"{done + j:07d}.png")
        if sum(len(v) for v in preview) < 64:
            preview.append(arrays[: 64 - sum(len(v) for v in preview)])
        done += keep
        shard += 1
        nfe += calls
        print(f"{done}/{n} images; {nfe} batched score calls", flush=True)
    preview_grid(np.concatenate(preview), out / "preview.png")
    settings.update(
        complete=True,
        num_samples=done,
        batched_score_calls_total=nfe,
        terminal_out_of_range_fraction=clipped / elements,
        wall_seconds=time.perf_counter() - start,
    )
    json_write(settings, out / "settings.json")


if __name__ == "__main__":
    main()
