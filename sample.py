"""Generate EMA PNGs + sharded uint8 NHWC NPZ + protocol metadata."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from fourier_score.trainer.checkpoints import load_inference
from fourier_score.images import write_png, preview_grid
from fourier_score.parse_config import CustomArgs, add_options, cli_overrides
from fourier_score.provenance import file_sha256
from fourier_score.utils import json_write, environment
from fourier_score.model.sampling import sample_batch

SAMPLING = [
    CustomArgs(["--num-samples"], "sampling.num_samples", type=int),
    CustomArgs(["--batch-size"], "sampling.batch_size", type=int),
    CustomArgs(["--steps"], "sampling.steps", type=int),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device")
    args = add_options(parser, SAMPLING).parse_args()
    changes = cli_overrides(args, SAMPLING)
    model, cfg, device, ckpt = load_inference(args.resume, changes, args.device)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be empty: {out}")
    (out / "png").mkdir(parents=True, exist_ok=True)
    settings = {
        "complete": False,
        "checkpoint_sha256": file_sha256(args.resume),
        "checkpoint_step": ckpt["step"],
        "training_wall_seconds": ckpt.get("training_wall_seconds"),
        "weights": "EMA",
        "config": cfg,
        "environment": environment(device, cfg),
        "npz_format": "uint8 NHWC, key=samples",
        "png_dir": "png",
    }
    if "pretrained_source" in ckpt:
        settings["pretrained_source"] = ckpt["pretrained_source"]
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
