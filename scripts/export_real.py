"""Export the SAME deterministic preprocessing used in training for FID."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fourier_score.config import add_config_args, from_args
from fourier_score.data import build_data
from fourier_score.images import write_png
from fourier_score.utils import json_write


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--split", choices=["train", "validation"], default="train")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    cfg = from_args(a)
    bundle = build_data(cfg)
    ds = bundle.train if a.split == "train" else bundle.validation
    if a.limit is not None:
        if not 1 <= a.limit <= len(ds):
            raise ValueError("limit must be between 1 and dataset size")
        ds = Subset(ds, range(a.limit))
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output must be empty")
    (out / "png").mkdir(parents=True, exist_ok=True)
    done = 0
    for x in DataLoader(
        ds,
        batch_size=128,
        num_workers=0,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    ):
        unit = (x + 1) / 2 if cfg["data_loader"]["args"]["centered"] else x
        arrays = (unit.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
        for arr in arrays:
            write_png(arr, out / "png" / f"{done:07d}.png")
            done += 1
    json_write(
        {
            "config": cfg,
            "split": a.split,
            "effective_split": a.split
            if a.split == "train"
            else bundle.metadata["eval_split"],
            "count": done,
            "random_augmentation_applied": False,
            "dataset_fingerprint": bundle.metadata["dataset_fingerprint"],
        },
        out / "settings.json",
    )
    print(f"Exported {done} images to {out}/png")


if __name__ == "__main__":
    main()
