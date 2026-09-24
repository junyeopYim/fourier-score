"""Explicit upstream splits, native preprocessing and mmap posterior caches."""

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fourier_score.data_loader.statistics import estimate_stats
from fourier_score.provenance import digest_json, file_sha256
from fourier_score.utils import atomic_save, json_write, load_checkpoint

from .first_stage import load_first_stage, sample_posterior

CACHE_FORMAT = "fourier-ldm-cache-v1"


class ImageList(Dataset):
    def __init__(self, root, listing, size, preprocessing):
        self.root = Path(root).resolve()
        self.listing = Path(listing)
        self.size, self.preprocessing = size, preprocessing
        self.names = self.listing.read_text().splitlines()
        if not self.names or any(not s.strip() for s in self.names):
            raise ValueError(f"Empty path/list: {listing}")
        self.paths = [(self.root / name).resolve() for name in self.names]
        if any(not p.is_relative_to(self.root) for p in self.paths):
            raise ValueError("Image lists must contain paths inside data.root")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError(f"Duplicate images in {listing}")
        h = hashlib.sha256()
        for name, path in zip(self.names, self.paths):
            st = path.stat()
            h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        self.fingerprint = h.hexdigest()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 3 or arr.shape[0] != 3 or arr.dtype != np.uint8:
                raise ValueError("CelebA-HQ arrays must be uint8 [1,3,H,W] or [3,H,W]")
            arr = arr.transpose(1, 2, 0)
        else:
            with Image.open(path) as image:
                arr = np.asarray(image.convert("RGB")).copy()
        h, w = arr.shape[:2]
        if self.preprocessing == "lsun_bicubic":
            side = min(h, w)
            arr = arr[
                (h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2
            ]
            arr = np.asarray(
                Image.fromarray(arr).resize(
                    (self.size, self.size), Image.Resampling.BICUBIC
                )
            )
        elif self.preprocessing == "faces_cv2":
            import cv2

            scale = self.size / min(h, w)
            if scale != 1:
                arr = cv2.resize(
                    arr,
                    (round(w * scale), round(h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            h, w = arr.shape[:2]
            arr = arr[
                (h - self.size) // 2 : (h + self.size) // 2,
                (w - self.size) // 2 : (w + self.size) // 2,
            ]
        else:
            raise ValueError("Unsupported image preprocessing")
        # Match the original uint8 -> float64 division -> float32 convention.
        return (
            torch.from_numpy((arr / 127.5 - 1.0).astype(np.float32))
            .permute(2, 0, 1)
            .contiguous()
        )


def image_splits(cfg, spec):
    a = cfg["data"]
    splits = {
        key: ImageList(
            a["root"], a[f"{key}_list"], spec["image_size"], a["preprocessing"]
        )
        for key in ("train", "validation")
    }
    if set(splits["train"].paths) & set(splits["validation"].paths):
        raise ValueError("Train and validation image lists overlap")
    if len(splits["train"]) < 2:
        raise ValueError("At least two training images are required")
    return splits


def representation(spec):
    return {
        k: spec[k]
        for k in (
            "first_stage",
            "image_size",
            "latent_shape",
            "scale_factor",
            "scale_by_std",
        )
    }


def settings(cfg):
    return {"data": cfg["data"], "power_floor": cfg["cache"]["power_floor"]}


def encoding_identity(first_stage):
    return {key: first_stage[key] for key in ("kind", "scale_factor", "posterior")}


def prepare_cache(cfg, spec, device, progress=print):
    started = time.perf_counter()
    splits = image_splits(cfg, spec)
    source = {
        key: {
            "list_sha256": file_sha256(cfg["data"][f"{key}_list"]),
            "file_metadata_sha256": ds.fingerprint,
            "count": len(ds),
        }
        for key, ds in splits.items()
    }
    identity_inputs = {
        "format": CACHE_FORMAT,
        "representation": representation(spec),
        "settings": settings(cfg),
        "sources": source,
        "first_stage_sha256": file_sha256(cfg["first_stage"]["checkpoint"]),
    }
    target = Path(cfg["cache"]["dir"])
    if target.exists():
        cache = open_cache(cfg, spec)
        identity_inputs["encoding"] = encoding_identity(cache["first_stage"])
        identity = digest_json(identity_inputs)
        if cache["identity"] != identity:
            raise ValueError(
                "Existing latent cache differs from data/encoder; choose a new cache.dir"
            )
        return cache
    stage, first = load_first_stage(cfg["first_stage"]["checkpoint"], spec, device)
    identity_inputs["encoding"] = encoding_identity(first)
    identity = digest_json(identity_inputs)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}-", dir=target.parent
    ) as temporary:
        work = Path(temporary)
        arrays = {}
        for split, ds in splits.items():
            views = 2 if split == "train" and cfg["data"]["random_flip"] else 1
            channels = spec["latent_shape"][0] * (
                2 if stage.kind == "AutoencoderKL" else 1
            )
            shape = (len(ds), views, channels, *spec["latent_shape"][1:])
            arr = np.lib.format.open_memmap(
                work / f"{split}.npy", mode="w+", dtype=np.float32, shape=shape
            )
            loader = DataLoader(
                ds,
                batch_size=cfg["cache"]["batch_size"],
                shuffle=False,
                num_workers=cfg["cache"]["num_workers"],
                multiprocessing_context="spawn"
                if cfg["cache"]["num_workers"]
                else None,
                generator=torch.Generator().manual_seed(0),
            )
            offset, last = 0, 0.0
            for images in loader:
                images = images.to(device)
                encoded = [stage.encode_parameters(images)]
                if views == 2:
                    # The encoder is not assumed to commute with horizontal flip.
                    encoded.append(stage.encode_parameters(images.flip(-1)))
                value = torch.stack(encoded, dim=1).cpu().numpy()
                if value.shape[1:] != shape[1:] or not np.isfinite(value).all():
                    raise ValueError("Nonfinite/incorrectly shaped first-stage output")
                arr[offset : offset + len(value)] = value
                offset += len(value)
                if time.perf_counter() - last > 5 or offset == len(ds):
                    progress(
                        f"[cache] {split}: {offset}/{len(ds)} images ({views} encoded views)"
                    )
                    last = time.perf_counter()
            arr.flush()
            del arr
            arrays[split] = {"file": f"{split}.npy", "shape": list(shape)}
        # Free the frozen model before CPU statistics or denoiser construction.
        del stage
        params = np.load(work / "train.npy", mmap_mode="r", allow_pickle=False)
        kl = first["kind"] == "AutoencoderKL"

        def batches(parameters):
            for start in range(0, len(parameters), cfg["cache"]["batch_size"]):
                p = (
                    torch.from_numpy(
                        parameters[start : start + cfg["cache"]["batch_size"]].copy()
                    )
                    .flatten(0, 1)
                    .double()
                )
                if kl:
                    mean, logvar = p.chunk(2, dim=1)
                    yield (
                        first["scale_factor"] * mean,
                        first["scale_factor"] ** 2 * logvar.exp(),
                    )
                else:
                    yield first["scale_factor"] * p

        stats = estimate_stats(
            batches(params), cfg["cache"]["power_floor"], posterior_variance=kl
        )
        stats.update(
            identity=identity,
            source_split="train",
            posterior_moments="analytic" if kl else "empirical",
        )
        atomic_save(stats, work / "stats.pt")
        del params
        files = {
            name: {
                "sha256": file_sha256(work / name),
                "bytes": (work / name).stat().st_size,
            }
            for name in ("train.npy", "validation.npy", "stats.pt")
        }
        manifest = {
            **identity_inputs,
            "identity": identity,
            "complete": True,
            "first_stage": first,
            "arrays": arrays,
            "files": files,
            "preparation_wall_seconds": time.perf_counter() - started,
            "storage": "float32 unscaled posterior parameters; no fixed KL draw",
        }
        json_write(manifest, work / "manifest.json")
        os.rename(work, target)
    return manifest


def open_cache(cfg, spec):
    root = Path(cfg["cache"]["dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != CACHE_FORMAT or not manifest.get("complete"):
        raise ValueError("Incomplete or unsupported latent cache")
    if manifest["representation"] != representation(spec) or manifest[
        "settings"
    ] != settings(cfg):
        raise ValueError(
            "Latent cache model/preprocessing/augmentation/statistics mismatch"
        )
    identity_keys = (
        "format",
        "representation",
        "settings",
        "sources",
        "first_stage_sha256",
        "encoding",
    )
    if manifest["identity"] != digest_json({k: manifest[k] for k in identity_keys}):
        raise ValueError("Invalid cache identity")
    first = manifest["first_stage"]
    if (
        manifest["encoding"] != encoding_identity(first)
        or first["kind"] != spec["first_stage"]["kind"]
        or not math.isfinite(first["scale_factor"])
        or first["scale_factor"] <= 0
        or first["checkpoint_sha256"] != manifest["first_stage_sha256"]
    ):
        raise ValueError("Latent cache encoding/scale mismatch")
    if file_sha256(cfg["first_stage"]["checkpoint"]) != manifest["first_stage_sha256"]:
        raise ValueError("Latent cache frozen checkpoint mismatch")
    for name in ("train.npy", "validation.npy", "stats.pt"):
        expected = manifest["files"][name]
        if (root / name).stat().st_size != expected["bytes"] or file_sha256(
            root / name
        ) != expected["sha256"]:
            raise ValueError(f"Corrupt latent cache file: {name}")
    stats = load_checkpoint(root / "stats.pt")
    if stats["identity"] != manifest["identity"]:
        raise ValueError("Statistics do not match latent cache")
    return manifest


class LatentDataset(Dataset):
    def __init__(self, root, split, manifest):
        self.path = str(Path(root) / f"{split}.npy")
        self.shape = manifest["arrays"][split]["shape"]
        self.array = None

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
            if list(self.array.shape) != self.shape or self.array.dtype != np.float32:
                raise ValueError("Latent array shape/dtype mismatch")
        return torch.from_numpy(self.array[index].copy())

    def __getstate__(self):
        return {**self.__dict__, "array": None}


def draw_latents(batch, first_stage, generator):
    views = torch.randint(batch.shape[1], (len(batch),), generator=generator)
    params = batch[torch.arange(len(batch)), views]
    return sample_posterior(
        params, first_stage["kind"], first_stage["scale_factor"], generator
    )
