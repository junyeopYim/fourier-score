# Dataset preparation and focused tests — 2026-09-21

Working-tree verification based on `db69898`.
Package source SHA-256: `36c8907a0bdc7a2499e8dbf63a99dec1299c593539a38d0a9f3388feb1eb65dd`.

The default suite was reduced from 201 collected cases (199 passing, two MPS
skips) to **44 required cases** with all optional dependencies installed. Repeated
backbone combinations, config permutations, and terminal presentation assertions
were removed. Gaussian/filter math, paired initialization, training-only statistics,
exact resume, EMA, the checkpoint source regression, KL/VQ cache/train/decode,
native sampler behavior, and download integrity remain covered.

```bash
uv run --locked --extra ldm --extra metrics --extra datasets python -m pytest -q
uv lock --check --offline
```

Download verification:

* Reused and hash-verified the existing official Churches weight bundle through
  `scripts/download_ldm.py --model lsun_churches`.
* Downloaded the real official split lists into their configured `data/` paths;
  SHA-256 and counts matched for all four datasets:
  FFHQ 60,000/10,000, CelebA-HQ 25,000/5,000,
  Churches 121,227/5,000, Bedrooms 3,028,042/5,000.
* Downloaded NVIDIA's complete FFHQ metadata and one original image (`00000.png`)
  through the new downloader. Publisher MD5 verification passed; PIL read it as
  RGB, 1024×1024. This smoke artifact is ignored under
  `saved/ldm_data_download_smoke/`, not installed as a complete dataset.
* Local HTTP fixtures verified range resume, corruption rejection and offline
  reuse. A real tiny LMDB/ZIP verified byte-preserving LSUN export and custom
  CompVis partition membership. A tiny CelebA-HQ `.npy` archive verified correct
  names and that missing images never mark a dataset complete.

Full FFHQ/LSUN image collections were not downloaded. CelebA-HQ requires a
user-supplied original `.npy` directory/ZIP or its URL; no equivalent complete
archive is provided by the upstream reconstruction instructions. This change
has no long-training or image-quality result. See [the usage guide](../../docs/LDM.md).
