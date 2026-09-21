# Native LDM optimizer timing — 2026-09-21

This record measures real from-scratch denoiser updates on latents encoded by the
original frozen first stages. It is a throughput experiment on repeated small
image subsets, not a convergence or image-quality experiment.

## Results

**All 16 runs completed 100 optimizer updates each (1,600 total).** See the
[full timing table](RESULTS.md), [machine-readable measurements](measurements.csv),
and [audit result](audit.json). Per-update logs and resolved configurations are
preserved in [runs/](runs/). The audit verifies full effective batches, finite
losses/gradients, paired initialization/cache identities, and recomputes every
reported total, mean and ETA from the raw logs.

The existing [45-test suite passed](pytest.txt) after measurement (no new tests
were added). The benchmark CLI help, Ruff checks and local documentation links
also passed verification.

| Dataset | Effective batch | Paper updates | 100-update time across arms | Full-budget ETA across arms, one seed |
|---|---:|---:|---:|---:|
| FFHQ | 42 | 635,000 | 384.45–400.30 s | 28.09–29.30 days |
| CelebA-HQ | 48 | 410,000 | 428.69–447.35 s | 20.36–21.34 days |
| LSUN Churches, common L2 | 96 | 500,000 | 357.82–383.07 s | 20.67–22.29 days |
| LSUN Bedrooms | 48 | 1,900,000 | 442.96–453.92 s | 97.04–99.92 days |

Peak allocated CUDA memory was 6.60 GiB for Churches and about 7.11 GiB for the
VQ-f4 models. CUDA reserved memory, desktop use and total device usage differ
from this allocation counter. Each number describes the configuration below;
changing the microbatch or precision requires a new timing run.

## Protocol

* NVIDIA GeForce RTX 5060 Ti 16 GB; PyTorch FP32; TF32 and AMP disabled.
* Seed 42, native U-Net, AdamW and LitEma; microbatch 4 with accumulation to the
  paper effective batches: FFHQ 42, CelebA-HQ 48, Churches 96 and Bedrooms 48.
* Epsilon, scalar Gaussian, Fourier Gaussian unscaled, and Fourier Gaussian arms.
  Initial U-Net and latent-cache identities are checked across arms.
* 100 actual optimizer updates per arm. The first 10 are included in the measured
  100-update total, but excluded from the mean used for the full-budget ETA.
  This is only a timing exclusion; the native LR schedule is unchanged, including
  Churches' 10,000-update LR warmup.
* CUDA synchronization surrounds `Trainer.train_step`, including cached data
  access, posterior draws, corruption, forward/backward, gradient norm, AdamW
  and EMA. The first stage is frozen and absent during cached training.
* ETA is mean update time × configured paper updates, for one seed on one GPU:
  635,000 / 410,000 / 500,000 / 1,900,000 respectively. It excludes downloads,
  encoding/statistics, evaluation, sample generation and checkpoint writes.
* Churches uses the common L2 preset for every arm; the native L1 protocol is a
  separate experiment. No reduced effective batch or smaller U-Net is used.

The source base is `6ac7427`. Each result records the exact package and benchmark
script SHA-256, runtime, resolved configuration, first-stage checkpoint identity,
cache identity, initial U-Net hash, timing variation and peak CUDA allocation.
The [exact executed script](benchmark_ldm.executed.py.txt) matches the recorded
script hash; the published runner only had its standard-library imports reordered
after measurement to satisfy the linter. Training code was unchanged.
Only one benchmark job occupies the GPU at a time. Ordinary desktop activity
and clock variation are not controlled; these ETAs are estimates, not deadlines
or a claim that small speed differences between methods are statistically robust.

## Actual images and limits

| Dataset | Train / validation subset | Source and partition |
|---|---:|---|
| FFHQ | 84 / 4 | Original 1024 PNGs from NVIDIA metadata; publisher MD5 verified; original CompVis split membership |
| CelebA-HQ | 96 / 4 | 1024 PNG mirror at pinned revision; custom disjoint timing split; original `imgHQ` index mapping is unverified |
| LSUN Churches | 192 / 4 | Original publisher training LMDB; encoded bytes preserved; original CompVis split membership |
| LSUN Bedrooms | 96 / 4 | fast.ai repack and its Hugging Face copy; original image keys and CompVis split membership; publisher-LMDB pixel equivalence not audited |

Exact lists and download manifests are retained in [subsets/](subsets/). Training
statistics use only the listed training images. Validation images are encoded to
use the normal cache contract, but validation is not run inside the timed loop.
The subset sizes are exact multiples of the effective batch, so no timed update
uses a short final batch. Each arm sees the same repeated subset and noise draws.
These small caches fit in RAM; full-dataset storage behavior may add overhead.

FFHQ and Churches originals are also linked into their normal image directories,
with no false complete-dataset marker. The complete Churches archive and LMDB
are reusable under the standard downloader cache path. All four original LDM
checkpoint bundles are under `pretrained/ldm/`. Large data, caches and weights are
ignored by Git. Benchmark runs intentionally do not save large trained checkpoints.
Their download receipts are archived in [checkpoints/](checkpoints/); the subset
records also include cache manifests with the latent and statistics file hashes.

Sources: [FFHQ](https://github.com/NVlabs/ffhq-dataset),
[CelebA-HQ PNG mirror](https://huggingface.co/datasets/Iceclear/CelebA-HQ1024/tree/651d03038cb239c8653a93fe7bd0295919ea9c91),
[original LSUN](https://github.com/fyu/lsun),
[Bedrooms repack provenance](https://huggingface.co/datasets/pcuenq/lsun-bedrooms/blob/1acf5c322e8f1e3e440adf7130d697a677110119/README.md).

## Reproduce

The timing entry point is [benchmark_ldm.py](../../scripts/benchmark_ldm.py).
Use an existing full latent cache, or explicit subset lists with a separate cache.
The same effective batch, microbatch, precision and source are required to compare
these timings. The run writes each update and a per-arm summary; invocations queue
through a shared process lock on Linux/macOS.

```bash
uv run --locked --extra ldm python scripts/benchmark_ldm.py \
  -c configs/ldm/ffhq.json -o saved/ffhq_timing100 \
  --steps 100 --warmup 10 --set training.microbatch_size=4
```

Portable copies of the measured subset configurations are in [configs/](configs/).
With the recorded image subsets installed under `data/ldm_benchmark100/`, run:

```bash
uv run --locked --extra ldm python scripts/benchmark_ldm.py \
  -c verification/2026-09-21-ldm-timing/configs/ffhq.json \
  -o saved/ffhq_subset_timing_repeat --prepare --steps 100 --warmup 10
```

Change the config filename for another dataset and choose a fresh output path.
These configs describe the measured subsets; they do not download a full dataset.
The original working configs are under `saved/ldm_benchmark100/configs/`;
the original per-update logs remain under
`saved/ldm_benchmark100/<model>_micro4/runs/` (Churches uses `churches_micro4`).
