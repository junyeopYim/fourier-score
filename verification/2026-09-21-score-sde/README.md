# Research README and official Score-SDE references — 2026-09-21

Working-tree verification based on `0f22f8a`.
Package source SHA-256: `af75ddfa916e5e6a4e909f3b15ac20da85317f30c08f46f76f8972c930fcc969`.
Raw download manifests, checkpoint identities, CUDA forward checks and inference
settings are recorded in [summary.json](summary.json).

## Download and conversion

All three supported original checkpoints were downloaded from the official
Score-SDE Google Drive model zoo, with source configs and license pinned to
`cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44`. Bundles live under the ignored
`pretrained/score_sde/` directory; weights are not committed.

| Model | Original file | Bytes | Recorded training step | Parameters including frozen embedding |
|---|---|---:|---:|---:|
| CIFAR-10 NCSN++ | `checkpoint_24.pth` | 1,004,850,464 | 1,200,000 | 62,758,915 |
| CIFAR-10 NCSN++ deep | `checkpoint_12.pth` | 1,722,588,864 | 600,000 | 107,587,075 |
| FFHQ-256 NCSN++ | `checkpoint_48.pth` | 1,049,992,328 | 2,400,007 | 65,574,549 |

Each bundle passed offline size/SHA-256 verification and strict EMA import to
`ema.pt`. Repeating conversion reused the existing export. Every imported model
produced a finite CUDA score with its native image shape at continuous time 0.5.
Only the missing deterministic `sigmas` buffer was reconstructed; original
learned tensors and fixed Fourier embeddings were retained.

```bash
python -S scripts/download_score_sde.py --all --dry-run
uv run --locked --extra datasets python scripts/download_score_sde.py --all
uv run --locked python scripts/import_score_sde.py --model cifar10_ncsnpp_continuous
uv run --locked python scripts/import_score_sde.py --model cifar10_ncsnpp_deep_continuous
uv run --locked python scripts/import_score_sde.py --model ffhq_256_ncsnpp_continuous
uv run --locked python sample.py \
  -r pretrained/score_sde/cifar10_ncsnpp_continuous/ema.pt \
  -o saved/score_sde_reference_smoke --device cuda --num-samples 4 --batch-size 4
```

The public CIFAR-10 sample command completed with **4 images, 1,000 PC steps,
one corrector step and 2,000 batched score calls**. Elapsed sampling time was
67.85 seconds on an NVIDIA GeForce RTX 5060 Ti in FP32. The generated preview
was inspected, and sampling metadata retained the original checkpoint hash,
upstream revision, converter hash and EMA selection. This is an execution check,
not an image-quality estimate.

## README and focused checks

The README's synthetic workflow completed three optimizer updates, generated four
EMA samples and evaluated four held-out synthetic images. Its final pixel-mean
DSM was 0.695274. Both comparison dry runs resolved successfully: four pixel arms
and three latent arms, each with seeds 42, 43 and 44.

```bash
uv run --locked python train.py -c configs/smoke.json --device cpu --set name=readme_smoke
uv run --locked python sample.py -r saved/readme_smoke/last.pt -o saved/readme_smoke_samples
uv run --locked python evaluate.py dsm -r saved/readme_smoke/last.pt -o saved/readme_smoke_dsm.json
bash scripts/reproduce_cifar10.sh 50k --device cuda --seeds 42 43 44 \
  --set trainer.microbatch_size=32 --dry-run
uv run --locked --extra ldm python ldm.py compare \
  -c configs/ldm/lsun_churches_l2.json --device cuda --seeds 42 43 44 --dry-run
uv run --locked --extra ldm --extra metrics --extra datasets python -m pytest -q
uv lock --check --offline
uv run --locked --extra figures python scripts/plot_method.py
```

The focused suite passed **45 tests in 8.66 seconds** ([output](pytest.txt)). One
new integration test covers verified local HTTP download and offline reuse,
ordered EMA mapping, fixed embeddings, the real missing-buffer case, public
inference/DSM metadata and corrupt-file rejection. No public model is downloaded
by pytest. New Python files passed Ruff, and local documentation links resolved.

The committed SVG/PNG method figure was rendered and visually inspected. Its
curve is the analytic residual scale, not an empirical training curve. No full
FID, long-training result, learning-efficiency improvement or equivalence to the
complete upstream sampling/evaluation protocol was established by these checks.
