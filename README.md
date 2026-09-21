# Fourier Score

A research companion for **Fourier Gaussian output parameterization** in image generation.
We compare the same NCSN++ backbone and denoising score matching (DSM) objective,
changing how the network output is converted into a score.

For noisy data `y = αx + σε`, the proposed parameterization is

```text
σ sθ(y,t) = σ sG(y,t) + F⁻¹[b(t) · F(hθ(y,t))]
b(t)     = sqrt(α²P / (α²P + σ²))
```

`μ` and `P` are estimated from training data. The Gaussian reference `sG` has
coefficient one. Start with [the method](fourier_score/method.py),
[statistics estimation](fourier_score/statistics.py), and [the derivation](docs/MATH.md).

## Quick start

Run commands from the repository root. Python 3.11 and the tracked lockfile are used.

```bash
uv sync --locked --python 3.11

# Three training updates on synthetic data; no dataset download.
uv run --locked python train.py -c configs/smoke.json --device cpu
uv run --locked python sample.py \
  -r saved/synthetic_ve_fourier_gaussian_s0/last.pt -o saved/smoke_samples
uv run --locked python evaluate.py dsm \
  -r saved/synthetic_ve_fourier_gaussian_s0/last.pt -o saved/smoke_dsm.json
```

This checks the pipeline, not image quality. Existing run/sample directories are
not overwritten; use a new `--set name=...` and output path when repeating it.
For real-data training, `train.py --download` downloads MNIST/CIFAR-10 and prepares
the training statistics automatically.

## Run an experiment

The main CIFAR-10 presets inherit directly from [base.json](configs/base.json).

| Preset | Optimizer updates | Training images | Held-out DSM split |
|---|---:|---:|---|
| [50K pilot](configs/cifar10_ablation.json) | 50,000 | 45,000 | 5,000-image validation |
| [950K](configs/cifar10_950k.json) | 950,000 | 50,000 | CIFAR-10 test |
| [1.3M](configs/cifar10_1m3.json) | 1,300,000 | 50,000 | CIFAR-10 test |

The 950K and 1.3M protocols differ only in update budget and run name. Test DSM is
for reporting, not hyperparameter selection. Run names include the budget,
split, parameterization, and seed. Here **950K means 950,000**, not 95,000.

```bash
# Inspect the paired comparison without downloading data or starting training.
bash scripts/reproduce_cifar10.sh 950k --device cuda --seeds 42 --dry-run

# Train one arm. --parameterization selects an output convention, not a new loss.
uv run --locked python train.py -c configs/cifar10_950k.json --download \
  --device cuda --parameterization fourier_gaussian --set trainer.microbatch_size=32
```

Supported comparisons: `score`, `scalar_gaussian`, `fourier_gaussian_unscaled`,
and `fourier_gaussian`. `diffusion` is an optional sign-convention check.
See [experiment recipes](docs/EXPERIMENTS.md) for paired seeds, sampling, FID/IS,
and legacy presets; see [ablations](docs/ABLATIONS.md) for interpretation.

## Read or modify the code

| Question | Start here |
|---|---|
| What is the proposed method? | [method.py](fourier_score/method.py) |
| How are μ and P estimated without validation leakage? | [statistics.py](fourier_score/statistics.py), [data.py](fourier_score/data.py) |
| How does the unchanged backbone produce a score? | [model.py](fourier_score/model.py), [backbones/](fourier_score/backbones/) |
| What is optimized? | [loss.py](fourier_score/loss.py), [training.py](fourier_score/training.py) |
| How are samples and metrics computed? | [diffusion.py](fourier_score/diffusion.py), [evaluation.py](fourier_score/evaluation.py) |

The public entry points are **train.py**, **sample.py**, and **evaluate.py**.
Supporting preparation and diagnostic tools live in `scripts/`. EMA, RNG state,
consumed-batch resume, and the original backbone parameter names are preserved.

## Official LDM weights

```bash
python scripts/download_ldm.py --list
python scripts/download_ldm.py --model ffhq --dry-run
python scripts/download_ldm.py --model ffhq
```

This saves the checkpoint, matching upstream config, and download hashes under
`pretrained/ldm/ffhq/`. [The LDM guide](docs/LDM.md) covers frozen-first-stage
loading, KL/VQ latent caches, native U-Net training, EMA/DDIM sampling and RGB FID.

```bash
uv sync --locked --extra ldm --extra metrics
uv run --locked --extra ldm python ldm.py inspect -c configs/ldm/lsun_churches.json
# Requires the official weights and dataset/split lists described in docs/LDM.md.
uv run --locked --extra ldm python ldm.py prepare -c configs/ldm/lsun_churches.json --device cuda
uv run --locked --extra ldm python ldm.py compare -c configs/ldm/lsun_churches_l2.json --seeds 42 --dry-run
```

The separate `ldm.py` CLI preserves the pixel-space checkpoint format. LDM
presets use the paper's effective batch/LR/update budgets and pinned native
architectures. Churches' upstream L1 loss is retained; the `_l2` preset explicitly
changes the common loss for all compared parameterizations.

## Scope and reproducibility

This repository supports pixel-space MNIST, CIFAR-10, and image-folder experiments
on one CPU, CUDA GPU, or MPS device. Unconditional CompVis LDM experiments are
available through the optional `ldm` dependencies; text/class conditioning is
outside this implementation. **Score-SDE pretrained import/download is not implemented yet.**
See [planned extensions](docs/EXPERIMENTS.md#planned-extensions).
No long-training FID/IS result is claimed by the smoke tests or architecture audits.

FID/IS uses the optional `torch-fidelity` dependency and downloads Inception
weights on first use. Re-evaluate all compared models under the same protocol;
these values are not interchangeable with the original Score-SDE TensorFlow metrics.

- [Installation and devices](docs/INSTALL.md) · [Configuration](docs/CONFIG.md) · [MPS](docs/MPS.md)
- [Development, tests, and migration](docs/DEVELOPMENT.md) · [Historical verification](verification/README.md)
- [Source attribution](docs/PROVENANCE.md) · [한국어 사용 안내](docs/USAGE.md)

The NCSN++ code derives from Score-SDE through the pinned fork documented in
[PROVENANCE.md](docs/PROVENANCE.md). Preserve the attributions in [NOTICE](NOTICE)
when reusing the code.
