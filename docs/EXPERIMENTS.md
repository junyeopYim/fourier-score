# Experiment recipes

Run from the repository root. The commands below describe executable protocols;
they do not claim that the long training runs or their FID/IS results are complete.

## CIFAR-10 protocols

| Config | Updates | Train / diagnostic split | Default run prefix |
|---|---:|---|---|
| `configs/cifar10_ablation.json` | 50,000 | 45,000 / 5,000 validation | `cifar10_50k_holdout5000` |
| `configs/cifar10_950k.json` | 950,000 | 50,000 / test | `cifar10_950k_full` |
| `configs/cifar10_1m3.json` | 1,300,000 | 50,000 / test | `cifar10_1m3_full` |

Every prefix is followed by `_{parameterization}_s{seed}`. The two long-run presets
share the same data, optimizer, architecture, sampler, and diagnostic settings.
The pilot has a different split and is not a controlled comparison of budget alone.
Use validation/pilot runs to select hyperparameters; do not tune on the test split.

```bash
# Preview all commands: no data loading, downloading, or training.
bash scripts/reproduce_cifar10.sh 50k --device cuda --seeds 42 --dry-run

# Two Gaussian arms, three paired seeds; six runs without baseline retraining.
bash scripts/reproduce_cifar10.sh 950k --download --device cuda \
  --seeds 42 43 44 --parameterizations scalar_gaussian fourier_gaussian \
  --set trainer.microbatch_size=32

# Include a paired score baseline explicitly, or choose the 1.3M budget instead.
bash scripts/reproduce_cifar10.sh 1m3 --device cuda --seeds 42 --dry-run \
  --parameterizations score scalar_gaussian fourier_gaussian
```

The recipe calls `scripts/run_comparison.py`. Before training, it checks **all**
destinations and prepares the common statistics cache once. It then runs arms
sequentially. Missing datasets require `--download`; the dry run never downloads.
A failed comparison can be resumed per run with `train.py -r <last.pt>`; the
comparison runner does not automatically skip or resume existing runs.

The 50K pilot enables noise × frequency DSM diagnostics. Its early ranking does
not establish the final ranking. See [ABLATIONS.md](ABLATIONS.md) for the controls.

## GPU choice and runtime estimates

Start CIFAR-10 with one 24–32 GB NVIDIA GPU (RTX 4090 / RTX 5090) and an
effective batch of 128, accumulated in microbatches of 32. Check memory on the
actual host before a long run; smaller microbatches trade memory for extra time.
The current trainer uses FP32, disables TF32 by default, and has no AMP or DDP.
Multiple GPUs can run independent arms/seeds, but do not automatically accelerate
one training run. Keep backend and batch settings identical between compared arms.

Use a dedicated short run to measure the full CIFAR-10 architecture, not the
smaller synthetic smoke model:

```bash
uv run --locked python train.py -c configs/cifar10_950k.json --download \
  --device cuda --parameterization fourier_gaussian \
  --set name=timing_cifar10_fg_s42 --set trainer.iterations=500 \
  --set trainer.microbatch_size=32 --set evaluation.batch_size=32
```

Give each additional timing run a new name. Use the `steps_per_second` windows
after the first 100 updates in `saved/timing_cifar10_fg_s42/metrics.jsonl`.
For `r` updates/second, `U` updates, `A` arms, `S` seeds, and price `p` USD/hour:

```text
training hours = U / r / 3600
GPU cost       = training hours * p * A * S
```

This excludes setup, periodic diagnostics, checkpoint I/O, and final sampling/FID.
Measure each selected arm; do not assume identical throughput. At an illustrative
1 update/second, 50K / 950K / 1.3M take 13.9 / 263.9 / 361.1 training hours per
arm and seed. These are arithmetic scenarios, not measured GPU benchmarks.
The default three-arm recipe with three seeds means **9 runs**; the explicit
two-Gaussian-arm recipe above means **6 runs**.

Use Runpod on-demand Pods for these long jobs. Check current
[GPU rates](https://www.runpod.io/pricing) and
[storage billing](https://docs.runpod.io/pods/pricing) separately. Keep `saved/`
on persistent storage and preserve the Git revision / locked environment for
resume. A 48 GB card (RTX 6000 Ada / L40S) is a reasonable starting point for
the native LDM runs with a frozen autoencoder. The runner supports smaller cards
through microbatching; benchmark the selected architecture and effective batch
before budgeting full training.

## Sample and evaluate a checkpoint

```bash
uv run --locked python sample.py \
  -r saved/cifar10_950k_full_fourier_gaussian_s42/last.pt \
  -o saved/cifar10_950k_fg_50k_samples --device cuda \
  --num-samples 50000 --batch-size 64

uv run --locked python evaluate.py dsm \
  -r saved/cifar10_950k_full_fourier_gaussian_s42/last.pt \
  -o saved/cifar10_950k_fg_dsm.json --device cuda

uv sync --locked --extra metrics
uv run --locked python scripts/export_real.py -c configs/cifar10_950k.json \
  --split train -o saved/cifar10_real
uv run --locked python evaluate.py fid \
  --real saved/cifar10_real/png --generated saved/cifar10_950k_fg_50k_samples/png \
  --device cuda -o saved/cifar10_950k_fg_fid.json
```

Sampling always uses EMA weights. Both `last.pt` and `ema_*.pt` can be evaluated.
FID also reports IS; its Inception weights are downloaded on first use. For a model
trained on MPS, run FID/IS on CPU or CUDA. Export each shared real split once and
reuse it. Keep real split, preprocessing, sample count, batch size, sampler, NFE,
and metric implementation identical between arms.

Sampling writes PNGs, sharded uint8 NHWC NPZ files, a preview, and `settings.json`
with checkpoint hash/step, training time, generation time, score-call count, and
completion status. A partial output must not be treated as a completed evaluation.
Checkpoint sweeps and report aggregation are still manual.

Budget sampling separately: the default PC sampler has 1,000 predictor steps
and one corrector per step, or 2,000 score calls per image trajectory. At batch
64, 50,000 images require 782 full sampling batches; only 16 images are retained
from the last batch. Keeping the final sampling batch full preserves the
batch-mean Langevin normalization. Time a 64-image run with the exact
same sampler/batch settings, then multiply its `settings.json` `wall_seconds`
by 782 for an initial generation estimate. FID/IS computation is additional.
Reducing sampler steps changes the evaluation protocol and must be reported.

## MNIST and explicit process comparisons

```bash
uv run --locked python train.py -c configs/mnist.json --download \
  --parameterization fourier_gaussian
uv run --locked python scripts/run_comparison.py -c configs/mnist.json \
  --download --seeds 42 --parameterizations score scalar_gaussian fourier_gaussian
```

`--parameterization` and `--set parameterization=...` select the same internal
v1 `loss.type` field. This preserves checkpoint format and does not change the
backbone, forward process, or reduction. DDPM presets explicitly change both
process and sampler; compare them as a separate experiment, not a loss-only ablation.

## Legacy presets

These files preserve their original numerical protocols. Their run names now
separate the protocol, parameterization, and seed.

| File | Preserved settings |
|---|---|
| `cifar10.json` | 1,300,001 updates, 5,000-image holdout |
| `cifar10_full.json` | 1,300,001 updates, full train, 128 diagnostic images every 100 updates |
| `cifar10_paper950k.json` | 950,000 updates, full train, 128 diagnostic images every 100 updates |

Use the explicit `cifar10_950k.json` / `cifar10_1m3.json` pair for new budget
comparisons. The legacy `paper` filename does not establish paper reproduction.
The repository has no 95,000-update preset.

## Additional settings

Unconditional CompVis LDM first-stage loading, latent statistics, training from
scratch, native DDIM/DDPM sampling and decoded RGB evaluation are implemented in
`ldm.py`. See [LDM.md](LDM.md) for native paper presets and paired comparisons.
All arms share a frozen first stage; the public pretrained denoiser is a separate
reference. The optional dependencies keep pixel experiments independently runnable.

Official continuous-VE NCSN++ checkpoint download and EMA inference import are
available through [SCORE_SDE.md](SCORE_SDE.md). These public references are separate
from the matched, from-scratch arms.

Conditioned LDMs, pretrained LDM fine-tuning/conversion, and automatic aggregate
FID learning-curve reports remain future extensions.
