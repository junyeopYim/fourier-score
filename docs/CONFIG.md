# Configuration

Run commands from the repository root. `configs/base.json` defines the pixel
schema and defaults; other JSON presets override fields through `extends`.
Inheritance paths are relative to the config file. Data and output paths are
relative to the working directory. Native latent presets use their own schema
in `fourier_score/ldm/config.py`; see [LDM.md](LDM.md).

```bash
uv run --locked python train.py -c configs/cifar10_950k.json --dry-run
uv run --locked python train.py -c configs/cifar10_950k.json --dry-run \
  --parameterization scalar_gaussian --set trainer.microbatch_size=32
```

Unknown keys, invalid types, incompatible process/sampler combinations and
inconsistent image/architecture shapes are rejected. Quote list overrides:
`--set 'arch.args.ch_mult=[1,2,2,2]'`.

| Setting | Field |
|---|---|
| Dataset, image shape, effective batch, workers | `data_loader.args.*` |
| Learned backbone layers | `arch.args.*` |
| Corruption process and noise schedule | `process.*` |
| Output parameterization / DSM reduction | `parameterization` / `loss.reduction` |
| Gaussian power floor and cache | `fourier.*` |
| Adam | `optimizer.args.*` |
| Update budget, warmup, clipping, EMA, microbatch | `trainer.*` |
| Device | `device` or `--device` |
| Precision and spectral implementation | `backend.*` |
| Sampling seed, batch size and solver | `sampling.*` |
| Diagnostic seed, batch size, noise/frequency bins | `evaluation.*` |

Supported pixel parameterizations are `score`, `diffusion`, `scalar_gaussian`
and `fourier_gaussian`. `--parameterization`, top-level `parameterization` and
`--set parameterization=...` resolve to the checkpoint's `loss.type` field.
Conflicting aliases are rejected. Selecting a parameterization does not change
the architecture, process or sampler. A `_ddpm.json` preset explicitly changes
the forward process and sampler and is a separate experimental comparison.

The backbone config is derived from the single image shape and process schedule.
Sigma division occurs once in the output adapter. The scalar and Fourier arms
share training statistics and the full mean image. Set
`evaluation.frequency_bins=6` to record radial-frequency DSM diagnostics; zero
disables them. The 50K pilot enables these diagnostics by default.

## Run naming and comparison

New presets use `{parameterization}` and `{seed}` in their names, for example
`cifar10_950k_full_fourier_gaussian_s42`. `name=auto` produces
`{dataset}_{process}_{loss}_s{seed}`. Completed output directories are never
overwritten. The comparison runner adds an arm/seed suffix to fixed names.

`--parameterizations` (alias `--objectives`) selects arms; `--seeds 42 43 44`
selects paired seeds. A dry run prints commands without downloading, preparing
statistics or training. A real comparison checks all destinations, prepares a
shared cache, then runs sequentially. It does not automatically resume old runs.

## Logging

`trainer.console=human` prints preparation, progress, checkpoint paths and ETA.
Terminal updates use `trainer.progress_every_seconds` (default 5), independently
of JSONL `trainer.log_every`. `console=json` emits JSON records to stdout;
`console=quiet` keeps only file logging. Redirect human logs with `2>&1 | tee`.

`loss_avg` is image-weighted over the logging window. `loss_pixel_mean_avg`
converts `half_sum` reduction using `2 / (C*H*W)`; evaluation also uses pixel-mean
DSM, but with EMA weights and separate observations. Throughput includes data
waiting and updates; the train ETA excludes future evaluation and saves. CUDA
memory fields report PyTorch allocation/reservation, not whole-device usage.

```bash
uv run --locked --extra tensorboard python train.py -c configs/cifar10_950k.json \
  --set trainer.tensorboard=true
uv run --locked --extra tensorboard tensorboard --logdir saved
```

## Resume

`train.py -r last.pt` uses the saved resolved config and rejects simultaneous
`-c`. It permits limited operational overrides such as update budget and
logging, but verifies data/split fingerprints, source, architecture, objective,
optimizer, microbatch, backend, PyTorch version and device. Inference can relocate
devices and records the new environment.

The checkpoint records the **consumed** batch cursor rather than worker
prefetch position. Evaluation and saving isolate RNG state. Incomplete optimizer
steps are not saved. A longer update budget retains the existing run name.
Use the original source revision and environment to continue existing runs;
see [DEVELOPMENT.md](DEVELOPMENT.md) for debugging and migration boundaries.
