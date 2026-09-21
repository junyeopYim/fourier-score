# Official Score-SDE reference weights

The supported references are original PyTorch releases from the
[Score-SDE model zoo](https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/README.md#pretrained-checkpoints).
The upstream source/config revision is pinned to
`cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44`. These are continuous VE NCSN++ models,
with `likelihood_weighting=False` and the source's sigma-squared-weighted DSM.

| `--model` | Configured residual blocks | Resolution | Released checkpoint |
|---|---:|---:|---|
| `cifar10_ncsnpp_continuous` | `num_res_blocks=4` | 32 | `checkpoint_24.pth` |
| `cifar10_ncsnpp_deep_continuous` | `num_res_blocks=8` | 32 | `checkpoint_12.pth` |
| `ffhq_256_ncsnpp_continuous` | `num_res_blocks=2` | 256 | `checkpoint_48.pth` |

This catalog covers the repository's continuous NCSN++ references; it does not
claim support for every upstream architecture, VP/sub-VP model, resolution or
discrete SMLD checkpoint. In particular, the CelebA-64 folder preset is a new
continuous-VE protocol and has no directly substituted pretrained reference here.

## Download and reuse

```bash
python scripts/download_score_sde.py --list
python scripts/download_score_sde.py --all --dry-run
uv run --locked --extra datasets python scripts/download_score_sde.py --all
# Or fetch only the standard CIFAR-10 model:
uv run --locked --extra datasets python scripts/download_score_sde.py \
  --model cifar10_ncsnpp_continuous
```

Each bundle is installed under `pretrained/score_sde/<model>/` and contains:

* the unchanged original `checkpoint_N.pth`, including upstream training state;
* the commit-pinned model config, its default config and license under `upstream/`;
* `download.json` with the source folder/file URLs, revision, sizes and SHA-256 values.

Completed bundles are checked offline before reuse. Partial Google Drive downloads
can resume. Existing unmanaged/corrupted files are rejected instead of overwritten.
Google Drive's access/quota errors remain explicit. No publisher checkpoint digest
is supplied by the model-zoo table; the recorded hashes identify received bytes.
`--sha256 HEX` enforces a separately trusted digest for one `--model`.
Downloading does not load the checkpoint or execute its Python config files.

## Import EMA and sample

```bash
uv run --locked python scripts/import_score_sde.py --model cifar10_ncsnpp_continuous
uv run --locked python sample.py \
  -r pretrained/score_sde/cifar10_ncsnpp_continuous/ema.pt \
  -o saved/official_cifar10_reference --device cuda --num-samples 64 --batch-size 64
```

Use the same import command with the other model names. `--input-dir` changes
the bundle prefix; `--output` selects a new export path. By default the export
is `ema.pt` beside the original checkpoint. Repeating an unchanged conversion
reuses a matching export. An incompatible existing export requires a new path.

The converter loads tensors with `weights_only=True`, strips the original
DataParallel `module.` prefix, and strictly matches model keys and tensor shapes.
The EMA list must match the exact order and shapes of trainable parameters.
Frozen Fourier embedding weights come from the raw model state. Released files
can omit `sigmas`; only this deterministic buffer is reconstructed from the
corresponding config. Continuous Fourier conditioning passes sigma directly, so
that buffer is unused in the supported forward path. Missing learned tensors,
missing EMA entries and ambiguous parameter ordering fail the conversion.

The original NCSN++ score is obtained by dividing the raw network output by sigma
exactly once. The local `score` adapter performs this division outside the backbone.
The imported artifact is **inference-only**: it does not synthesize a training RNG,
data cursor, optimizer state or a trainable Fourier reference. `train.py --resume`
requires this project's own training checkpoints.

Export metadata records the original checkpoint hash, model/revision, conversion
script hash and EMA selection. Sampling and DSM evaluation propagate this identity.
No training mean/power is needed for the pure score reference.

## Interpretation

Pretrained references have different training histories from the controlled
from-scratch arms. The standard and deep CIFAR-10 references also have different
architectures. Keep them labeled separately from matched-budget learning curves.

Sampling uses this repository's implementation and recorded batch/NFE/precision.
It is not a claim of reproducing the source's full sampler, data pipeline,
TensorFlow FID/IS or paper result. Use the same local evaluation protocol when
comparing generated images. The exact checked downloads and smoke execution are
recorded under [verification/](../verification/README.md).
