# Provenance and architecture preservation

Originally prepared: 2026-09-19; updated for the companion layout on 2026-09-20.
This project does not claim full upstream runner equivalence or reproduced FID.
No pretrained models or datasets are bundled.

## Native unconditional LDM modules

`fourier_score/ldm/upstream/` vendors `openaimodel.py`, `model.py`, `attention.py`,
`util.py` and `ema.py` from CompVis latent-diffusion revision
`a506df5756472e2ebaf9078affdde2c4f1502cd4`. Only package imports are changed;
the upstream/local SHA-256 pairs are recorded in `upstream/PROVENANCE.json`.
The MIT license is included. Native training YAML snapshots under
`configs/ldm/upstream/` come from that revision's `configs/latent-diffusion/`.

The frozen KL/VQ wrapper, cache, native epsilon adapter, training loop and sampler
integration are new implementations. KL/VQ module names and computational state
shapes match the original first-stage classes. VQ lookup follows
`taming-transformers@3ba01b241669f5ade541ce990f7650a3b8f65318`, with only query
chunking to bound memory; its MIT license is also included. Face preprocessing
follows that source's ImagePaths/NumpyPaths, and LSUN preprocessing follows the
pinned CompVis source. Current OpenCV/Pillow/PyTorch versions are recorded by the
lockfile; historical-library bitwise equivalence is not claimed.

Paper Tables 1/12 supply sampling NFE, effective batches, target learning rates
and update budgets. The native L1 Churches loss is preserved; an explicitly
labeled L2 protocol changes that common loss. See [LDM.md](LDM.md) for scope,
exact choices and the distinction between architecture/configuration fidelity
and long-training quality reproduction.

## Project organization

The initial template-style layout, JSON configuration, and checkpoint lifecycle
were informed by
https://github.com/victoresque/pytorch-template . Its README was inspected; its
old MNIST classifier, legacy DataParallel setup, and old config parser were not
copied. The training/configuration infrastructure here was written for this
image-generation project. The implementation now lives in the `fourier_score` package; the training loop
has no BaseTrainer inheritance. This is not an unchanged checkout of that repository.

## Actual NCSN++ source

Source: https://github.com/junyeopYim/fourier-score

Pinned source commit: `54db4b1bff975a72ef4ff2d301bd9cdb08e02c04`.

That repository derives its model code from
`junyeopYim/score_sde_pytorch@534c75478bf30acbff83568a52d843d0ff99913f`,
which in turn derives from https://github.com/yang-song/score_sde_pytorch .
Copyright 2020 The Google Research Authors; Apache-2.0 attribution is retained.

The following files are byte-for-byte identical to the pinned immediate source.
Their Git blob identities are recorded below; the focused backbone tests check
paired initialization and the original score-output convention.

| Packaged file | Original Git blob SHA-1 |
|---|---|
| fourier_score/backbones/ncsnpp.py | ea16eb35b5e6f2fa92db10be2552b3acdfe8fa7b |
| fourier_score/backbones/layers.py | eb772b2e6606ed92295dd031cb43be8a82a992c7 |
| fourier_score/backbones/layerspp.py | 2eb4e5de372e799f0608272408718664c35719c0 |

This preserves the U-Net module ordering, attention/NIN, normalization, dropout,
BigGAN++ and DDPM++ residual blocks, time conditioning, initialization,
progressive paths, and learned parameter shapes. Relative to the original
Google repository, the immediate source already contained package-layout
adaptations, pruned unused primitives, and a keyword-only interpolation fix.
The byte identity assertion is to the pinned **immediate source**, not to every
file in the Google repository.

### Deliberate changes around the backbone

* `fourier_score/model.py` generates the old backbone config from the single new config
  schema. `scale_by_sigma=False` is used in **all** experiment arms. The
  deterministic division by sigma is performed in the shared objective adapter.
  No learned layer is added or removed by selecting the objective. A test checks
  that the score adapter matches the source's inline sigma division exactly.
* The source sigma buffer is converted to float32 on CPU **before** transfer to
  MPS. Shape and learned parameters do not change.
* Fourier Gaussian adds fixed mean/power buffers and fixed linear spectral
  filters, not learned parameters. It does change the mathematical output
  parameterization; only the **backbone** and its initial weights are identical.
* `fourier_score/backbones/up_or_down_sampling.py` replaces 6-dimensional zero-insertion
  padding with equivalent 4-dimensional padding for MPS portability. The kernel,
  zero insertion, cropping, convolution and downsampling math are preserved.
  An independent copy of the original 6D algorithm serves as the value/gradient
  test oracle. The immediate source already used native PyTorch FIR and fixed
  an unused transposed-convolution branch.

The original upfirdn/resampling lineage includes StyleGAN2 as recorded in NOTICE.
This archive does not grant rights to separately obtained NVIDIA weights/data.

## Objective and infrastructure implementation

The MMSE scalar gate, linear gate, isotropic variants, and corresponding mode
names were **not** carried over. The output parameterizations are `score`, `diffusion`, `scalar_gaussian`,
`fourier_gaussian_unscaled`, and `fourier_gaussian`. All share the DSM objective.

Fourier Gaussian implements the previous discussion's unattenuated Gaussian
score plus frequencywise residual scaling. The mathematical equations and
limitations are in MATH.md. The matrix-DFT path, strict config, safe checkpointing,
stateful sampler, evaluation, CLI helpers and tests are new implementations.

Noise-prediction and score losses share a forward process unless the user
explicitly selects a different process preset. Discrete DDPM presets are not
passed off as a loss-only comparison to continuous VE. CIFAR architecture and
core source hyperparameters are available; the complete original input pipeline,
TensorFlow FID stack and sampler implementation are not claimed to be replicated.
In particular, the new default samplers and folder-image preprocessing define
new evaluation protocols.

The CelebA folder preset uses continuous VE (the old source preset used discrete
SMLD). The FFHQ folder preset preserves the source model-size settings but reads
ordinary image files, not the source TFRecords; neither preset implies paper
reproduction.

## Verification and dependencies

The tracked lockfile fixes the software environment. Dated test, device, and
architecture records are indexed in [verification/README.md](../verification/README.md).
Those records apply to the source revisions they identify; they do not establish
long-training FID/IS reproduction. Current commands and checkpoint migration
boundaries are in [DEVELOPMENT.md](DEVELOPMENT.md).

## Official Score-SDE downloads

The reference catalog in `scripts/download_score_sde.py` uses the original
Google Drive folders linked by `yang-song/score_sde_pytorch` at revision
`cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44`. Downloaded source configs and license
are saved verbatim with the original checkpoint, outside Git. Their local
manifests retain URLs, hashes and sizes. No downloaded Python is executed.

`scripts/import_score_sde.py` strictly maps model keys and the ordered EMA tensors
to the existing backbone and exports only an inference snapshot. A missing,
deterministic `sigmas` buffer can be rebuilt from config; this buffer is unused
by the supported continuous Fourier embedding. All learned weights must exist.
The local score adapter performs the original sigma division exactly once.
The exported metadata and generated-sample metadata retain the original source
identity. See [SCORE_SDE.md](SCORE_SDE.md) for scope and evaluation differences.

## Dataset download sources

`fourier_score/ldm/dataset_sources.py` records the original CompVis train/validation
lists and SHA-256 values. Face lists are pinned to taming-transformers revision
`3ba01b241669f5ade541ce990f7650a3b8f65318`; LSUN lists come from the authors'
`https://ommer-lab.com/files/lsun.zip`. The downloader preserves image identities
and never generates a replacement holdout split.

FFHQ image URLs, sizes and MD5 values come from NVIDIA's
[official metadata](https://github.com/NVlabs/ffhq-dataset). The metadata/license
remain in the ignored local source cache. LSUN image bytes come from the training
LMDB URLs in the [official downloader](https://github.com/fyu/lsun/blob/master/download.py)
and use its original key-based `.webp` export convention. For CelebA-HQ, users supply
the original `.npy` files described by
[CompVis](https://github.com/CompVis/taming-transformers#celeba-hq) and the
[PGGAN reconstruction instructions](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training).
The repository does not redistribute any dataset or downloaded weights.
