# Provenance and architecture preservation

Prepared: 2026-09-19. This is a new self-contained research project, not a claim
of full upstream runner equivalence or reproduced FID. No GitHub repository was
modified. No pretrained models, datasets, credentials, or proprietary fonts are
included.

## Project organization

The base/model/data_loader/trainer/logger/utils separation, JSON configuration,
train/test CLI, and checkpoint lifecycle follow the design of
https://github.com/victoresque/pytorch-template . Its README was inspected; its
old MNIST classifier, legacy DataParallel setup, and old config parser were not
copied. The training/configuration infrastructure here was written for this
image-generation project. This is template-style organization, not an unchanged
checkout of that repository.

## Actual NCSN++ source

Source: https://github.com/junyeopYim/fourier-score

Pinned source commit: `54db4b1bff975a72ef4ff2d301bd9cdb08e02c04`.

That repository derives its model code from
`junyeopYim/score_sde_pytorch@534c75478bf30acbff83568a52d843d0ff99913f`,
which in turn derives from https://github.com/yang-song/score_sde_pytorch .
Copyright 2020 The Google Research Authors; Apache-2.0 attribution is retained.

The following files are byte-for-byte identical to the pinned immediate source.
`tests/test_backbone.py` independently computes the Git blob hash of their bytes.

| Packaged file | Original Git blob SHA-1 |
|---|---|
| model/backbones/ncsnpp.py | ea16eb35b5e6f2fa92db10be2552b3acdfe8fa7b |
| model/backbones/layers.py | eb772b2e6606ed92295dd031cb43be8a82a992c7 |
| model/backbones/layerspp.py | 2eb4e5de372e799f0608272408718664c35719c0 |

This preserves the U-Net module ordering, attention/NIN, normalization, dropout,
BigGAN++ and DDPM++ residual blocks, time conditioning, initialization,
progressive paths, and learned parameter shapes. Relative to the original
Google repository, the immediate source already contained package-layout
adaptations, pruned unused primitives, and a keyword-only interpolation fix.
The byte identity assertion is to the pinned **immediate source**, not to every
file in the Google repository.

### Deliberate changes around the backbone

* `model/model.py` generates the old backbone config from the single new config
  schema. `scale_by_sigma=False` is used in **all three** experiment arms. The
  deterministic division by sigma is performed in the shared objective adapter.
  No learned layer is added or removed by selecting the objective. A test checks
  that the score adapter matches the source's inline sigma division exactly.
* The source sigma buffer is converted to float32 on CPU **before** transfer to
  MPS. Shape and learned parameters do not change.
* Fourier Gaussian adds fixed mean/power buffers and fixed linear spectral
  filters, not learned parameters. It does change the mathematical output
  parameterization; only the **backbone** and its initial weights are identical.
* `model/backbones/up_or_down_sampling.py` replaces 6-dimensional zero-insertion
  padding with equivalent 4-dimensional padding for MPS portability. The kernel,
  zero insertion, cropping, convolution and downsampling math are preserved.
  An independent copy of the original 6D algorithm serves as the value/gradient
  test oracle. The immediate source already used native PyTorch FIR and fixed
  an unused transposed-convolution branch.

The original upfirdn/resampling lineage includes StyleGAN2 as recorded in NOTICE.
This archive does not grant rights to separately obtained NVIDIA weights/data.

## Objective and infrastructure implementation

The MMSE scalar gate, linear gate, isotropic variants, and corresponding mode
names were **not** carried over. The only objective names are `score`,
`diffusion`, and `fourier_gaussian`.

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

## Latest dependencies / execution distinction

The GitHub release APIs were read on 2026-09-19:

* https://github.com/pytorch/pytorch/releases/tag/v2.14.0
* https://github.com/pytorch/vision/releases/tag/v0.29.0

The project targets that exact package pair. Actual local tests used the already
installed torch 2.10.0 CPU runtime. New package resolution failed; no uv.lock or
2.14 runtime certification was fabricated. See VALIDATION.md and INSTALL.md.
