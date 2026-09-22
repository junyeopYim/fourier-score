# Controlled comparisons and interpretation

The study asks whether a fixed, frequency-dependent Gaussian reference helps a
network learn structure beyond first and second moments. The data distribution
need not be Gaussian. The backbone, corruption process and DSM objective are
shared; only the output parameterization changes.

| Parameterization | Reference | Residual scale | Role |
|---|---|---|---|
| `score` | None | 1 | Direct scaled-score baseline |
| `scalar_gaussian` | Full mean image, channelwise constant variance | Channelwise | Gaussian control without frequency-dependent covariance |
| `fourier_gaussian` | Full mean image, per-channel Fourier power | Frequencywise | Proposed parameterization |

`scalar_gaussian` averages the floored power over spatial frequencies **within
each channel** and uses that variance in both the reference and residual scale.
It does not use one shared scalar across channels. Its constant spectral
multiplier can be evaluated pixelwise without FFT. Both Gaussian arms use the
same training-only statistics cache and full mean image.

This comparison changes covariance in both the reference and normalization. It
does not separately identify their individual contributions. `diffusion` is an
optional noise-prediction sign convention; the native latent baseline is
`epsilon`. Neither Gaussian arm reproduces the complete EDM recipe.

## Current research questions

| Setting | Question / observation | Scope |
|---|---|---|
| MNIST | The two Gaussian parameterizations gave similar FID in the exploratory run; no additional benefit from Fourier covariance was observed there. | Qualitative observation from that run, not a multi-seed conclusion. |
| CIFAR-10 | Does frequency-dependent covariance help relative to scalar covariance? | Two Gaussian arms, three paired seeds, 950K updates; final results pending. |
| LSUN Churches | Does the construction help in the latent space of a fixed autoencoder? | Two Gaussian arms, three paired seeds, target 500K updates; final results pending. |
| Synthetic Gaussian / GMM | Can the residual recover non-Gaussian structure when the reference moments are exactly correct? | Exact-score mechanism test, not an image-quality benchmark. |

Public pretrained baselines or numbers cited from a paper must be labeled as
external references, with their architecture, training history and metric
protocol. They are not paired baseline retraining runs or extra random seeds.

## Paired controls

Hold the split, augmentation, seed, initial backbone, microbatch size, optimizer,
EMA, noise sampling, validation and sampler settings fixed. Identical initial
backbone weights do not imply identical initial interpreted scores. At a flat
per-channel spectrum, the scalar and Fourier operators coincide up to numerical
roundoff; this is an implementation control, not an expected quality gain.

```bash
# The paper's two-arm comparison: six runs, no baseline retraining.
bash scripts/reproduce_cifar10.sh 950k --device cuda --seeds 42 43 44 \
  --parameterizations scalar_gaussian fourier_gaussian \
  --set trainer.microbatch_size=32 --dry-run
```

Remove `--dry-run` and add `--download` to train. The comparison runner checks
all output destinations and prepares shared statistics once, then runs jobs
sequentially. It does not schedule remote GPUs. Resume an individual interrupted
run with `train.py -r <last.pt>` in its original source checkout.

## Measurements

Report FID against optimizer updates **and** elapsed training time. Keep real
reference split, preprocessing, number of generated images, sampling seed,
batch size, actual NFE and metric backend identical across arms. Report the mean
and variation across independent training seeds. Do not select only the best
seed or compare 45K pilot references with 50K full-training references.

`metrics.jsonl` contains held-out DSM and optional noise × frequency diagnostics.
Frequency-band means must be weighted by mode counts to recover total DSM.
Empty bands are null, not zero. These quantities measure noisy denoising error;
they are not exact true-score error on real images. Observation-level standard
error is distinct from variation across independently trained seeds.

Record statistics/cache preparation separately. `training_wall_seconds` includes
setup, training, diagnostics and previous checkpoint writes, excludes downtime
between resumed sessions, and is not GPU rental time. The latent path also
records `optimizer_wall_seconds`. Preserve missing timing as missing.

## Why the GMM experiment is useful

The [notebook](../notebooks/gmm_fourier_residual.ipynb) constructs Gaussian and
Gaussian-mixture populations with the **same exact mean and covariance**. Their
Gaussian reference is identical; their true scores generally are not. The
noise-marginalized mixture has an analytic score, allowing direct measurement of
the learned score error. Population moments remove estimation error as an
alternative explanation for a residual.

An 8 × 8 spatial grid supplies genuine Fourier coordinates. Vary spectral
heterogeneity at fixed total power, include a flat-spectrum control, and compare
both Gaussian arms against the analytic reference. Improved true-score error
would support residual learning beyond the reference; the experiment does not
assume a Fourier advantage in advance. Gaussian/GMM score analysis itself is
not presented as a novelty claim.
