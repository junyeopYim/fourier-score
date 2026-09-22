# Fourier Score

**Fixed Gaussian reference scores with frequency-normalized neural residuals.**

Inspired by the asymptotic Gaussianity of Fourier transforms, this project
constructs a fixed Gaussian reference from the training-data mean and
frequency-dependent power spectrum. The data distribution is **not assumed to
be Gaussian**. An analytic reference score accounts for its first and second
moments, and a neural residual learns the remaining structure. Its output scale
comes from the second moment of the Gaussian-subtracted denoising target.

The experiments preserve the backbone and DSM objective. They study whether
frequency-dependent covariance helps relative to a channelwise scalar covariance,
in both **pixel-space NCSN++** and **frozen-autoencoder latent diffusion**.

[Quick start](#quick-start) · [Method](#method) · [Experiments](#reproduce-the-experiments) ·
[GMM notebook](#understand-the-mechanism) · [Figures](#figures) ·
[Development](#development)

![An analytic Gaussian score plus a frequency-scaled neural residual, trained with the same DSM objective.](assets/loss_comparison.svg)

*Analytic illustration, not a measured learning curve.
[Figure source](scripts/plot_method.py) · [PNG](assets/loss_comparison.png)*

## Research status

| Experiment | Question or observation | Status |
|---|---|---|
| MNIST | The two Gaussian parameterizations gave similar FID in the exploratory run; no additional Fourier-covariance benefit was observed there. | Preliminary observation; no multi-seed claim. |
| CIFAR-10 | Does frequency-dependent covariance help relative to scalar covariance? | Three paired seeds × two Gaussian arms, 950K updates; results pending. |
| LSUN Churches | Is the construction also useful in the latent space of a fixed autoencoder? | Three paired seeds × two Gaussian arms, target 500K updates; results pending. |
| Matched-moment Gaussian / GMM | Can the residual learn structure beyond an exactly known Gaussian reference? | Reproducible synthetic mechanism experiment with an analytic true score. |

Public pretrained models and published FID values are external references, with
different training histories and potentially different metric protocols. They
are not paired baseline retraining runs. No image-quality improvement is claimed
before the controlled experiments are complete. The scalar–Fourier comparison
changes covariance in both the reference score and residual scale; it does not
separately identify the contribution of scaling alone.

## Quick start

Run from the repository root. Python 3.11 and the committed `uv.lock` specify the
environment. Optional dependencies are separate from the core pixel experiment.

```bash
uv sync --locked --python 3.11
uv run --locked python scripts/doctor.py --device cpu

# Three updates on synthetic data; no download or pretrained checkpoint needed.
uv run --locked python train.py -c configs/smoke.json --device cpu \
  --set name=readme_smoke
uv run --locked python sample.py -r saved/readme_smoke/last.pt \
  -o saved/readme_smoke_samples
uv run --locked python evaluate.py dsm -r saved/readme_smoke/last.pt \
  -o saved/readme_smoke_dsm.json
```

Use a new run name and output path when repeating the commands. Existing runs
are not overwritten. This verifies the pipeline, not generation quality.
For CUDA, run `scripts/doctor.py --device cuda` in the same environment before
training. On a supported Mac, use `--device mps`. Optional extras include
`ldm`, `datasets`, `metrics`, `notebooks`, `figures`, and `tensorboard`.

## Method

For $y=x+\sigma_t\epsilon$, $\epsilon\sim\mathcal N(0,I)$, all pixel arms minimize

$$
\mathcal L_{\mathrm{DSM}}=
\mathbb E_{x,t,\epsilon}\left\|\sigma_t s_\theta(y,t)+\epsilon\right\|^2.
$$

The baseline predicts the scaled score directly, $\sigma_t s_\theta=h_\theta$.
The Gaussian parameterizations instead use

$$
\sigma_t s_\theta(y,t)=\sigma_t s_G(y,t)
+\mathcal F^{-1}\!\left[b_t\mathcal F h_\theta(y,t)\right],
$$

$$
\widehat{s}_{G,k}=-\frac{\widehat y_k-\widehat\mu_k}{P_k+\sigma_t^2},
\qquad b_{t,k}=\sqrt{\frac{P_k}{P_k+\sigma_t^2}}.
$$

$\mu$ is the full mean image and $P_k$ the per-channel power of its centered,
orthonormal Fourier transform, estimated from **training data only**. The
reference has coefficient one and adds no learned parameters. Cross-channel
and cross-frequency covariance are not modeled by this reference; the neural
network still observes the entire image.

For $T=-\epsilon-\sigma_t s_G$, matching population moments give
$\mathbb E|\widehat T_k|^2=b_{t,k}^2$ without Gaussianity of the data. The scale
therefore normalizes the residual target's second moment. Estimated statistics
and numerical power floors make this a plug-in approximation in real experiments.
The original DSM objective retains the $b_{t,k}^2$ weighting in normalized
coordinates; unweighted MSE on $T/b_t$ would be a different objective.

| Parameterization | Gaussian reference | Residual scale |
|---|---|---|
| `score` | None | 1 |
| `scalar_gaussian` | Same full mean; frequency-averaged variance within each channel | Channelwise |
| `fourier_gaussian` | Same full mean; per-channel, per-frequency variance | Frequencywise |

`diffusion` is an additional epsilon-prediction sign convention. In the latent
path, $y=\alpha_t z+\sigma_t\epsilon$ replaces $\mu$ by $\alpha_t\mu$ and $P_k$
by $\alpha_t^2P_k$; the common adapter returns total epsilon. The first stage
stays frozen. The [method implementation](fourier_score/method.py) and
[GMM notebook](notebooks/gmm_fourier_residual.ipynb) give the corresponding
reference and residual calculations.

## Reproduce the experiments

### CIFAR-10: scalar versus Fourier covariance

```bash
# Prepare full-training-set statistics shared by both arms.
uv run --locked python scripts/prepare.py -c configs/cifar10_950k.json --download

# Inspect six commands: two arms × three paired seeds, 950K updates each.
bash scripts/reproduce_cifar10.sh 950k --device cuda --seeds 42 43 44 \
  --parameterizations scalar_gaussian fourier_gaussian \
  --set trainer.microbatch_size=32 --dry-run

# Remove --dry-run to launch those six jobs sequentially.
# Resume one interrupted run in the matching source checkout:
uv run --locked python train.py \
  -r saved/cifar10_950k_full_fourier_gaussian_s42/last.pt
```

The default runner also includes `score`; the explicit two-arm selection above
omits baseline retraining. Independent devices can run separate jobs. Training
uses single-device FP32 with microbatch accumulation. Keep the source revision,
locked environment, effective batch and microbatch identical across paired arms.
Replace `950k` with `50k` for the 45K/5K train/holdout pilot or `1m3` for
1.3M updates on the full training set. Keep their different FID reference splits
separate. Inspect config overrides with `train.py -c <config> --dry-run`.

### LSUN Churches: frozen first stage

```bash
uv sync --locked --extra ldm --extra datasets --extra metrics
uv run --locked --extra datasets python scripts/download_ldm.py \
  --model lsun_churches --with-data
uv run --locked --extra ldm python ldm.py prepare \
  -c configs/ldm/lsun_churches_l2.json --device cuda
uv run --locked --extra ldm python ldm.py compare \
  -c configs/ldm/lsun_churches_l2.json --device cuda --seeds 41 42 43 \
  --parameterizations scalar_gaussian fourier_gaussian --dry-run
```

Remove `--dry-run` to train after preparation. The target budget is 500K updates.
The `_l2` preset uses common L2/DSM; the original CompVis Churches preset uses
**L1**, so this is not an exact reproduction of its published training objective.
If a run stops early, report its actual update count and compare matched budgets.
Weights are stored under `pretrained/ldm/`, with latent caches under
`data/ldm_cache/`. Preserve the frozen first-stage weights for decoding.
`python ldm.py --help` lists preparation, training, sampling, evaluation and FID
commands; each subcommand has its own `--help`.

### Sampling and FID

```bash
uv run --locked python sample.py \
  -r saved/cifar10_950k_full_fourier_gaussian_s42/last.pt \
  -o saved/cifar10_fg_s42_samples50k --device cuda \
  --num-samples 50000 --batch-size 64
uv run --locked python scripts/export_real.py -c configs/cifar10_950k.json \
  --split train -o saved/cifar10_real
uv run --locked --extra metrics python evaluate.py fid \
  --real saved/cifar10_real/png --generated saved/cifar10_fg_s42_samples50k/png \
  --device cuda -o saved/cifar10_fg_s42_fid.json
```

Inference uses EMA. Keep real split, preprocessing, sample count, sampler,
actual NFE, seed, batch size and metric backend fixed. The local
`torch-fidelity` protocol differs from Score-SDE's original TensorFlow metrics.
Report variation across independent training seeds, and curves against both
updates and training time. To inspect available pretrained references, run
`python scripts/download_score_sde.py --list` or
`python scripts/download_ldm.py --list`. Score-SDE weights are converted with
`scripts/import_score_sde.py`; its `--help` describes inference-only import.

## Understand the mechanism

![Forward noising and reverse denoising of an analytic Gaussian mixture, with stochastic and probability-flow trajectories.](assets/stochastic_process.svg)

*Original analytic illustration using the exact mixture score; no learned model
is used. [PNG](assets/stochastic_process.png) · [PDF](assets/stochastic_process.pdf).*

The [GMM notebook](notebooks/gmm_fourier_residual.ipynb) compares a Gaussian and
a non-Gaussian mixture with identical **population** mean and covariance on an
8 × 8 spatial grid. Their reference scores match, but the mixture's exact score
contains a residual. It measures true-score error and varies spectral
heterogeneity at fixed average power, including the flat-spectrum control.

```bash
uv sync --locked --extra notebooks
uv run --locked --extra notebooks jupyter lab notebooks/gmm_fourier_residual.ipynb
```

The default smoke preset checks the complete experiment on CPU. Use the full
preset for a substantive mechanism study and report its actual settings; smoke
curves are not evidence of image-model performance. The figure scripts generate
original analytic illustrations independently of model training.
[Generate the figures](#figures).

## Figures

```bash
uv run --locked --extra figures python scripts/plot_method.py
uv run --locked --extra figures python scripts/plot_diagnostics.py
```

Both scripts run on CPU without datasets and write PNG, SVG and PDF to `assets/`.
Use `--output saved/figures` for a separate export. The diagnostic script also
writes synthetic source arrays and numerical checks; its default seed is
20260922 with 32,768 observations per distribution for the moment diagnostic.

| Figure | Interpretation | Export |
|---|---|---|
| Method | Shared DSM objective and Gaussian residual parameterization | [PNG](assets/loss_comparison.png) · [PDF](assets/loss_comparison.pdf) |
| Stochastic process | Exact evolving mixture density with numerical SDE/ODE paths | [PNG](assets/stochastic_process.png) · [PDF](assets/stochastic_process.pdf) |
| Gaussian / GMM residual | Same population moments, different exact scores | [PNG](assets/gaussian_residual.png) · [PDF](assets/gaussian_residual.pdf) |
| Frequency scaling | Analytic residual-target moments checked by Monte Carlo | [PNG](assets/frequency_scaling.png) · [PDF](assets/frequency_scaling.pdf) |

These illustrations use known distributions rather than learned models. The
one-dimensional process figure uses $\sigma_t^2=4t$ and starts the reverse SDE
from the exact finite-time noisy mixture; the dashed ODE curves are deterministic
trajectories viewed in both directions. The frequency figure uses an 8 × 8 grid
with matched population covariance. Its error bars are two Monte Carlo standard
errors, not variation across training seeds. Target second moments are distinct
from the optimal conditional residual's variance.

[Source settings and checks](assets/diagnostics.json) accompany the
[synthetic arrays](assets/diagnostics_data.npz), which can be read using
`numpy.load(path, allow_pickle=False)`. The GMM notebook separately exports
measured learning curves. Keep smoke outputs distinct from completed experiments;
image-model result figures require completed checkpoint evaluations.

## Repository layout

```text
train.py / sample.py / evaluate.py   Pixel training, inference and metrics
ldm.py                             Native latent pipeline
configs/                           Versioned experimental protocols
fourier_score/                     Method, statistics, trainers and samplers
  backbones/                       Attributed NCSN++ implementation
  ldm/                             Frozen-first-stage latent implementation
notebooks/                         Runnable mechanism experiments
scripts/                           Preparation, diagnostics and figure generation
tests/                             Numerical and end-to-end regression checks
assets/                            Public figures and synthetic source arrays
```

Runs save resolved config, environment/source fingerprint, initial backbone
fingerprint, metrics, optimizer/RNG/data-cursor checkpoints, and EMA snapshots.
Sampling saves checkpoint identity and settings alongside images.
`docs/` is reserved for private local research notes and is ignored by Git.
`data/`, `saved/`, `pretrained/`, `verification/`, `local/` and `notes/` are local,
ignored artifact locations. Share curated results with their manifests, rather
than raw machine-specific orchestration logs.

```bash
uv run --locked --all-extras python -m pytest -q
uv run --locked --extra notebooks python scripts/check_notebooks.py
```

The CI workflow checks numerical behavior and notebook execution on CPU. See
the [development notes below](#development) for source navigation and debugging.

## Development

Start with `fourier_score/method.py` for the Gaussian adapter,
`statistics.py` for training-only moments, and `model.py` / `loss.py` for the
shared DSM path. Pixel training and samplers are in `training.py` and
`diffusion.py`; native latent equivalents live under `fourier_score/ldm/`.

```bash
uv sync --locked --all-extras
uv run --no-sync python scripts/doctor.py --device cpu
uv run --no-sync python scripts/check_notebooks.py --execute
```

Use `--dry-run` to inspect resolved configs before allocating a long experiment.
For memory issues, measure a separate short run with smaller microbatches and
keep that setting fixed across paired arms. Resume with the saved config and
matching source/runtime; an existing run directory is not a destination for a
new protocol. Keep active training checkouts at their original revision while
editing in another checkout. Public notebooks, figure scripts and tests run
without the private notes directory.

## References and attribution

- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), ICLR 2021. [Official code](https://github.com/yang-song/score_sde_pytorch).
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022. [Official code](https://github.com/CompVis/latent-diffusion).
- Karras et al., [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364), NeurIPS 2022. [Official code](https://github.com/NVlabs/edm).
- [PyTorch Template](https://github.com/victoresque/pytorch-template) inspired the config-driven entry points, checkpointing and separation of concerns. The project uses a single trainer per domain rather than copying classifier abstractions.

See [LICENSE](LICENSE), [NOTICE](NOTICE) and the
[native LDM source manifest](fourier_score/ldm/upstream/PROVENANCE.json).
Downloaded third-party weights and datasets retain their respective terms.
