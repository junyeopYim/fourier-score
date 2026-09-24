# Fourier Score

**Fixed Gaussian reference scores with frequency-normalized neural residuals.**

Inspired by the asymptotic Gaussianity of Fourier transforms, this project
constructs a fixed Gaussian reference from the training-data mean and
frequency-dependent power spectrum. The analytic reference score captures these
statistics, while a neural residual models the remaining structure
and dependencies. The residual output scale follows from the second moment of
the Gaussian-subtracted denoising target.

By default, the experiments preserve the backbone and DSM objective. They study
whether frequency-dependent covariance helps relative to a channelwise scalar covariance,
in both **pixel-space NCSN++** and **frozen-autoencoder latent diffusion**.

[Quick start](#quick-start) · [Method](#method) · [Experiments](#reproduce-the-experiments) ·
[GMM notebook](#understand-the-mechanism) · [Loss comparison](#gmm-loss-comparison) ·
[Gated comparison](#gmm-gated-comparison) · [Plateau experiment](#gmm-plateau-comparison) · [Figures](#figures) ·
[Development](#development)

![Method schematic: direct score prediction and a fixed Gaussian score plus a spectrally scaled neural residual share the DSM objective; the analytic residual scale is shown below.](assets/loss_comparison.svg)

*Score parameterizations under a shared DSM objective. (a) Direct scaled-score
prediction. (b) A fixed Gaussian reference plus a spectrally scaled neural
residual, using the same backbone architecture. (c) The analytic adapter and its
residual scale, determined by fixed training-data moments.
[Figure source](scripts/plot_method.py) · [PNG](assets/loss_comparison.png) ·
[PDF](assets/loss_comparison.pdf)*

## Research status

| Experiment | Question or observation | Status |
|---|---|---|
| MNIST | The two Gaussian parameterizations gave similar FID in the exploratory run. | Preliminary single-run observation. |
| CIFAR-10 | Does frequency-dependent covariance help relative to scalar covariance? | Three paired seeds × two Gaussian arms, 950K updates; results pending. |
| LSUN Churches | Is the construction also useful in the latent space of a fixed autoencoder? | Three paired seeds × two Gaussian arms, target 500K updates; results pending. |
| Matched-moment Gaussian / GMM | Can the residual learn structure beyond an exactly known Gaussian reference? | Reproducible synthetic mechanism experiment with an analytic true score. |
| GMM loss comparison | At spectral heterogeneity $\lambda=1$, normalized residual loss reduced Fourier true-score error by 33.4% relative to Fourier DSM; Scalar DSM remained best. | 45 runs, three paired seeds, 5,000 updates; [results and protocol](#gmm-loss-comparison). |
| GMM gated comparison | At $\lambda=1$, Gated Fourier reduced error by 25.8% versus Scalar DSM, with a high-noise tradeoff. | 99 runs, validation-selected gate, independent test banks; [results](#gmm-gated-comparison). |
| GMM plateau gate | Forcing $g=1$ above $\sigma=1$ increased Fourier error by 12.6% overall and 25.8% at high noise versus the sigmoid gate. | Fixed transition $[0.8,1.0]$, 18 new runs, three paired seeds; [results](#gmm-plateau-comparison). |

The scalar–Fourier comparison measures the combined effect of frequency-dependent
covariance in the reference score and residual scale. Pretrained models and
published FID values serve as external reference points.

## Quick start

Run from the repository root. Python 3.11 and the committed `uv.lock` specify the
environment. Optional dependencies are separate from the core pixel experiment.

```bash
uv sync --locked --python 3.11
uv run --locked python scripts/doctor.py --device cpu

# Train, sample and evaluate a small synthetic example.
uv run --locked python train.py -c configs/smoke.json --device cpu \
  --set name=readme_smoke
uv run --locked python sample.py -r saved/readme_smoke/last.pt \
  -o saved/readme_smoke_samples
uv run --locked python evaluate.py dsm -r saved/readme_smoke/last.pt \
  -o saved/readme_smoke_dsm.json
```

This example runs three optimizer updates. Use a new run name and output path
when repeating it.
For CUDA, run `scripts/doctor.py --device cuda` in the same environment before
training. On a supported Mac, use `--device mps`. Optional extras include
`ldm`, `datasets`, `metrics`, `notebooks`, `figures`, and `tensorboard`.

## Method

For $y=x+\sigma_t\epsilon$, $\epsilon\sim\mathcal N(0,I)$, all pixel arms default to
`loss.objective=dsm` and minimize

$$
\mathcal L_{\mathrm{DSM}}=
\mathbb E_{x,t,\epsilon}\left\|\sigma_t s_\theta(y,t)+\epsilon\right\|^2.
$$

The baseline predicts the scaled score directly, $\sigma_t s_\theta=h_\theta$.
The Gaussian parameterizations use

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
reference has coefficient one and a fixed diagonal covariance in the Fourier
basis. The neural network observes the entire image and learns dependencies
across channels and frequencies.

For $T=-\epsilon-\sigma_t s_G$, matching population moments give
$\mathbb E|\widehat T_k|^2=b_{t,k}^2$ for Gaussian and non-Gaussian data. The scale
therefore normalizes the residual target's second moment. Estimated statistics
and numerical power floors make this a plug-in approximation in real experiments.
The original DSM objective retains the $b_{t,k}^2$ weighting in normalized
residual coordinates.

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

### Normalized residual loss (VE)

Set `loss.objective=normalized_residual` with `scalar_gaussian` or
`fourier_gaussian` to regress the raw backbone output directly against

$$
\tau_k=\frac{\sigma_t\mathcal F(x-\mu)_k-P_k\mathcal F\epsilon_k}
{\sqrt{P_k(P_k+\sigma_t^2)}},\qquad
\mathcal L_{\mathrm{norm}}=\mathbb E\|h_\theta-\mathcal F^{-1}\tau\|^2.
$$

The scalar control uses the same formula with the channelwise average power.
The target is computed directly from clean data to avoid cancellation at large
noise levels. Both objectives use the configured `mean` or `half_sum` reduction.
The score adapter and sampler still apply the same residual scale $b_k$.
This objective currently supports pixel-space VE only.

```bash
uv run --locked python scripts/run_comparison.py -c configs/mnist.json \
  --parameterizations scalar_gaussian fourier_gaussian --seeds 0 \
  --set loss.objective=normalized_residual \
  --set trainer.iterations=30000 --dry-run
```

Use `loss.objective=dsm` for the paired DSM controls. Normalized runs append
`_normalized_residual` to the resolved run name, and record the objective in
training logs and checkpoint configs. Changing objectives is rejected on resume;
older configs without this field mean `dsm`. Compare common validation DSM,
noise/frequency diagnostics and matched-sampler FID, rather than comparing the
training loss values across objectives. `grad_norm_before_clip` is also logged.

### Noise-gated Gaussian residuals (VE)

An optional fixed gate smoothly connects direct score prediction at low noise
to the Gaussian residual parameterization at high noise:

$$
g(\sigma)=\operatorname{sigmoid}\!\left[p(\log\sigma-\log\sigma_c)\right],
\qquad c_{\sigma,k}^2=(1-g)^2+g(2-g)\frac{P_k}{P_k+\sigma^2},
$$

$$
\sigma s_\theta=g\sigma s_G+\mathcal F^{-1}[c\mathcal Fh_\theta],
\qquad
\tau_{g,k}=\frac{g\sigma\mathcal F(x-\mu)_k-
[P_k+(1-g)\sigma^2]\mathcal F\epsilon_k}{(P_k+\sigma^2)c_{\sigma,k}}.
$$

With `loss.objective=normalized_residual`, training minimizes
$\mathbb E\|h_\theta-\mathcal F^{-1}\tau_g\|^2$.
At $g=0$ this is Score/DSM; at $g=1$ it recovers the original normalized residual
objective. The second-moment identity requires matched population mean and
power, but not Gaussian data. Scalar covariance normalizes the **channel-average**
target second moment; Fourier covariance normalizes each frequency. With estimated
or floored statistics, these are plug-in approximations.

```bash
uv run --locked python train.py -c configs/smoke.json \
  --set loss.objective=normalized_residual \
  --set fourier.gate.mode=log_sigma \
  --set fourier.gate.sigma_switch=1.0 \
  --set fourier.gate.sharpness=4.0 --dry-run
```

`fourier.gate.mode=none` is the default and preserves the original $g=1$ path.
`constant` with `fourier.gate.value=0` or `1` provides endpoint controls.
The gate is part of the model adapter, so training and sampling use the same
settings. Active gates require a VE Gaussian parameterization. They can also
be trained with DSM as a separate ablation. Run names include the gate settings;
resume and inference reject changes to the trained gate. Older configs without
gate fields retain their original behavior and configuration signatures.

The implementation computes $g$ and $1-g$ from opposite sigmoid arguments,
and constructs the target directly from clean data and noise. This avoids
subtracting nearly equal values when the reference dominates at high noise.

An optional `log_sigma_plateau` gate retains this sigmoid below a specified
noise level and reaches **exactly one** at a finite upper boundary:

$$
z=\operatorname{clip}\!\left(\frac{\log\sigma-\log\sigma_L}
{\log\sigma_H-\log\sigma_L},0,1\right),\qquad
w=z^2(3-2z),\qquad g_{\rm plateau}=g+w(1-g).
$$

Both the residual scale and normalized target use this same modified gate.
For $\sigma\leq\sigma_L$, the adapter and target retain the sigmoid calculation;
for $\sigma\geq\sigma_H$, they recover the original $g=1$ parameterization.
This preserves the **equations**, not the trained high-noise predictions of a
different checkpoint: all noise levels still share the newly trained backbone.
There is no teacher, ensemble or added model capacity. Retrain when changing
gate modes; do not substitute the plateau into a trained sigmoid checkpoint.

```bash
uv run --locked python train.py -c configs/smoke.json \
  --set loss.objective=normalized_residual \
  --set fourier.gate.mode=log_sigma_plateau \
  --set fourier.gate.sigma_switch=1.5 \
  --set fourier.gate.sharpness=4.0 \
  --set fourier.gate.sigma_lo=0.8 --set fourier.gate.sigma_hi=1.0 --dry-run
```

The implementation computes $1-g_{\rm plateau}=(1-z)^2(1+2z)(1-g)$ directly
and enforces the endpoints using comparisons of $\sigma$, avoiding subtraction
cancellation near the plateau. Bounds must satisfy $0<\sigma_L<\sigma_H$;
they are included in experiment names and resume signatures for this mode.
The added bounds do not change existing sigmoid-gate configuration signatures.

Run the controlled GMM comparison with:

```bash
uv run --locked --extra figures python scripts/run_gmm_comparison.py \
  --output saved/gmm_gated_example --workers 3
```

This trains five existing baselines and six gate candidates: Scalar/Fourier
with $\sigma_c\in\{0.5,1,1.5\}$ and $p=4$, at three spectra and three paired
seeds, for **99 runs of 5,000 updates**. One shared switch is chosen by the mean
final EMA **validation** error across both covariances, all spectra and seeds.
Only then are the five baselines and two selected gated arms evaluated on a
new, shared test bank (63 evaluations). The runner records the selection before
constructing the test bank. It uses the notebook's shared implementation in
[`fourier_score/gmm.py`](fourier_score/gmm.py), without rewriting notebook cells.

Use `--preset smoke --seeds 42` for a short pipeline check. `--bank-version`
controls independent evaluation banks; changing only the output directory does
not change them. Matching interrupted runs resume from saved optimizer, EMA and
data RNG states. Tables and PNG/SVG/PDF figures are saved in `<output>/report`,
or a directory supplied with `--report`.
`--workers` schedules individual runs, each using one CPU thread; use, for
example, `--workers 12` to occupy more cores without changing numerical training
settings. The worker count can change on resume. Launcher invocations are
recorded in `execution_history.json`, while numerical source and protocol
checks continue to guard checkpoint compatibility.

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

The command selects the two Gaussian arms. Add `score` to include the direct
score baseline. Independent devices can run separate jobs. Training
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
The `_l2` preset applies **L2/DSM** to both arms. The original CompVis Churches
preset uses **L1**.
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

To re-evaluate existing CIFAR samples with the original TF-Hub/TF-GAN backend,
use [`scripts/evaluate_score_sde.py`](scripts/evaluate_score_sde.py) in a separate
TensorFlow environment. See [setup and protocol](scripts/README-score-sde-eval.md).
This reads exactly 50,000 saved images and the official CIFAR reference statistics;
it does not generate new samples.

## Understand the mechanism

![Forward noising and reverse denoising of an analytic Gaussian mixture, with stochastic and probability-flow trajectories.](assets/stochastic_process.svg)

*Forward and reverse dynamics using the exact Gaussian-mixture score. [PNG](assets/stochastic_process.png) · [PDF](assets/stochastic_process.pdf).*

The [GMM notebook](notebooks/gmm_fourier_residual.ipynb) compares a Gaussian and
a non-Gaussian mixture with identical **population** mean and covariance on an
8 × 8 spatial grid. Their reference scores match, but the mixture's exact score
contains a residual. It measures true-score error and varies spectral
heterogeneity at fixed average power, including the flat-spectrum control.

```bash
uv sync --locked --extra notebooks
uv run --locked --extra notebooks jupyter lab notebooks/gmm_fourier_residual.ipynb
```

The default `smoke` preset runs a small CPU example. `pilot` and `experiment`
provide the 8 × 8 study with larger budgets and repeated seeds. The notebook
exports learning curves, spectral diagnostics and per-seed result tables.
[Generate the figures](#figures).

### GMM loss comparison

The normalized residual objective improves Fourier's true-score error in this
study, with its largest gain at the strongest spectral heterogeneity. **At
$\lambda=1$, Fourier error falls by 33.4% relative to Fourier DSM, but remains
14.0% above Scalar DSM.** Applying the same objective change to Scalar increases
its error by 5.2% in this condition.

The table reports the final EMA model's noise-scaled true-score MSE,
$\mathbb E[\sigma^2\|s_\theta(y,\sigma)-s_{\mathrm{exact}}(y,\sigma)\|^2/64]$.
Lower is better. Values are mean ± sample standard deviation across three
training seeds, with the best mean in each column in bold.

| Parameterization / objective | $\lambda=0$ (flat) | $\lambda=0.5$ | $\lambda=1$ |
|---|---:|---:|---:|
| Score / DSM | **0.05290 ± 0.00022** | **0.05873 ± 0.00068** | 0.07764 ± 0.00110 |
| Scalar / DSM | 0.13277 ± 0.00492 | 0.08617 ± 0.00335 | **0.07490 ± 0.00073** |
| Fourier / DSM | 0.13277 ± 0.00492 | 0.12872 ± 0.00317 | 0.12821 ± 0.00664 |
| Scalar / normalized residual | 0.12484 ± 0.00623 | 0.09286 ± 0.00185 | 0.07880 ± 0.00182 |
| Fourier / normalized residual | 0.12484 ± 0.00623 | 0.11998 ± 0.00460 | 0.08539 ± 0.00445 |

These runs, completed on 2026-09-24, use a GMM-only extension of the notebook's
experiment preset with two additional `loss.objective=normalized_residual`
arms. The notebook itself defaults to the three DSM arms. The comparison uses:

- An 8 × 8, 64-dimensional, equal-weight 128-component GMM with mean zero,
  average power one, $\rho=0.85$, and fixed geometry seed 31415. Only spectral
  heterogeneity changes across $\lambda\in\{0,0.5,1\}$.
- The same 102,016-parameter MLP (width 192, depth 3), batch size 128,
  Adam learning rate 0.001, EMA decay 0.99, and 5,000 updates for every arm.
  Training seeds 42, 43, and 44 share initial backbone weights and training
  sample/noise streams across methods. Training uses CPU float32; the oracle
  and error aggregation use float64. No run activated gradient clipping.
- VE noise with $\sigma\in[0.1,3]$, sampled log-uniformly during training.
  Evaluation uses nine log-sigma midpoint bins with 2,048 independent test
  observations each: 18,432 per spectrum, shared across methods and seeds.
  The final 5,000-step EMA is evaluated without test-based checkpoint selection.

The oracle is the analytic **joint GMM score**; the expectation is estimated
from the test bank. An independent float64 autograd check of the dense mixture
log-density agreed within $1.9\times10^{-13}$ maximum absolute score error.
The three seeds measure training variation for one fixed GMM geometry and test
bank. Conclusions are limited to this geometry and 5,000-update budget;
long-run convergence and MNIST performance require separate experiments.

At $\lambda=1$, removing the $\sigma^2$ evaluation weighting also shows a Fourier
improvement: unweighted true-score MSE falls from 1.51425 to 1.02619 (32.2%).
Scalar's unweighted error rises from 0.62083 to 0.78444 (26.4%). At $\lambda=0$,
Scalar and Fourier agree to within $4\times10^{-7}$ in mean noise-scaled error
under either objective, as expected for the flat-spectrum control.

![GMM loss comparison: final true-score error across spectral heterogeneity, validation curves, and error by noise level.](assets/gmm_loss_comparison/gmm_loss_comparison.svg)

*Mean ± one training-seed standard deviation; the last two panels use
$\lambda=1$. [PNG](assets/gmm_loss_comparison/gmm_loss_comparison.png) ·
[PDF](assets/gmm_loss_comparison/gmm_loss_comparison.pdf).*

Archived measurements: [summary](assets/gmm_loss_comparison/summary.csv) ·
[per-seed results](assets/gmm_loss_comparison/per_seed.csv) ·
[paired comparisons](assets/gmm_loss_comparison/paired_comparisons.csv) ·
[noise levels](assets/gmm_loss_comparison/noise_resolved.csv) ·
[frequency bands](assets/gmm_loss_comparison/frequency_resolved.csv) ·
[validation curves](assets/gmm_loss_comparison/validation_curves.csv).
The [run configuration and provenance](assets/gmm_loss_comparison/summary.json)
record source commit `f8df48f`, environment, source hashes, and successful checks
of paired initialization, data streams, and test banks.
[Oracle validation](assets/gmm_loss_comparison/oracle_validation.json) records
the independent numerical check.

## GMM gated comparison

The noise-gated adapter improves the overall error in all three spectra in this
5,000-update experiment. At $\lambda=1$, **Gated Fourier reduces test error by
25.8% versus the best existing baseline (Scalar/DSM), 34.7% versus Fourier
normalized residual, and 5.2% versus Gated Scalar**. Each of these paired
comparisons improves in all three training seeds.

The same MLP, Adam settings, online training streams and final EMA budget as the
[previous comparison](#gmm-loss-comparison) are used. All 45 existing baseline
runs reproduce their archived final EMA weights **exactly**. The following
numbers use a **new test bank**, so baseline errors differ slightly from the
previous table. The metric is noise-scaled true-score MSE; values are mean ±
sample standard deviation across seeds 42, 43 and 44.

| Parameterization / objective | $\lambda=0$ | $\lambda=0.5$ | $\lambda=1$ |
|---|---:|---:|---:|
| Score / DSM | 0.05328 ± 0.00020 | 0.05961 ± 0.00059 | 0.07764 ± 0.00121 |
| Scalar / DSM | 0.13302 ± 0.00511 | 0.08625 ± 0.00306 | 0.07455 ± 0.00077 |
| Fourier / DSM | 0.13302 ± 0.00511 | 0.12927 ± 0.00347 | 0.12789 ± 0.00659 |
| Scalar / normalized residual | 0.12523 ± 0.00593 | 0.09297 ± 0.00184 | 0.07834 ± 0.00159 |
| Fourier / normalized residual | 0.12523 ± 0.00593 | 0.12037 ± 0.00465 | 0.08466 ± 0.00380 |
| Gated Scalar / normalized residual | **0.03904 ± 0.00009** | 0.04295 ± 0.00040 | 0.05834 ± 0.00069 |
| Gated Fourier / normalized residual | **0.03904 ± 0.00009** | **0.04286 ± 0.00037** | **0.05530 ± 0.00072** |

One common switch was selected on validation across both covariances, all three
spectra and all three seeds. With sharpness $p=4$, the mean validation errors
for $\sigma_c=0.5,1.0,1.5$ were **0.08400, 0.05041, 0.04617**, respectively.
The selected **$\sigma_c=1.5$** is the best of these three candidates. It is
shared by Gated Scalar and Gated Fourier for every spectrum and seed.
Selection used final-step validation only; the test bank was constructed after
recording the choice. There were 99 training runs and 63 final test evaluations,
with 18,432 independent test observations per spectrum, shared across all
methods and training seeds. No selected run activated gradient clipping.

**The high-noise tradeoff remains.** At $\lambda=1$, grouping the nine log-noise
bins into low ($\sigma<0.3$), middle ($0.3\leq\sigma<1$), and high
($\sigma\geq1$) regions gives:

| Noise region | Fourier normalized residual | Gated Fourier | Relative change |
|---|---:|---:|---:|
| Low | 0.07787 | 0.03413 | −56.2% |
| Middle | 0.12633 | 0.07203 | −43.0% |
| High | 0.04979 | 0.05974 | +20.0% |

Thus the gate substantially improves low/middle noise, but does not fully retain
the ungated normalized model's high-noise accuracy. At $\lambda=0$ the two gated
covariances agree numerically; at $\lambda=0.5$ their difference is small.
These conclusions concern one fixed GMM geometry and 5,000 updates; the experiment
does not establish long-run or image-generation performance.

![Gated GMM comparison: final error across spectra and noise-resolved error for lambda one.](assets/gmm_gated_comparison/gmm_gated_comparison.svg)

*Mean ± one training-seed standard deviation. [PNG](assets/gmm_gated_comparison/gmm_gated_comparison.png) ·
[PDF](assets/gmm_gated_comparison/gmm_gated_comparison.pdf).*

Reproduce with:

```bash
uv run --locked --extra figures python scripts/run_gmm_comparison.py \
  --output saved/gmm_gated_reproduction --workers 12
```

Archived [summary and protocol](assets/gmm_gated_comparison/summary.json) ·
[per-seed results](assets/gmm_gated_comparison/per_seed.csv) ·
[paired comparisons](assets/gmm_gated_comparison/paired_comparisons.csv) ·
[validation selection](assets/gmm_gated_comparison/validation_selection.csv) ·
[validation curves](assets/gmm_gated_comparison/validation_curves.csv) ·
[noise levels](assets/gmm_gated_comparison/noise_resolved.csv) ·
[frequency bands](assets/gmm_gated_comparison/frequency_resolved.csv).
[Verification](assets/gmm_gated_comparison/verification.json) records passing tests,
notebook execution, CPU diagnostics and unchanged numerical source hashes.
[Baseline reproduction](assets/gmm_gated_comparison/baseline_reproduction.json)
checks all 45 archived final EMA hashes. An independent
[dense log-density autograd check](assets/gmm_gated_comparison/oracle_validation.json)
agrees with the 64-dimensional oracle within $2.7\times10^{-14}$ maximum absolute
score error. [Execution history](assets/gmm_gated_comparison/execution_history.json)
records the launcher used after increasing CPU concurrency to 12 workers.

## GMM plateau comparison

**The fixed plateau did not improve the sigmoid gate in this experiment.**
At $\lambda=1$, Plateau Fourier's overall test error increased by **12.6%**
relative to Gated Fourier, and its high-noise error increased by **25.8%**.
The overall regression occurs in all three paired seeds at every spectrum.
Plateau Fourier still beats Scalar/DSM by 16.4% at $\lambda=1$, but does not
retain the better result of the existing sigmoid gate.

This experiment fixes $\sigma_c=1.5$, $p=4$, $\sigma_L=0.8$ and $\sigma_H=1.0$
**before training and evaluation**, without searching the bounds or selecting
on test results. The backbone, parameter count, initialization, online training
streams, Adam settings, EMA and 5,000-update budget match the gated comparison.
Only the gate changes; its reference, residual scale and normalized target are
updated together. No teacher loss or additional model is used.

There are **18 new training runs** (two covariances × three spectra × three
seeds). The other 63 checkpoints are reused after checking their configurations,
statistics, initial/final EMA hashes and exact reproduction of every archived
validation noise-bin metric. All 81 models are evaluated on a **new common test
bank** (`gmm-plateau-v1`): nine log-noise midpoint bins, 2,048 observations per
bin, shared across methods and training seeds. These new observations explain
the small differences from earlier baseline tables. The 12-worker launcher
runs each job with one CPU thread. No run activated gradient clipping.

Values below are noise-scaled true-score MSE, mean ± one sample standard
deviation over training seeds 42, 43 and 44; lower is better.

| Parameterization / objective | $\lambda=0$ | $\lambda=0.5$ | $\lambda=1$ |
|---|---:|---:|---:|
| Score / DSM | 0.05316 ± 0.00018 | 0.05947 ± 0.00057 | 0.07801 ± 0.00101 |
| Scalar / DSM | 0.13331 ± 0.00487 | 0.08684 ± 0.00332 | 0.07474 ± 0.00078 |
| Fourier / DSM | 0.13332 ± 0.00487 | 0.12939 ± 0.00359 | 0.12823 ± 0.00629 |
| Scalar / normalized residual | 0.12541 ± 0.00638 | 0.09348 ± 0.00203 | 0.07866 ± 0.00159 |
| Fourier / normalized residual | 0.12541 ± 0.00638 | 0.12045 ± 0.00475 | 0.08509 ± 0.00387 |
| Gated Scalar / normalized residual | **0.03901 ± 0.00013** | 0.04296 ± 0.00044 | 0.05853 ± 0.00054 |
| Gated Fourier / normalized residual | **0.03901 ± 0.00013** | **0.04291 ± 0.00044** | **0.05553 ± 0.00064** |
| Plateau Scalar / normalized residual | 0.06117 ± 0.00062 | 0.05649 ± 0.00092 | 0.06914 ± 0.00138 |
| Plateau Fourier / normalized residual | 0.06117 ± 0.00062 | 0.05993 ± 0.00064 | 0.06251 ± 0.00020 |

At $\lambda=1$, the low/middle/high regions use the same boundaries as before
($\sigma<0.3$, $0.3\leq\sigma<1$, and $\sigma\geq1$):

| Noise region | Fourier normalized | Gated Fourier | Plateau Fourier | Plateau vs gated |
|---|---:|---:|---:|---:|
| Low | 0.07816 | 0.03455 | **0.02928** | −15.3% |
| Middle | 0.12659 | **0.07197** | 0.08272 | +14.9% |
| High | **0.05053** | 0.06005 | 0.07554 | +25.8% |

The low-noise improvement is outweighed by middle/high-noise regressions.
Despite exactly reaching $g=1$, the plateau's high-noise error is **49.5%**
above the original Fourier-normalized model. A shared, newly trained backbone
does not recover the original model's predictions merely by restoring its
high-noise adapter equations.

![Plateau GMM comparison across spectra and noise levels.](assets/gmm_plateau_comparison/gmm_plateau_comparison.svg)

A separate diagnostic evaluates 15 noise levels around the transition at
$\lambda=1$, with 1,024 fresh observations per level. These points are **not**
pooled into the primary nine-bin average. At $\sigma=1$, the test error is
**0.15923** for Plateau Fourier, versus **0.10203** for Gated Fourier and
**0.10703** for Fourier normalized. The increased error near the transition is
visible in every seed. A narrow transition or shared-backbone optimization may
contribute; this experiment does not isolate those causes, optimize the bounds,
or establish sample/image quality. It uses one fixed GMM geometry and budget.

![Plateau gate and dense transition diagnostic.](assets/gmm_plateau_comparison/gmm_plateau_transition.svg)

Reproduce all 81 runs from scratch:

```bash
uv run --locked --extra figures python scripts/run_gmm_plateau.py \
  --output saved/gmm_plateau_reproduction --workers 12
```

If the previous comparison checkpoints are available, add
`--reuse-baselines saved/gmm_gated_20260924` to train only the 18 plateau runs.
Reuse is read-only and requires exact validation reproduction. Use
`--preset smoke --seeds 42` for a short pipeline check. The bounds are configurable
with `--sigma-lo` and `--sigma-hi`; use a new output directory and an independent
test-bank version for subsequent model-selection experiments.

Archived [protocol](assets/gmm_plateau_comparison/protocol.json) ·
[summary](assets/gmm_plateau_comparison/summary.json) ·
[paired comparisons](assets/gmm_plateau_comparison/paired_comparisons.csv) ·
[per-seed results](assets/gmm_plateau_comparison/per_seed.csv) ·
[noise regions](assets/gmm_plateau_comparison/noise_regions.csv) ·
[transition diagnostics](assets/gmm_plateau_comparison/transition_resolved.csv) ·
[checkpoint audit](assets/gmm_plateau_comparison/checkpoint_audit.json) ·
[verification](assets/gmm_plateau_comparison/verification.json).
Figures: [comparison PNG](assets/gmm_plateau_comparison/gmm_plateau_comparison.png) /
[PDF](assets/gmm_plateau_comparison/gmm_plateau_comparison.pdf),
[transition PNG](assets/gmm_plateau_comparison/gmm_plateau_transition.png) /
[PDF](assets/gmm_plateau_comparison/gmm_plateau_transition.pdf).

## Figures

```bash
uv run --locked --extra figures python scripts/plot_method.py
uv run --locked --extra figures python scripts/plot_diagnostics.py
```

Both scripts generate analytic examples on CPU and write PNG, SVG and PDF to
`assets/`.
Use `--output saved/figures` for a separate export. The diagnostic script also
writes synthetic source arrays and numerical checks; its default seed is
20260922 with 32,768 observations per distribution for the moment diagnostic.

| Figure | Interpretation | Export |
|---|---|---|
| Method | Shared DSM objective and Gaussian residual parameterization | [PNG](assets/loss_comparison.png) · [PDF](assets/loss_comparison.pdf) |
| Stochastic process | Exact evolving mixture density with numerical SDE/ODE paths | [PNG](assets/stochastic_process.png) · [PDF](assets/stochastic_process.pdf) |
| Gaussian / GMM residual | Same population moments, different exact scores | [PNG](assets/gaussian_residual.png) · [PDF](assets/gaussian_residual.pdf) |
| Frequency scaling | Analytic residual-target moments checked by Monte Carlo | [PNG](assets/frequency_scaling.png) · [PDF](assets/frequency_scaling.pdf) |
| GMM loss comparison | Trained DSM and normalized residual objectives against the analytic joint score | [PNG](assets/gmm_loss_comparison/gmm_loss_comparison.png) · [PDF](assets/gmm_loss_comparison/gmm_loss_comparison.pdf) |

The one-dimensional process figure uses $\sigma_t^2=4t$ and starts the reverse SDE
from the exact finite-time noisy mixture; the dashed ODE curves are deterministic
trajectories viewed in both directions. The frequency figure uses an 8 × 8 grid
with matched population covariance. Its error bars are two Monte Carlo standard
errors. The frequency curves describe second moments of the noisy residual
regression target.

The [synthetic arrays](assets/diagnostics_data.npz) can be read with
`numpy.load(path, allow_pickle=False)` and reused for custom figure layouts.

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
assets/                            Public figures, synthetic arrays and result tables
```

Each run saves its configuration, metrics, training checkpoints and EMA
snapshots under `saved/`. Generated images are stored with their sampling
settings. `docs/` holds local research notes and is excluded from Git.

```bash
uv run --locked --all-extras python -m pytest -q
uv run --locked --extra notebooks python scripts/check_notebooks.py
```

The CI workflow checks numerical behavior and notebook execution on CPU. See
the [development notes below](#development) for source navigation and debugging.

## Development

Start with `fourier_score/method.py` for the Gaussian adapter,
`statistics.py` for training-only moments, and `model.py` / `loss.py` for the
pixel objectives. Pixel training and samplers are in `training.py` and
`diffusion.py`; native latent equivalents live under `fourier_score/ldm/`.

```bash
uv sync --locked --all-extras
uv run --no-sync python scripts/doctor.py --device cpu
uv run --no-sync python scripts/check_notebooks.py --execute
```

Use `--dry-run` to inspect configs and `--help` for each command's options.
Reduce microbatch size to fit device memory while keeping the effective batch
fixed. Resume a run with `train.py -r <checkpoint>` in its original environment.
Give each experiment a distinct run name.

## References and attribution

- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), ICLR 2021. [Official code](https://github.com/yang-song/score_sde_pytorch).
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022. [Official code](https://github.com/CompVis/latent-diffusion).
- Karras et al., [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364), NeurIPS 2022. [Official code](https://github.com/NVlabs/edm).
- [PyTorch Template](https://github.com/victoresque/pytorch-template) inspired the config-driven entry points, checkpointing and separation of concerns. Each domain has its own trainer.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for source attribution.
Downloaded third-party weights and datasets retain their respective terms.
