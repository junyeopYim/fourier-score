# GMM linear and tanh gates

[Back to README](../../README.md#gmm-linear-and-tanh-gates) · [All reports](../README.md) · Provenance: [epoch 0](../provenance.md)

**Both new gates improve the sigmoid at $\lambda=1$, while the spectral gate
remains best overall.** Linear reduces Fourier error by **2.27%** and tanh by
**2.33%** relative to the sigmoid, in all three paired seeds. Relative to the
spectral gate, they instead increase error by **1.78%** and **1.72%**, also in
all three seeds. Tanh's overall mean is only **0.06%** below Linear's and it
wins two of three pairs; this experiment does not show a clear overall winner
between the two new shapes.

At $\lambda=0$ and $0.5$, Linear increases error by **2.72%** and **1.10%**
versus sigmoid; tanh increases it by **1.25%** and **0.39%**. These regressions
hold in every paired seed. Neither new schedule is a universal improvement.

This comparison changes the gate itself to `linear_sigma` or `tanh_sigma`,
with **one backbone and one normalized residual MSE** per model. Both new
shapes have center $\sigma_c=1.5$, value $g(\sigma_c)=0.5$ and local slope
$2/3$, matching the existing $p=4$ log-sigma sigmoid. These settings were fixed
before training; there was no hyperparameter search or test-based selection.
The tanh uses a sigma-linear argument: log-sigma tanh would be identical to
the original sigmoid and is checked algebraically rather than trained again.
This experiment compares complete noise schedules, not an independent effect
of an activation name.

The **36 new runs** cover both gates, Scalar and Fourier covariance, three
spectra ($\lambda=0,0.5,1$) and seeds 42–44. Each run uses the same
102,016-parameter MLP, initial weights, online data/noise stream, Adam settings,
EMA and 5,000-update budget as its controls. Training uses 16 concurrent
single-threaded CPU jobs. The 99 existing controls are audited against their
stored configurations, initialization and EMA hashes, training statistics and
every final validation noise bin before reuse. No control checkpoint is used
as a teacher or for initialization of a new model.

All **135 models** use the same new test bank per spectrum,
`gmm-linear-tanh-v1`: 2,048 observations at each of nine log-noise midpoint
bins. Numbers for the previous models therefore differ slightly from earlier
tables. A separate transition diagnostic has 18 noise levels from $\sigma=0.5$
to $2.5$, with 1,024 observations each at $\lambda=1$; these observations do
not enter the primary score. Reported errors are noise-scaled true-score MSE,
mean ± one sample standard deviation over three training seeds.

| Parameterization / objective | $\lambda=0$ | $\lambda=0.5$ | $\lambda=1$ |
|---|---:|---:|---:|
| Score / DSM | 0.05387 ± 0.00011 | 0.05897 ± 0.00074 | 0.07802 ± 0.00106 |
| Scalar / DSM | 0.13385 ± 0.00507 | 0.08578 ± 0.00329 | 0.07492 ± 0.00091 |
| Fourier / DSM | 0.13385 ± 0.00507 | 0.12872 ± 0.00347 | 0.12797 ± 0.00663 |
| Scalar / normalized | 0.12569 ± 0.00583 | 0.09231 ± 0.00188 | 0.07892 ± 0.00180 |
| Fourier / normalized | 0.12569 ± 0.00583 | 0.11974 ± 0.00393 | 0.08510 ± 0.00398 |
| Sigmoid Scalar | **0.03939 ± 0.00011** | 0.04251 ± 0.00037 | 0.05857 ± 0.00056 |
| Sigmoid Fourier | **0.03939 ± 0.00011** | **0.04239 ± 0.00037** | 0.05559 ± 0.00071 |
| Plateau Scalar | 0.06134 ± 0.00048 | 0.05608 ± 0.00088 | 0.06917 ± 0.00151 |
| Plateau Fourier | 0.06134 ± 0.00048 | 0.05956 ± 0.00063 | 0.06231 ± 0.00028 |
| Spectral Scalar | 0.04078 ± 0.00007 | 0.04302 ± 0.00029 | 0.05800 ± 0.00073 |
| Spectral Fourier | 0.04078 ± 0.00007 | 0.04347 ± 0.00038 | **0.05337 ± 0.00052** |
| Linear Scalar | 0.04046 ± 0.00047 | 0.04281 ± 0.00019 | 0.05800 ± 0.00058 |
| Linear Fourier | 0.04046 ± 0.00047 | 0.04286 ± 0.00049 | 0.05432 ± 0.00053 |
| Tanh Scalar | 0.03988 ± 0.00006 | 0.04245 ± 0.00015 | 0.05827 ± 0.00059 |
| Tanh Fourier | 0.03988 ± 0.00006 | 0.04255 ± 0.00038 | 0.05429 ± 0.00058 |

The $\lambda=1$ regional means show the tradeoff:

| Noise region | Fourier normalized | Sigmoid Fourier | Plateau Fourier | Spectral Fourier | Linear Fourier | Tanh Fourier |
|---|---:|---:|---:|---:|---:|---:|
| Low, $\sigma<0.3$ | 0.07813 | 0.03433 | **0.02889** | 0.03052 | 0.03141 | 0.03193 |
| Middle, $0.3\leq\sigma<1$ | 0.12718 | 0.07237 | 0.08270 | 0.07080 | **0.07050** | 0.07115 |
| High, $\sigma\geq1$ | **0.04998** | 0.06006 | 0.07535 | 0.05880 | 0.06106 | 0.05980 |

Relative to sigmoid, Linear improves low/middle errors by **8.53% / 2.58%**
but worsens high-noise error by **1.67%**. Tanh improves low/middle/high by
**7.01% / 1.69% / 0.44%**. Each regional direction holds in all three seed
pairs. Linear's middle-noise advantage over Spectral is small (**0.42%**),
though present in all three pairs. Neither new gate attains Plateau's low-noise
error or Fourier-normalized's high-noise error: at high noise Linear and tanh
remain **22.17%** and **19.64%** above Fourier normalized.

Linear reaches the Fourier-normalized adapter exactly at $\sigma\geq2.25$,
but this does not force its retrained shared backbone to make the same
predictions. The transition diagnostic shows no plateau-like spike for either
new shape. All runs completed without gradient clipping.

![Gate shape comparison across spectra and noise levels.](../../assets/gmm_gate_shape_comparison/gmm_gate_shape_comparison.svg)

![Gate shapes, residual scales and transition errors.](../../assets/gmm_gate_shape_comparison/gmm_gate_shape_diagnostics.svg)

Reproduce all 135 runs from scratch:

```bash
uv run --locked --extra figures python scripts/run_gmm_gate_shapes.py \
  --output saved/gmm_gate_shapes_reproduction --workers 16
```

To train only the 36 new arms when prior checkpoints are available, add:

```bash
--reuse-baselines saved/gmm_gated_20260924 saved/gmm_plateau_20260924 saved/gmm_spectral_20260924
```

Use `--preset smoke --seeds 42` for a short pipeline check. Changes to gate
center or sharpness require a new output directory. Future tuning should use
validation and a fresh test-bank version for final evaluation. This study uses
one GMM geometry and does not establish performance with longer training or on
sample/image quality.

Archived [protocol](../../assets/gmm_gate_shape_comparison/protocol.json) ·
[summary](../../assets/gmm_gate_shape_comparison/summary.json) ·
[paired comparisons](../../assets/gmm_gate_shape_comparison/paired_comparisons.csv) ·
[per-seed results](../../assets/gmm_gate_shape_comparison/per_seed.csv) ·
[noise regions](../../assets/gmm_gate_shape_comparison/noise_regions.csv) ·
[transition diagnostics](../../assets/gmm_gate_shape_comparison/transition_resolved.csv) ·
[gate and scale curves](../../assets/gmm_gate_shape_comparison/gate_profile.csv) ·
[checkpoint audit](../../assets/gmm_gate_shape_comparison/checkpoint_audit.json) ·
[verification](../../assets/gmm_gate_shape_comparison/verification.json).
Figures: [comparison PNG](../../assets/gmm_gate_shape_comparison/gmm_gate_shape_comparison.png) /
[PDF](../../assets/gmm_gate_shape_comparison/gmm_gate_shape_comparison.pdf),
[diagnostic PNG](../../assets/gmm_gate_shape_comparison/gmm_gate_shape_diagnostics.png) /
[PDF](../../assets/gmm_gate_shape_comparison/gmm_gate_shape_diagnostics.pdf).
