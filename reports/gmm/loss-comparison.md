# GMM loss comparison

[Back to README](../../README.md#gmm-loss-comparison) · [All reports](../README.md) · Provenance: [epoch 0](../provenance.md)

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

![GMM loss comparison: final true-score error across spectral heterogeneity, validation curves, and error by noise level.](../../assets/gmm_loss_comparison/gmm_loss_comparison.svg)

*Mean ± one training-seed standard deviation; the last two panels use
$\lambda=1$. [PNG](../../assets/gmm_loss_comparison/gmm_loss_comparison.png) ·
[PDF](../../assets/gmm_loss_comparison/gmm_loss_comparison.pdf).*

Archived measurements: [summary](../../assets/gmm_loss_comparison/summary.csv) ·
[per-seed results](../../assets/gmm_loss_comparison/per_seed.csv) ·
[paired comparisons](../../assets/gmm_loss_comparison/paired_comparisons.csv) ·
[noise levels](../../assets/gmm_loss_comparison/noise_resolved.csv) ·
[frequency bands](../../assets/gmm_loss_comparison/frequency_resolved.csv) ·
[validation curves](../../assets/gmm_loss_comparison/validation_curves.csv).
The [run configuration and provenance](../../assets/gmm_loss_comparison/summary.json)
record source commit `f8df48f`, environment, source hashes, and successful checks
of paired initialization, data streams, and test banks.
[Oracle validation](../../assets/gmm_loss_comparison/oracle_validation.json) records
the independent numerical check.
