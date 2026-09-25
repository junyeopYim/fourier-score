# Reports

[Back to README](../README.md) · [Provenance](provenance.md)

Full write-ups of the method details and mechanism studies. They were moved
from the top-level README with their text and results unchanged; the README
keeps a short summary with one figure under each original heading. Figures,
tables and archived measurements stay in [assets/](../assets/).

## Method

| Report | Summary |
|---|---|
| [Noise-gated residuals and gate designs](method/gates.md) | Equations, stable implementations, `--dry-run` commands and signature rules for every `fourier.gate.mode`, the log-axis gate design, and the options shared by the GMM comparison runners. |

## GMM mechanism studies

All studies use the 8 × 8 matched-moment GMM of the
[GMM notebook](../notebooks/README.md), a 102,016-parameter MLP, 5,000 updates,
paired seeds 42–44 and noise-scaled true-score MSE against the analytic score.
Later studies check earlier checkpoints against their archived hashes and
metrics before comparing with them, and evaluate every model on a new shared
test bank, so baseline numbers differ slightly between reports. Every report
after the loss comparison gives its reproduction command near the end.

| Report | Summary |
|---|---|
| [Loss comparison](gmm/loss-comparison.md) | At $\lambda=1$, normalized residual loss reduced Fourier error by 33.4% relative to Fourier DSM; Scalar DSM remained best. 45 runs. |
| [Gated comparison](gmm/gated-comparison.md) | A validation-selected sigmoid gate ($\sigma_c=1.5$): Gated Fourier is 25.8% below Scalar DSM at $\lambda=1$, but 20.0% above ungated Fourier at high noise. 99 runs. |
| [Plateau comparison](gmm/plateau-comparison.md) | Forcing $g=1$ above $\sigma=1$ increased Fourier error by 12.6% overall and 25.8% at high noise versus the sigmoid gate. 18 new runs. |
| [Spectral gate](gmm/spectral-gate.md) | A frequencywise gate reduced error by 4.0% versus the sigmoid gate at $\lambda=1$; flat and intermediate spectra regressed slightly. 18 new runs. |
| [Linear and tanh gates](gmm/linear-tanh-gates.md) | Sigma-linear and sigma-tanh gates both improve the sigmoid by about 2.3% at $\lambda=1$; Spectral retains the lowest overall mean. 36 new runs. |
| [Log-axis gates](gmm/log-axis-gates.md) | Log-linear nearly ties the sigmoid; bounded S lowers low-noise error by 13.5% but raises total error by 8.6% at $\lambda=1$. Both regress at $\lambda=0,0.5$. 36 new runs, 135 audited controls. |

## Provenance

Every result above belongs to provenance epoch 0, the history up to tag
`pre-template-refactor`. [provenance.md](provenance.md) lists the epochs, the
source hash each study recorded, and how to resume or re-evaluate runs from an
earlier epoch; [path-map.json](path-map.json) maps epoch-0 file paths to the
current layout.

## Adding a report

Put it under `reports/<area>/<name>.md`, starting with an H1 title and a link
back to its README section. Add a row above and keep a two-to-four sentence
summary with one figure in the README. Links are relative to the report, so
repository files are reached through `../../`.
