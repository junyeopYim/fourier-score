# Usage guide

Start with the [README](../README.md) for a synthetic end-to-end run, the method
and current research questions. All commands run from the repository root.

| Task | Guide |
|---|---|
| Install a locked environment and choose optional dependencies | [INSTALL.md](INSTALL.md) |
| Train CIFAR-10, sample and compute FID | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Understand config overrides and resume | [CONFIG.md](CONFIG.md) |
| Prepare frozen-autoencoder latent experiments | [LDM.md](LDM.md) |
| Import official inference references | [SCORE_SDE.md](SCORE_SDE.md) |
| Understand the derivation and controls | [MATH.md](MATH.md), [ABLATIONS.md](ABLATIONS.md) |
| Run the matched-moment GMM notebook | [Notebook](../notebooks/gmm_fourier_residual.ipynb) |
| Generate scientific figures | [FIGURES.md](FIGURES.md) |
| Debug, test, and keep local operational notes | [DEVELOPMENT.md](DEVELOPMENT.md) |

The primary pixel comparison uses `score`, `scalar_gaussian` and
`fourier_gaussian`; explicitly select the two Gaussian arms for a study without
baseline retraining. `diffusion` supplies an additional noise-prediction sign
convention. Native latent runs use `epsilon` as their baseline.

Raw data, checkpoints, execution logs and private notes are local artifacts.
Public configs, derivations, figure sources and tests describe how to reproduce
the protocol. A smoke run verifies execution and does not establish image quality.
