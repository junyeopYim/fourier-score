# Installation

Use Python 3.11 and the committed `uv.lock` to reproduce the software environment.
The project pins PyTorch 2.14.0 and TorchVision 0.29.0. Device availability depends
on the installed wheel and host; verify it instead of assuming CUDA or MPS.

```bash
uv python install 3.11
uv sync --locked --python 3.11
uv run --locked python scripts/doctor.py --device cpu
uv run --locked python -m pytest -q
```

Run commands from the repository root. Keep all paired experiments on the same
locked package versions, backend and precision. Update the lockfile deliberately
when dependencies change. A successful lock check is not a clean-install or
cross-device numerical-equivalence test.

## Optional dependencies

| Extra | Purpose |
|---|---|
| `metrics` | FID/IS through torch-fidelity; Inception weights download on first use |
| `ldm` | Native frozen KL/VQ first-stage latent experiments |
| `datasets` | Google Drive and LSUN LMDB preparation |
| `tensorboard` | Training log visualization |
| `figures` | Matplotlib figure export |
| `notebooks` | Locked JupyterLab, kernel, notebook execution and plotting |

```bash
uv sync --locked --extra ldm --extra datasets --extra metrics
uv run --locked --extra notebooks jupyter lab notebooks/gmm_fourier_residual.ipynb
uv run --locked --extra figures python scripts/plot_method.py
# Everything needed for development and CI:
uv sync --locked --all-extras
```

`uv run` can synchronize the environment to its requested extras. For development,
use `--all-extras` consistently, or run `.venv/bin/python` after syncing. Training
should use its own environment, unchanged for the lifetime of a run and resume.

## Devices

```bash
uv run --locked python scripts/doctor.py --device cuda
# On a Mac with supported Apple hardware:
uv run --locked python scripts/doctor.py --device mps
```

The default package source is PyPI. If selecting a specific CPU/CUDA wheel index,
configure **both** torch and torchvision with an explicit uv source and regenerate
the lockfile; record that environment as a different experimental setup. See
[uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) and the
[official PyTorch installer](https://pytorch.org/get-started/locally/) for host-specific
installation. Do not mix wheel variants between paired arms or during resume.

Model weights, datasets, wheels and virtual environments are not redistributed.
Official reference weights have their own source/license manifests; see
[SCORE_SDE.md](SCORE_SDE.md), [LDM.md](LDM.md) and [PROVENANCE.md](PROVENANCE.md).
