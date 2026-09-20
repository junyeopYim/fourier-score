# Installation and version policy

`pyproject.toml` pins PyTorch 2.14.0 and TorchVision 0.29.0. Python 3.11 is the
default, and the repository includes a tracked `uv.lock`.

The September 20 verification ran in the existing `.venv` with Python 3.11.15,
torch 2.14.0+cu130 and torchvision 0.29.0+cu130. CPU and CUDA doctor checks
passed on an NVIDIA GeForce RTX 5060 Ti. `uv lock --check --offline` also passed.
See [the verification record](VALIDATION.md) for exact environment and scope;
this was not a clean-install or MPS validation.

```bash
uv python install 3.11
uv sync --locked --python 3.11
uv run --locked python scripts/doctor.py
uv run --locked python -m pytest -q
```

Use `uv sync --locked` for subsequent installs and keep all experimental arms
on the same wheel/library versions. Regenerate and commit the lockfile only
when intentionally changing dependency requirements or sources.

`verification/uv_lock_attempt.txt` documents the original September 19 package
access failure. Its older CPU-only environment and missing-lockfile statements
are preserved in [the historical record](VALIDATION_2026-09-19.md); they do not
describe the current checkout.

The default dependency source is PyPI. To choose a specific official CUDA or
CPU wheel index, configure **both torch and torchvision** with uv sources;
do not use an extra-index flag that can pull unrelated packages from that index.
For example, the CPU-only source for Linux/Windows can be appended to pyproject:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]
torchvision = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]
```

For CUDA, substitute the official index for the chosen supported wheel/driver
combination, and regenerate the lockfile. This project intentionally does not
invent a fixed CUDA 2.14 wheel variant or driver minimum without validating it.
macOS remains on PyPI for its MPS-capable build. PyPI on Windows may select a
CPU-only wheel; verify availability via doctor rather than assuming CUDA.

Official uv integration guide:
https://docs.astral.sh/uv/guides/integration/pytorch/
Official PyTorch installer:
https://pytorch.org/get-started/locally/

Optional packages:

```bash
uv sync --locked --extra metrics        # torch-fidelity; downloads Inception on use
uv sync --locked --extra tensorboard    # SummaryWriter
```

Every time optional dependency selections change, preserve the corresponding
uv command / environment for all arms. The historical CPU build also exercised
Python 3.13 with older torch/torchvision versions. No `.venv`, wheels, datasets
or model weights are tracked in the repository. No credentials or repository
access are required at runtime.
