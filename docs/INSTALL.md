# Installation and version policy

2026-09-19 checked official release records:

- PyTorch 2.14.0, published 2026-09-02:
  https://github.com/pytorch/pytorch/releases/tag/v2.14.0
- TorchVision 0.29.0, published 2026-09-02:
  https://github.com/pytorch/vision/releases/tag/v0.29.0

`pyproject.toml` pins these versions. This is a target environment, not a claim
that those wheels were installed in the author's CPU-only build environment.
That environment already had torch 2.10.0+cpu / torchvision 0.25.0+cpu. External
DNS/package access failed; see `verification/uv_lock_attempt.txt`.

```bash
uv python install 3.11
uv sync --python 3.11       # creates .venv and a REAL uv.lock
uv run --locked python doctor.py
uv run --locked python -m pytest -q
```

Commit uv.lock after successful resolution. Later use `uv sync --locked`.
Do not create a placeholder lockfile or silently install a different torch.
Keep all experimental arms on the same wheel/library versions.

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
uv sync --extra metrics        # torch-fidelity; downloads Inception on use
uv sync --extra tensorboard    # SummaryWriter
```

Every time optional dependency selections change, preserve the corresponding
uv command / environment for all arms. Python 3.11 is the default; the code was
also exercised on Python 3.13. No `.venv`, wheels, datasets or model weights are
included in the ZIP. No credentials or repository access are required at runtime.
