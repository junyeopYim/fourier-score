# Verification record — 2026-09-19

## What actually ran

`python -m pytest -q`: **114 passed, 3 skipped, 0 failed** (9.40 seconds in this
execution). Raw output: `verification/pytest.txt`.

Runtime: Python 3.13.5; torch 2.10.0+cpu; torchvision 0.25.0+cpu; NumPy 2.3.5;
Pillow 12.3.0; pytest 9.0.2. Linux x86-64, CPU only. The source fingerprint
for the final training implementation is `e3c05e2d4517f4b624d19a587b3ab9dac94227d25d3caedb60046d9a242a4995`.

**Target dependencies are torch 2.14.0 and torchvision 0.29.0, not the versions
used for these executions.** External PyPI access timed out during `uv lock`.
`verification/uv_lock_attempt.txt` preserves the error. There is no uv.lock in
this ZIP. First run `uv sync --python 3.11` on an internet-connected machine.
No test result below is a claim of live 2.14, Apple MPS, or CUDA validation.

## Automated checks

- Three source backbone files match the original Git blob hashes. The original
  attention / NIN / residual module classes and parameter names remain present.
- All three objectives have identical architecture and initial backbone weights.
  The full CIFAR preset, not just the smoke model, was instantiated and audited:
  **62,758,787 trainable parameters**, 62,758,915 total parameters (including fixed
  random time embedding), **6 attention blocks and 44 BigGAN++ residual blocks**.
  Results: `verification/cifar10_architecture.json`. MNIST has its own audit.
- 36 combinations of residual block, FIR on/off, progressive output and
  progressive input have real NCSN++ forward/backward coverage; both embedding
  styles are tested.
- FIR values and gradients match an independent implementation of the source's
  6D padding algorithm. The new real DFT matmul values and gradients match FFT,
  including odd and rectangular dimensions. The explicit CPU FFT transfer path
  retains autograd.
- Gaussian score, residual scaling, DDPM alpha factors, conjugate symmetry,
  Welford statistics, non-Gaussian moment identity, score/epsilon equivalence,
  and both forward processes' objective/sampling paths are checked.
- Training-only split and augmentation-consistent statistics are checked. The
  stateful data stream resumes at consumed, not prefetched, batches with zero
  and two workers.
- Uninterrupted 4 updates vs 2 updates + resume to 4 give bitwise identical CPU
  weights, EMA, stream and RNG state. EMA snapshot and full-checkpoint inference
  agree. Evaluation restores RNG and is repeatable. Changed objective, process
  or architecture is rejected on resume.
- **3 device tests skipped**: real MPS (automatic real DFT), real MPS (explicit
  CPU FFT fallback), and CUDA. These require hardware not present here.

## End-to-end command checks

`verification/cli_commands.json` records actual commands and return codes.
For **score, diffusion, and Fourier Gaussian**, the commands exercised 3 training
updates, EMA checkpoint loading, DSM evaluation, and PNG/NPZ/preview generation.
The final sample round intentionally uses 3 requested images with batch size 2;
all generated image files decode and NPZ counts/shapes match (output_integrity.json).

A separate discrete DDPM smoke run also trained and sampled. A Fourier Gaussian
checkpoint resumed from update 3 to 4. Real-image export was checked on the
synthetic fixture. A verification command initially used an unsupported
`--num-images` flag; argparse correctly rejected it. The corrected `--limit 4`
command passed; both records remain visible. This was a command typo, not a
silently accepted configuration setting.

`doctor.py --device cpu` and `doctor.py --device cpu --spectral matmul` completed
real small-NCSN++ forward/backward/Adam/sampling tests for every objective and
RNG restoration checks; their raw JSON files are included. CUDA/MPS versions of
these commands still need to run on the actual target hardware.

The CLI logs use absolute temporary build paths for auditability. These paths
are **not** project configuration defaults and are not needed by users.

## Not verified / no implied quality claim

There was no downloaded real-dataset training, long NCSN++ run, FID/IS benchmark,
real Mac/GPU execution, optional Inception download, TensorBoard run, or latest
pinned-environment installation. Smoke images are untrained functional outputs,
not evidence of generation quality. The new Fourier method is not claimed to
improve FID merely because its formula and software tests pass.

No existing checkpoint from the user's old repository is silently converted.
This is a new experiment/checkpoint format. Same-hardware CPU tests do not imply
bitwise reproducibility across GPUs, operating systems, Torch versions or MPS.

To repeat locally after successful dependency resolution:

```bash
uv run --locked python -m pytest -q
uv run --locked python doctor.py --device mps   # on an eligible Mac
uv run --locked python doctor.py --device cuda  # on an NVIDIA system
uv run --locked python inspect_model.py -c configs/cifar10.json
```
