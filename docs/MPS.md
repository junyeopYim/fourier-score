# Apple MPS execution

Select `--device mps` or `device=mps`. Automatic device selection tries CUDA,
then MPS, then CPU. An explicitly requested unavailable device raises an error.

On MPS, `backend.spectral_transform=auto` uses a **real separable DFT matrix
multiplication**. It applies the same full orthonormal Fourier multiplier without
creating a complex tensor. Model and filter forward/backward computations stay
on MPS; statistics are estimated on CPU in float64, then stored as float32.
On CPU/CUDA, `auto` uses `torch.fft.fft2/ifft2`. The real matrix implementation
has higher asymptotic cost than FFT and may be slow at larger resolutions.

```bash
# Check the complete path on the actual Mac.
uv run --locked python scripts/doctor.py --device mps

# Explicit alternative: differentiable copies around CPU Fourier operations.
uv run --locked python scripts/doctor.py --device mps --spectral cpu
uv run --locked python train.py -c configs/mnist.json --device mps \
  --set backend.spectral_transform=cpu
```

The CPU filter path preserves gradients back to the backbone. `--spectral fft`
can test native MPS FFT support on the installed backend, but failures do not
silently fall back. Record the chosen backend and keep it fixed across arms.

For MPS portability, the FIR implementation replaces six-dimensional
zero-insertion padding with equivalent four-dimensional operations. Independent
CPU forward/gradient tests cover that arithmetic. The source sigma buffer is
converted to float32 before device transfer. Checkpoints save and restore MPS RNG.

CPU equivalence checks do not validate a real Apple device. Run the doctor
command on the target Mac and consult the installed version's
[PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
for hardware and OS availability.
