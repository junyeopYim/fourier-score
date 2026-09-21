"""Estimate training-only image means and Fourier power spectra."""

import torch

from fourier_score.spectral import conjugate_symmetrize


@torch.no_grad()
def estimate_stats(batches, floor: float = 1e-4, *, posterior_variance=False) -> dict:
    """Population moments; optionally integrate diagonal Gaussian posteriors.

    With posterior_variance=True, batches yield (mean, variance). Independent
    spatial posterior noise contributes its spatial-average variance to EVERY
    orthonormal Fourier mode, so KL statistics need no Monte Carlo draw.
    """
    if floor <= 0:
        raise ValueError("floor must be positive")
    n = 0
    mean = None
    m2 = None
    within = None
    # Batch-combined Welford in image space / spectral second moments.
    for batch in batches:
        variance = None
        if posterior_variance:
            batch, variance = batch
        x = batch.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 4 or not len(x) or not torch.isfinite(x).all():
            raise ValueError("Statistics require nonempty finite BCHW data")
        b = len(x)
        if variance is not None:
            variance = variance.detach().to(device="cpu", dtype=torch.float64)
            if (
                variance.shape != x.shape
                or not torch.isfinite(variance).all()
                or (variance < 0).any()
            ):
                raise ValueError("Invalid diagonal posterior variance")
            contribution = variance.mean((-2, -1), keepdim=True).sum(0)
            within = contribution if within is None else within + contribution
        bm = x.mean(0)
        z = torch.fft.fft2(x - bm, norm="ortho")
        bm2 = z.abs().square().sum(0)
        if mean is None:
            mean = bm
            m2 = bm2
            n = b
        else:
            if bm.shape != mean.shape:
                raise ValueError("Inconsistent image shapes")
            delta = bm - mean
            m2 += bm2 + torch.fft.fft2(delta, norm="ortho").abs().square() * (
                n * b / (n + b)
            )
            mean += delta * (b / (n + b))
            n += b
    if n < 2:
        raise ValueError("At least two training images required")
    raw = conjugate_symmetrize(
        (m2 + (within if within is not None else 0)) / n
    ).clamp_min(0)
    return {
        "mean": mean.float(),
        "power": raw.clamp_min(floor).float(),
        "n_effective": n,
        "floor": float(floor),
        "floored_fraction": float((raw < floor).double().mean()),
    }
