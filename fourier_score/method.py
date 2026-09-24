"""Gaussian reference scores and frequencywise residual parameterizations.

Read this file with the method section in README.md. The loss is selected separately.
"""

import torch
from torch import nn
from fourier_score.spectral import SpectralFilter, conjugate_symmetrize


GAUSSIAN_OBJECTIVES = {
    "scalar_gaussian": {"covariance": "scalar"},
    "fourier_gaussian": {"covariance": "fourier"},
}
OBJECTIVES = ("score", "diffusion", *GAUSSIAN_OBJECTIVES)
# Score and diffusion are sign conventions, so only one is in the main ablation.
COMPARISON_OBJECTIVES = ("score", *GAUSSIAN_OBJECTIVES)


class FourierGaussian(nn.Module):
    """Fixed Gaussian reference plus a neural residual; no learned parameters.

    ``stats['mean']`` and ``stats['power']`` have shape [C, H, W].
    The full method uses the empirical Fourier power P. The scalar control
    replaces P by a channelwise constant. Both normalize the residual using
    the second moment of the denoising target after reference subtraction.
    """

    def __init__(self, stats: dict, backend="auto", *, covariance="fourier"):
        super().__init__()
        if covariance not in ("fourier", "scalar"):
            raise ValueError("Invalid Gaussian covariance")
        mean = stats["mean"].detach().cpu().float().clone()
        power = stats["power"].detach().cpu().float().clone()
        if mean.ndim != 3 or power.shape != mean.shape:
            raise ValueError("Statistics must be [C,H,W]")
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(power).all()
            or (power <= 0).any()
        ):
            raise ValueError("Invalid Fourier statistics")
        if not torch.allclose(power, conjugate_symmetrize(power), atol=1e-6, rtol=1e-5):
            raise ValueError("Power is not conjugate symmetric")
        self.covariance = covariance
        # Average the SAME floored training spectrum, retaining the full mean.
        # The shared statistics cache must never be flattened in place.
        if covariance == "scalar":
            power = power.mean(dim=(-2, -1), keepdim=True)
        self.register_buffer("mean", mean)
        self.register_buffer("power", power)
        self.filter = SpectralFilter(*mean.shape[-2:], backend)

    def resolved_backend(self, device):
        return (
            "elementwise"
            if self.covariance == "scalar"
            else self.filter.resolved_backend(device)
        )

    @torch.no_grad()
    def normalized_target(self, clean, noise, sigma):
        """Return the VE target F^-1[(sigma*F(x-mu) - P*F(noise))/sqrt(P*(P+sigma^2))].

        Compute directly from clean data, avoiding cancellation in the Gaussian
        score at large sigma followed by division by a small residual scale b.
        The scalar control uses the same stored channelwise power without FFTs.
        """
        s = sigma[:, None, None, None]
        root_power = self.power[None].sqrt()
        root_total = (self.power[None] + s.square()).sqrt()
        clean_scale = (s / root_total) / root_power
        noise_scale = root_power / root_total
        centered = clean - self.mean[None]
        if self.covariance == "scalar":
            return clean_scale * centered - noise_scale * noise
        return self.filter(centered, clean_scale) - self.filter(noise, noise_scale)

    def scaled_score(self, raw, y, alpha, sigma):
        """Return the scaled score sigma*s, with shape [B, C, H, W].

        ``raw`` is h_theta, the backbone output before sigma division.
        ``y`` and ``raw`` have shape [B, C, H, W]; alpha and sigma are [B].

        V = alpha^2 P, D = V + sigma^2, b = sqrt(V / D).
        sigma*s_G = -sigma F^-1[F(y - alpha*mu) / D].
        sigma*s_theta = sigma*s_G + F^-1[b F(raw)].

        VE alpha=1 is the proposed method. For explicit DDPM experiments,
        V_k = alpha^2 P_k, mean_t = alpha mu (documented generalization).
        """
        a = alpha[:, None, None, None]
        s = sigma[:, None, None, None]
        prior = a.square() * self.power[None]
        denom = prior + s.square()
        if self.covariance == "scalar":
            # A spatially constant spectral multiplier is pointwise in pixels.
            gaussian = -s * (y - a * self.mean[None]) / denom
            residual = (prior / denom).sqrt() * raw
            return gaussian + residual
        gaussian = -s * self.filter(y - a * self.mean[None], denom.reciprocal())
        residual = self.filter(raw, (prior / denom).sqrt())
        return gaussian + residual
