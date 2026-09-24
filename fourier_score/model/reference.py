"""Gaussian reference scores and frequencywise residual parameterizations.

Read this file with the method section in README.md. The loss is selected separately.
"""

import math
import torch
from torch import nn
from fourier_score.gates import validate_gate
from fourier_score.model.spectral import SpectralFilter, conjugate_symmetrize


class FourierGaussian(nn.Module):
    """Fixed Gaussian reference plus a neural residual; no learned parameters.

    ``stats['mean']`` and ``stats['power']`` have shape [C, H, W].
    The full method uses the empirical Fourier power P. The scalar control
    replaces P by a channelwise constant. Both normalize the residual using
    the second moment of the denoising target after reference subtraction.
    """

    def __init__(self, stats: dict, backend="auto", *, covariance="fourier", gate=None):
        super().__init__()
        self.gate = validate_gate(gate)
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

    def gate_value(self, sigma):
        """Gate coefficients shared by target and score.

        Noise-only gates are [B,1,1,1]. spectral_cap uses [B,C,H,W] for
        Fourier covariance and [B,C,1,1] for scalar covariance.
        """
        return self._gate_weights(sigma)[0]

    def _gate_weights(self, sigma, prior=None):
        s = sigma[:, None, None, None]
        if self.gate["mode"] in ("linear_log_sigma", "bounded_log_sigmoid"):
            lo, hi = self.gate["sigma_lo"], self.gate["sigma_hi"]
            span = math.log(hi) - math.log(lo)
            z = ((s.log() - math.log(lo)) / span).clamp(0, 1)
            z = torch.where(s <= lo, torch.zeros_like(z),
                            torch.where(s >= hi, torch.ones_like(z), z))
            if self.gate["mode"] == "linear_log_sigma":
                return z, 1-z
            center = (math.log(self.gate["sigma_switch"]) - math.log(lo)) / span
            # sigmoid(p * (logit(z)-logit(center))) with exact flat plateaus.
            # Positive powers avoid log(0) at endpoints; scaling by the larger
            # term prevents simultaneous underflow even for steep transitions.
            a, b = z * (1-center), (1-z) * center
            common = torch.maximum(a, b)
            a, b = (a/common).pow(self.gate["sharpness"]), (b/common).pow(self.gate["sharpness"])
            return a/(a+b), b/(a+b)
        if self.gate["mode"] in ("linear_sigma", "tanh_sigma"):
            offset = s / self.gate["sigma_switch"] - 1
            if self.gate["mode"] == "linear_sigma":
                ramp = (self.gate["sharpness"] / 4) * offset
                return (.5 + ramp).clamp(0, 1), (.5 - ramp).clamp(0, 1)
            # (1+tanh(z/2))/2 = sigmoid(z). Opposite logits preserve the
            # small complement after tanh itself would round to one.
            # The argument is LINEAR in sigma; log-sigma tanh would be the
            # existing log_sigma gate. All three share center and local slope.
            logit = self.gate["sharpness"] * offset
            return torch.sigmoid(logit), torch.sigmoid(-logit)
        if self.gate["mode"] in ("log_sigma", "log_sigma_plateau", "spectral_cap"):
            logit = self.gate["sharpness"] * (s.log() - math.log(self.gate["sigma_switch"]))
            # Compute 1-g separately: subtracting a rounded sigmoid loses the
            # residual near g=1, especially when the power spectrum is small.
            g, complement = torch.sigmoid(logit), torch.sigmoid(-logit)
            if self.gate["mode"] == "spectral_cap":
                lo, hi = self.gate["sigma_lo"], self.gate["sigma_hi"]
                z = ((s.log() - math.log(lo)) / (math.log(hi) - math.log(lo))).clamp(0, 1)
                ramp = z.square() * (3 - 2 * z)
                ramp = torch.where(s <= lo, torch.zeros_like(ramp),
                                   torch.where(s >= hi, torch.ones_like(ramp), ramp))
                power = self.power[None] if prior is None else prior
                # Square q*s together to avoid underflow of q^2 at high noise.
                correction = ramp * ((complement * s) / (self.gate["delta"] * power.sqrt())).square()
                root = (1 + correction).sqrt()
                remaining = complement / root
                # q*(1-1/root) = (q/root)*correction/(root+1): retain small
                # corrections without subtracting two almost equal values.
                g = g + remaining * (correction / (root + 1))
                return g, remaining
            if self.gate["mode"] == "log_sigma_plateau":
                lo, hi = self.gate["sigma_lo"], self.gate["sigma_hi"]
                z = ((s.log() - math.log(lo)) / (math.log(hi) - math.log(lo))).clamp(0, 1)
                w = z.square() * (3 - 2 * z)
                # 1-w = (1-z)^2(1+2z), avoiding cancellation near the plateau.
                taper = (1 - z).square() * (1 + 2 * z)
                lifted = g + w * complement
                remaining = taper * complement
                # Compare sigma itself so the configured endpoints are exact
                # even when logarithms round differently in FP32.
                g = torch.where(s <= lo, g, torch.where(s >= hi, torch.ones_like(g), lifted))
                complement = torch.where(s <= lo, complement,
                                         torch.where(s >= hi, torch.zeros_like(complement), remaining))
            return g, complement
        value = self.gate["value"] if self.gate["mode"] == "constant" else 1.0
        return torch.full_like(s, value), torch.full_like(s, 1.0 - value)

    def _coefficients(self, sigma, prior):
        s = sigma[:, None, None, None]
        total = prior + s.square()
        g, complement = self._gate_weights(sigma, prior)
        # Positive terms avoid cancellation when g~1 and sigma^2 >> prior.
        scale = (complement.square() + g * (1 + complement) * (prior / total)).sqrt()
        return g, complement, scale, total

    @torch.no_grad()
    def normalized_target(self, clean, noise, sigma):
        """Return the VE normalized target after subtracting g*sigma*s_G.

        T_g = [g*sigma*F(x-mu) - (P+(1-g)*sigma^2)*F(noise)] / (P+sigma^2).
        c^2 = (1-g)^2 + g*(2-g)*P/(P+sigma^2); target = F^-1[T_g/c].
        With gate.mode=none, g=1 recovers the original normalized target.

        Compute directly from clean data, avoiding cancellation in the Gaussian
        score at large sigma followed by division by a small residual scale b.
        The scalar control uses the same stored channelwise power without FFTs.
        """
        if self.gate["mode"] == "constant" and self.gate["value"] == 0:
            return -noise
        s = sigma[:, None, None, None]
        if self.gate["mode"] == "none":
            # Preserve the original path, including its rounding and checkpoints.
            root_power = self.power[None].sqrt()
            root_total = (self.power[None] + s.square()).sqrt()
            clean_scale = (s / root_total) / root_power
            noise_scale = root_power / root_total
        else:
            g, complement, scale, total = self._coefficients(sigma, self.power[None])
            clean_scale = (g * (s / total)) / scale
            noise_scale = (self.power[None] / total + complement * (s.square() / total)) / scale
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

        For a gate g, replace the reference coefficient by g and the residual
        scale by c = sqrt((1-g)^2 + g*(2-g)*b^2). Active gates are exposed only
        for VE experiments; the disabled path preserves the original adapter.

        VE alpha=1 is the proposed method. For explicit DDPM experiments,
        V_k = alpha^2 P_k, mean_t = alpha mu (documented generalization).
        """
        if self.gate["mode"] == "constant" and self.gate["value"] == 0:
            return raw
        a = alpha[:, None, None, None]
        s = sigma[:, None, None, None]
        prior = a.square() * self.power[None]
        denom = prior + s.square()
        if self.gate["mode"] == "none":
            g, scale = 1.0, (prior / denom).sqrt()
        else:
            g, _, scale, denom = self._coefficients(sigma, prior)
        if self.covariance == "scalar":
            # A spatially constant spectral multiplier is pointwise in pixels.
            gaussian = -s * (y - a * self.mean[None]) / denom
            residual = scale * raw
            return g * gaussian + residual
        if self.gate["mode"] == "spectral_cap":
            # g_k is a FREQUENCY multiplier, not a pixelwise mask. Keep it
            # inside the transform; old noise-only paths retain their rounding.
            gaussian = -s * self.filter(y - a * self.mean[None], g / denom)
            return gaussian + self.filter(raw, scale)
        gaussian = -s * self.filter(y - a * self.mean[None], denom.reciprocal())
        residual = self.filter(raw, scale)
        return g * gaussian + residual
