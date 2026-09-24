"""Frozen native KL/VQ first stages; no Lightning or perceptual-loss dependency."""

import math
from pathlib import Path

import torch
from torch import nn

from fourier_score.provenance import file_sha256

from .upstream.model import Decoder, Encoder


def read_state(path):
    """Load a checkpoint with PyTorch's weights-only deserializer."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj.get("state_dict", obj)
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError("Expected a CompVis state_dict checkpoint")
    return state


class FrozenFirstStage(nn.Module):
    def __init__(self, spec, scale_factor=1.0):
        super().__init__()
        p = spec["first_stage"]["params"]
        self.kind = spec["first_stage"]["kind"]
        self.scale_factor = float(scale_factor)
        dd = p["ddconfig"]
        self.encoder = Encoder(**dd)
        self.decoder = Decoder(**dd)
        self.embed_dim = p["embed_dim"]
        kl = self.kind == "AutoencoderKL"
        if bool(dd["double_z"]) != kl:
            raise ValueError("First-stage posterior shape disagrees with kind")
        self.quant_conv = nn.Conv2d(
            dd["z_channels"] * (2 if kl else 1), self.embed_dim * (2 if kl else 1), 1
        )
        self.post_quant_conv = nn.Conv2d(self.embed_dim, dd["z_channels"], 1)
        if not kl:
            self.quantize = nn.Module()
            self.quantize.embedding = nn.Embedding(p["n_embed"], self.embed_dim)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent .train() calls must never enable first-stage dropout.
        return super().train(False)

    @torch.no_grad()
    def encode_parameters(self, images):
        """Unscaled continuous VQ features, or unscaled KL mean + log variance."""
        params = self.quant_conv(self.encoder(images))
        if self.kind == "AutoencoderKL":
            mean, logvar = params.chunk(2, dim=1)
            params = torch.cat((mean, logvar.clamp(-30.0, 20.0)), dim=1)
        return params

    @torch.no_grad()
    def encode(self, images, generator=None):
        return sample_posterior(
            self.encode_parameters(images), self.kind, self.scale_factor, generator
        )

    @torch.no_grad()
    def decode(self, latent):
        h = latent / self.scale_factor
        if self.kind == "VQModelInterface":
            # Native nearest-codebook quantization, chunked only across queries.
            flat = h.permute(0, 2, 3, 1).contiguous().reshape(-1, self.embed_dim)
            weight = self.quantize.embedding.weight
            norms = weight.square().sum(1)
            indices = []
            for block in flat.split(1024):
                distance = (
                    block.square().sum(1, keepdim=True)
                    + norms
                    - 2 * torch.einsum("bd,dn->bn", block, weight.t())
                )
                indices.append(distance.argmin(1))
            quantized = (
                self.quantize.embedding(torch.cat(indices))
                .view(h.shape[0], h.shape[2], h.shape[3], self.embed_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # Preserve even the native straight-through expression's rounding.
            h = h + (quantized - h)
        return self.decoder(self.post_quant_conv(h))


def sample_posterior(params, kind, scale, generator=None):
    if kind == "AutoencoderKL":
        mean, logvar = params.chunk(2, dim=-3)
        # CompVis samples on CPU then transfers. An explicit CPU RNG pairs arms.
        noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32).to(
            mean.device
        )
        return scale * (mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * noise)
    return scale * params


def load_first_stage(path, spec, device="cpu", expected_sha256=None):
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Frozen first-stage checkpoint hash mismatch")
    state = read_state(path)
    if spec["scale_by_std"]:
        if "scale_factor" not in state:
            raise ValueError(
                "KL LDM checkpoint must contain its learned scale_factor; do not recompute it"
            )
        scale = float(state["scale_factor"])
    else:
        scale = float(state.get("scale_factor", spec["scale_factor"]))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid checkpoint latent scale")
    model = FrozenFirstStage(spec, scale)
    prefix = "first_stage_model."
    extracted = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not extracted:
        raise ValueError(
            "Expected a full LDM checkpoint containing first_stage_model.*"
        )
    # Perceptual/discriminator training state is not part of the frozen mapping.
    ignored = [k for k in extracted if k.startswith(("loss.", "model_ema."))]
    for key in ignored:
        del extracted[key]
    model.load_state_dict(extracted, strict=True)
    model.to(device).eval()
    metadata = {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": digest,
        "scale_factor": scale,
        "kind": model.kind,
        "ignored_training_keys": ignored,
        "posterior": "resample_diagonal_gaussian"
        if model.kind == "AutoencoderKL"
        else "continuous_prequantization",
    }
    return model, metadata
