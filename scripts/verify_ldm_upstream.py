"""Opt-in original-source parity audit. Downloads pinned code, no model weights."""

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from fourier_score.ldm.config import defaults, load_spec
from fourier_score.ldm.first_stage import FrozenFirstStage
from fourier_score.ldm.model import Denoiser, sample_latents
from fourier_score.ldm.upstream import model as native_layers
from fourier_score.ldm.upstream import util as native_util
from fourier_score.utils import ROOT, configure_runtime, json_write, source_hash

parser = argparse.ArgumentParser(
    description="Compare KL/VQ first-stage and DDIM arithmetic against pinned original source (tiny models)."
)
parser.add_argument("--source-dir", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error(f"Output already exists: {args.output}")
# The original DDIM buffer loader assumes CUDA; this audit retains that behavior.
if not torch.cuda.is_available():
    parser.error(
        "This original-source parity audit requires CUDA; run tests/test_ldm.py for portable CPU checks"
    )
temporary = tempfile.TemporaryDirectory(prefix="fourier-ldm-parity-")
root = args.source_dir or Path(temporary.name)
ldm_revision = "a506df5756472e2ebaf9078affdde2c4f1502cd4"
taming_revision = "3ba01b241669f5ade541ce990f7650a3b8f65318"
provenance = json.loads(
    (ROOT / "fourier_score/ldm/upstream/PROVENANCE.json").read_text()
)
for path in (
    "ldm/models/autoencoder.py",
    "ldm/models/diffusion/ddim.py",
    "ldm/modules/distributions/distributions.py",
    "taming/modules/vqvae/quantize.py",
):
    destination = root / path
    if not destination.exists():
        if args.source_dir:
            parser.error(f"Missing original source: {destination}")
        project, revision = (
            ("taming-transformers", taming_revision)
            if path.startswith("taming/")
            else ("latent-diffusion", ldm_revision)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(
            f"https://raw.githubusercontent.com/CompVis/{project}/{revision}/{path}",
            timeout=30,
        ) as response:
            destination.write_bytes(response.read())
    if (
        hashlib.sha256(destination.read_bytes()).hexdigest()
        != provenance["verification_sources"][path]["sha256"]
    ):
        parser.error(f"Original source hash mismatch: {path}")
torch.set_num_threads(2)
torch.manual_seed(2026)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pl = types.ModuleType("pytorch_lightning")
pl.LightningModule = nn.Module
sys.modules["pytorch_lightning"] = pl
sys.modules["ldm.modules.diffusionmodules.model"] = native_layers
sys.modules["ldm.modules.diffusionmodules.util"] = native_util
util = types.ModuleType("ldm.util")
util.instantiate_from_config = lambda c: nn.Identity()
sys.modules["ldm.util"] = util
load(
    "ldm.modules.distributions.distributions",
    root / "ldm/modules/distributions/distributions.py",
)
load("taming.modules.vqvae.quantize", root / "taming/modules/vqvae/quantize.py")
ae = load("original_autoencoder", root / "ldm/models/autoencoder.py")
ddim = load("original_ddim", root / "ldm/models/diffusion/ddim.py")
cfg = defaults("ffhq")
cfg["device"] = "cuda"
configure_runtime(cfg)
spec = load_spec(cfg)
spec.update(image_size=8, latent_shape=[2, 4, 4], timesteps=10)
spec["unet"] = {
    "image_size": 4,
    "in_channels": 2,
    "out_channels": 2,
    "model_channels": 32,
    "attention_resolutions": [1, 2],
    "num_res_blocks": 1,
    "channel_mult": [1, 2],
    "num_heads": 4,
}
report = {
    "source_revision": "a506df5756472e2ebaf9078affdde2c4f1502cd4",
    "implementation_source_sha256": source_hash(),
    "methodology": "Original autoencoder.py and ddim.py loaded with Lightning replaced only by nn.Module; computational layers use the import-only vendored modules. RNG noise_like is aligned to the explicit CPU generator for stochastic DDIM. Tiny native architectures and identical weights/inputs.",
}
for kind in ["AutoencoderKL", "VQModelInterface"]:
    kl = kind == "AutoencoderKL"
    p = {
        "embed_dim": 2,
        "ddconfig": {
            "double_z": kl,
            "z_channels": 2,
            "resolution": 8,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 32,
            "ch_mult": [1, 2],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
        "lossconfig": {"target": "torch.nn.Identity"},
    }
    if not kl:
        p["n_embed"] = 16
    spec["first_stage"] = {"kind": kind, "params": p}
    ours = FrozenFirstStage(spec, 0.7)
    original = getattr(ae, kind)(**p).eval()
    original.load_state_dict(ours.state_dict(), strict=True)
    image = torch.randn(2, 3, 8, 8)
    z = torch.randn(2, 2, 4, 4)
    with torch.no_grad():
        ref = original.encode(image)
        if kl:
            ref = torch.cat([ref.mean, ref.logvar], dim=1)
        torch.testing.assert_close(ours.encode_parameters(image), ref, rtol=0, atol=0)
        torch.testing.assert_close(
            ours.decode(z), original.decode(z / 0.7), rtol=0, atol=0
        )
        report[kind] = {
            "encode_max_abs": float((ours.encode_parameters(image) - ref).abs().max()),
            "decode_max_abs": float(
                (ours.decode(z) - original.decode(z / 0.7)).abs().max()
            ),
        }
for arm in ["epsilon", "fourier_gaussian"]:
    torch.manual_seed(11)
    m = (
        Denoiser(
            spec,
            arm,
            {"mean": torch.randn(2, 4, 4) * 0.1, "power": torch.ones(2, 4, 4) * 0.6},
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        m.diffusion_model.out[-1].weight.normal_(std=0.01)

    class Bridge:
        device = torch.device("cuda")
        parameterization = "eps"

        def __init__(self, denoiser):
            self.denoiser = denoiser

        def __getattr__(self, name):
            return getattr(self.denoiser.schedule, name)

        def apply_model(self, z, t, c):
            return self.denoiser(z, t)

    bridge = Bridge(m)
    initial = torch.randn(1, 2, 4, 4).cuda()
    for eta in [0.0, 1.0]:
        g = torch.Generator().manual_seed(42)
        expected_gen = torch.Generator().manual_seed(42)
        ddim.noise_like = lambda shape, device, repeat, rng=expected_gen: torch.randn(
            shape, generator=rng
        ).to(device)
        oracle = ddim.DDIMSampler(bridge)
        expected, _ = oracle.sample(
            5, 1, [2, 4, 4], eta=eta, x_T=initial, verbose=False
        )
        actual, _ = sample_latents(
            m, 1, {"method": "ddim", "steps": 5, "eta": eta}, "cuda", g, initial
        )
        error = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        report[f"ddim_{arm}_eta{eta}"] = {"max_abs": error}
print(json.dumps(report, indent=2))
json_write(report, args.output)
temporary.cleanup()
