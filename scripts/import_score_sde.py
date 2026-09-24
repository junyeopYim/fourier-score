"""Convert a verified official VE NCSN++ EMA to this repository's inference format."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fourier_score.config import load_config
from fourier_score.model.model import architecture_report, build_model
from fourier_score.trainer.checkpoints import FORMAT
from fourier_score.utils import ROOT, atomic_save, environment, load_checkpoint, source_hash
from scripts.download_ldm import file_info
from scripts.download_score_sde import MODELS, download_plan, verify_bundle


def load_official_ema(state, cfg):
    """Match every original state key and the original trainable-parameter order."""
    if (cfg["loss"]["type"] != "score" or cfg["process"]["type"] != "ve"
            or cfg["arch"]["args"]["embedding_type"] != "fourier"):
        raise ValueError("Official VE checkpoints require the score parameterization")
    model = build_model(cfg, stats=None, device="cpu")
    original = state["model"]
    prefixed = [name.startswith("module.") for name in original]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed DataParallel key prefixes")
    weights = {(name[7:] if all(prefixed) else name): value for name, value in original.items()}
    # Released files can omit this deterministic buffer. Continuous Fourier
    # conditioning uses sigma directly; no learned tensor is reconstructed.
    weights.setdefault("sigmas", model.backbone.sigmas.detach().clone())
    model.backbone.load_state_dict(weights, strict=True)
    parameters = {name: value for name, value in model.backbone.named_parameters() if value.requires_grad}
    ordered = [name for name in weights if name in parameters]
    if ordered != list(parameters):
        raise ValueError("Original parameter order differs; refusing ambiguous EMA mapping")
    shadow = state["ema"]["shadow_params"]
    if len(shadow) != len(parameters):
        raise ValueError("EMA parameter count mismatch")
    with torch.no_grad():
        for (name, parameter), value in zip(parameters.items(), shadow):
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid EMA tensor: {name}")
            parameter.copy_(value)
    # Fixed Fourier embedding weights and the sigma buffer come from the raw state.
    model.eval()
    return model


def convert_model(model_name, input_dir="pretrained/score_sde", output=None):
    plan = download_plan(model_name, input_dir)
    bundle = Path(plan["destination"])
    manifest = verify_bundle(bundle, plan)
    cfg = load_config(ROOT / "configs/score_sde" / f"{model_name}.json", ["device=cpu"])
    checkpoint = bundle / plan["checkpoint_name"]
    target = Path(output) if output else bundle / "ema.pt"
    if target.resolve() == checkpoint.resolve():
        raise ValueError("Output must not replace the original checkpoint")
    provenance = {
        "kind": "official-score-sde-reference",
        "model": model_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["files"][plan["checkpoint_name"]]["sha256"],
        "upstream_revision": plan["upstream_revision"],
        "upstream_folder": plan["official_folder"],
        "converter_sha256": file_info(Path(__file__))["sha256"],
        "weights": "EMA",
        "scope": "External pretrained reference with the local sampler",
        "buffer_policy": "Missing sigmas buffers are restored from config; continuous Fourier conditioning uses sigma directly",
    }
    if target.exists():
        existing = load_checkpoint(target)
        if (existing.get("format") != FORMAT or existing.get("kind") != "ema"
                or existing.get("pretrained_source") != provenance or existing.get("config") != cfg):
            raise ValueError(f"Incompatible existing export: {target}; choose a new --output")
        return target
    torch.set_num_threads(cfg["backend"]["cpu_threads"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = load_official_ema(state, cfg)
    report = architecture_report(model.backbone)
    result = {
        "format": FORMAT, "kind": "ema", "config": cfg,
        "step": int(state["step"]), "model": model.state_dict(), "stats": None,
        "source_sha256": source_hash(), "environment": environment(torch.device("cpu"), cfg),
        "pretrained_source": provenance, "training_wall_seconds": None,
    }
    atomic_save(result, target)
    print(json.dumps({"checkpoint": str(target), "step": result["step"],
                      "parameters": report["all_parameters"], "weights": "EMA"}, indent=2))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("pretrained/score_sde"))
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    print(f"Ready: {convert_model(args.model, args.input_dir, args.output)}")


if __name__ == "__main__":
    main()
