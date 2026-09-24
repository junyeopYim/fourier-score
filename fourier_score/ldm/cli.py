"""An explicit LDM CLI keeps the existing pixel-space v1 artifacts compatible."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from fourier_score.parse_config import CustomArgs, add_options, cli_overrides, from_args
from fourier_score.provenance import file_sha256
from fourier_score.utils import ROOT, configure_runtime, json_write, load_checkpoint

from .config import (
    PARAMETERIZATIONS,
    experiment_name,
    load_config,
    load_spec,
    override,
    validate,
)


DEFAULT_CONFIG = ROOT / "configs/ldm/ffhq.json"
DEVICE = CustomArgs(["--device"], "device")
# Per-command flags that expand into --set entries after DEVICE.
OPTIONS = {
    "train": [
        CustomArgs(
            ["--parameterization"], "parameterization", choices=PARAMETERIZATIONS
        )
    ],
    "sample": [
        CustomArgs(["--num-samples"], "sampling.num_samples", type=int),
        CustomArgs(["--batch-size"], "sampling.batch_size", type=int),
        CustomArgs(["--steps"], "sampling.steps", type=int),
    ],
}


def resumed(path, changes):
    """train: the checkpoint's config with ``changes``."""
    state = load_checkpoint(path)
    return validate(override(state["config"], changes)), state


def trained(path, changes):
    """sample/evaluate: ``load_trained``'s (model, cfg, device, state)."""
    from .training import load_trained

    return load_trained(path, changes)


def config_args(parser):
    parser.add_argument("-c", "--config")
    parser.add_argument("--set", action="append", default=[])
    return add_options(parser, [DEVICE])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "inspect", "train", "compare"):
        p = config_args(commands.add_parser(command))
        p.add_argument("--dry-run", action="store_true")
        if command == "train":
            p.add_argument("-r", "--resume")
            add_options(p, OPTIONS["train"])
        if command == "compare":
            p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
            p.add_argument(
                "--parameterizations",
                nargs="+",
                choices=PARAMETERIZATIONS,
                default=["epsilon", "scalar_gaussian", "fourier_gaussian"],
            )
    p = config_args(commands.add_parser("sample"))
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("-r", "--resume")
    group.add_argument("--pretrained", action="store_true")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--weights", choices=("ema", "raw"), default="ema")
    add_options(p, OPTIONS["sample"])
    p = config_args(commands.add_parser("evaluate"))
    p.add_argument("-r", "--resume", required=True)
    p.add_argument("-o", "--output", required=True)
    for command in ("reconstruct", "export-real"):
        p = config_args(commands.add_parser(command))
        p.add_argument("-o", "--output", required=True)
        p.add_argument(
            "--split",
            choices=("train", "validation"),
            default="train" if command == "export-real" else "validation",
        )
    p = commands.add_parser("fid")
    p.add_argument("--real", required=True)
    p.add_argument("--generated", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.command == "fid":
        from fourier_score.evaluation import evaluate_images

        if Path(args.output).exists():
            raise FileExistsError(args.output)
        result = evaluate_images(
            args.real,
            args.generated,
            args.output,
            args.device,
            args.batch_size,
            inception_score=False,
        )
        print(json.dumps(result, indent=2))
        return
    resume = getattr(args, "resume", None)
    inference = bool(resume) and args.command in ("sample", "evaluate")
    loaded = from_args(
        parser,
        args,
        [DEVICE, *OPTIONS.get(args.command, [])],
        load_config,
        DEFAULT_CONFIG,
        trained if inference else resumed,
        "--resume uses the checkpoint config; do not also supply --config",
    )
    if inference:
        model, cfg, device, state = loaded
        spec = state["spec"]
    else:
        cfg, state = loaded
        spec = load_spec(cfg)
    if getattr(args, "dry_run", False) or args.command == "inspect":
        if args.command == "compare":
            print(json.dumps(comparison_commands(args, cfg), indent=2))
        else:
            print(
                json.dumps(
                    {
                        "config": cfg,
                        "spec": spec,
                        "run": experiment_name(cfg),
                        "note": "upstream uses the released loss; l2 uses L2/DSM.",
                    },
                    indent=2,
                )
            )
        return
    if args.command in ("prepare", "compare", "reconstruct", "export-real") or (
        args.command == "sample" and not resume
    ):
        device = configure_runtime(cfg)
    if args.command == "prepare":
        from .data import prepare_cache

        print(json.dumps(prepare_cache(cfg, spec, device), indent=2))
    elif args.command == "train":
        from .training import Trainer

        print(Trainer(cfg, spec, state).train())
    elif args.command == "compare":
        jobs = comparison_commands(args, cfg)
        for job in jobs:
            out = Path(job["run_dir"])
            if out.exists() and any(out.iterdir()):
                raise FileExistsError(f"Comparison run already exists: {out}")
        from .data import prepare_cache

        prepare_cache(cfg, spec, device)
        for job in jobs:
            subprocess.run(job["argv"], check=True)
    elif args.command == "sample":
        from .evaluation import generate
        from .first_stage import load_first_stage
        from .model import load_public_denoiser

        if resume:
            expected = state["cache"]["first_stage"]["checkpoint_sha256"]
            provenance = {
                "weights": "EMA",
                "checkpoint_sha256": file_sha256(resume),
                "step": state["step"],
                "training_wall_seconds": state["training_wall_seconds"],
                "optimizer_wall_seconds": state["optimizer_wall_seconds"],
                "cache_identity": state["cache"]["identity"],
            }
            if args.weights != "ema":
                parser.error("Trained experiment sampling always evaluates EMA weights")
        else:
            if cfg["parameterization"] != "epsilon":
                parser.error(
                    "Public pretrained sampling requires parameterization=epsilon"
                )
            expected = None
            model = load_public_denoiser(
                cfg["first_stage"]["checkpoint"], spec, args.weights, device
            )
            provenance = {
                "weights": args.weights,
                "reference": "External pretrained LDM reference",
            }
        stage, first = load_first_stage(
            cfg["first_stage"]["checkpoint"], spec, device, expected
        )
        provenance["first_stage"] = first
        generate(model, stage, cfg, args.output, provenance)
    elif args.command == "evaluate":
        from .data import open_cache
        from .evaluation import evaluate_latents

        if Path(args.output).exists():
            raise FileExistsError(args.output)
        cache = open_cache(cfg, spec)
        if cache["identity"] != state["cache"]["identity"]:
            raise ValueError("Evaluation cache differs from the training cache")
        result = evaluate_latents(model, cfg, cache, device)
        result.update(
            step=state["step"],
            weights="EMA",
            cache_identity=cache["identity"],
            training_wall_seconds=state["training_wall_seconds"],
            optimizer_wall_seconds=state["optimizer_wall_seconds"],
        )
        json_write(result, args.output)
        print(json.dumps(result, indent=2))
    elif args.command in ("reconstruct", "export-real"):
        from fourier_score.images import write_png

        from .data import image_splits
        from .evaluation import empty_output, reconstruct
        from .first_stage import load_first_stage

        ds = image_splits(cfg, spec)[args.split]
        provenance = {
            "split": args.split,
            "list_sha256": file_sha256(cfg["data"][args.split + "_list"]),
            "file_metadata_sha256": ds.fingerprint,
            "preprocessing": cfg["data"]["preprocessing"],
        }
        if args.command == "reconstruct":
            stage, first = load_first_stage(
                cfg["first_stage"]["checkpoint"], spec, device
            )
            provenance["first_stage"] = first
            print(
                json.dumps(
                    reconstruct(stage, ds, cfg, args.output, provenance), indent=2
                )
            )
        else:
            out = empty_output(args.output)
            (out / "png").mkdir()
            json_write({"complete": False, **provenance}, out / "settings.json")
            for i in range(len(ds)):
                # Recover the preprocessed real uint8 image without truncation error.
                image = (
                    ((ds[i] + 1) * 127.5)
                    .round()
                    .clamp(0, 255)
                    .byte()
                    .permute(1, 2, 0)
                    .numpy()
                )
                write_png(image, out / "png" / f"{i:07d}.png")
            json_write(
                {"complete": True, "num_images": len(ds), **provenance},
                out / "settings.json",
            )


def comparison_commands(args, cfg):
    if len(set(args.seeds)) != len(args.seeds) or len(
        set(args.parameterizations)
    ) != len(args.parameterizations):
        raise ValueError("Comparison seeds and parameterizations must be unique")
    jobs = []
    for seed in args.seeds:
        for parameterization in args.parameterizations:
            arm = copy.deepcopy(cfg)
            arm.update(seed=seed, parameterization=parameterization)
            if "{parameterization}" not in arm["name"] or "{seed}" not in arm["name"]:
                arm["name"] += "_{parameterization}_s{seed}"
            arm = validate(arm)
            name = experiment_name(arm)
            argv = [
                sys.executable,
                str(ROOT / "ldm.py"),
                "train",
                "-c",
                str(Path(args.config or DEFAULT_CONFIG).resolve()),
            ]
            for change in [
                *cli_overrides(args, [DEVICE]),
                f"seed={seed}",
                f"parameterization={parameterization}",
                "name=" + name,
            ]:
                argv.extend(("--set", change))
            jobs.append(
                {"argv": argv, "run_dir": str(Path(arm["training"]["save_dir"]) / name)}
            )
    return jobs
