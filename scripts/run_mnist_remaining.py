"""Train missing MNIST comparisons, reuse completed controls, and evaluate EMA.

Defaults: the 19 GMM comparison arms, seed 0, 100K updates, one GPU job at a
time. Existing runs are read-only; new runs live in a separate study directory.
Use --dry-run to inspect the queue without starting training or evaluation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourier_score.checkpoints import FORMAT, resume_signature
from fourier_score.config import apply_overrides, experiment_name, load_config, validate
from fourier_score.gmm import (
    BASELINE_ARMS, gated_arm, log_gate_arm, plateau_arm, shaped_gate_arm, spectral_cap_arm,
)
from fourier_score.utils import json_write, load_checkpoint, source_hash


ARMS = (
    *BASELINE_ARMS,
    *(gated_arm(cov, 1.5, 4.) for cov in ("scalar", "fourier")),
    *(plateau_arm(cov) for cov in ("scalar", "fourier")),
    *(spectral_cap_arm(cov) for cov in ("scalar", "fourier")),
    *(shaped_gate_arm(cov, mode) for mode in ("linear_sigma", "tanh_sigma")
      for cov in ("scalar", "fourier")),
    *(log_gate_arm(cov, mode) for mode in ("linear_log_sigma", "bounded_log_sigmoid")
      for cov in ("scalar", "fourier")),
)


def run_config(base, arm, seed, output, steps, device):
    cfg = apply_overrides(base, [f"seed={seed}", "name=auto", f"device={device}",
                                f"trainer.iterations={steps}", f"trainer.save_dir={output}",
                                f"loss.type={arm.parameterization}", f"loss.objective={arm.objective}"])
    cfg["fourier"]["gate"] = arm.gate
    return validate(cfg)


def compatible(cfg, other):
    try:
        return resume_signature(cfg) == resume_signature(validate(other))
    except (KeyError, TypeError, ValueError):
        # The search roots can also contain latent or legacy configurations.
        return False


def completed_checkpoint(folder, cfg, steps):
    """A folder name or an old log alone never establishes a completed run."""
    config = folder / "config.resolved.json"
    if not config.is_file() or not compatible(cfg, json.loads(config.read_text())):
        return None
    for path in (folder / f"ema_{steps:09d}.pt", folder / "last.pt"):
        if not path.is_file():
            continue
        state = load_checkpoint(path)
        if (state.get("format") == FORMAT and state.get("kind") in ("ema", "training")
                and state.get("step") == steps and compatible(cfg, state["config"])):
            return path
    return None


def plan_run(cfg, roots, steps):
    output = Path(cfg["trainer"]["save_dir"]) / experiment_name(cfg)
    candidates = [output]
    for root in roots:
        candidates.extend(sorted(p.parent for p in root.glob("*/config.resolved.json")))
    for folder in dict.fromkeys(candidates):
        checkpoint = completed_checkpoint(folder, cfg, steps)
        if checkpoint is not None:
            return dict(action="reuse", checkpoint=checkpoint, output=output, config=cfg)
    last = output / "last.pt"
    if last.is_file():
        state = load_checkpoint(last)
        if (state.get("format") != FORMAT or state.get("kind") != "training"
                or not 0 <= state.get("step", -1) < steps or not compatible(cfg, state["config"])
                or state.get("source_sha256") != source_hash()):
            raise ValueError(f"Cannot resume incompatible checkpoint: {last}; choose a new --output")
        return dict(action="resume", checkpoint=last, output=output, config=cfg)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Incomplete run without last.pt: {output}; preserve it and use a new --output")
    return dict(action="train", checkpoint=None, output=output, config=cfg)


def command(script, *args):
    argv = [sys.executable, str(ROOT / script), *map(str, args)]
    print(json.dumps(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_real_images(base, output):
    real = output / "real_validation"
    settings = real / "settings.json"
    if settings.is_file():
        previous = json.loads(settings.read_text())
        keys = ("dataset", "root", "image_size", "channels", "centered", "random_flip",
                "validation_size", "split_seed", "crop_size")
        old = previous["config"]["data_loader"]["args"]
        expected = base["data_loader"]["args"]
        if (previous.get("effective_split") != "validation"
                or previous.get("count") != expected["validation_size"]
                or any(old[k] != expected[k] for k in keys)
                or len(list((real / "png").glob("*.png"))) != previous["count"]):
            raise ValueError(f"Incompatible real-image export: {real}")
    else:
        config = output / "real_config.json"
        json_write(base, config)
        command("scripts/export_real.py", "-c", config, "--split", "validation", "-o", real)
    return real / "png"


def ensure_samples(checkpoint, cfg, folder, args):
    expected = {**cfg["sampling"], "num_samples": args.samples,
                "batch_size": args.sample_batch, "steps": args.sampling_steps}
    settings = folder / "settings.json"
    if settings.is_file():
        previous = json.loads(settings.read_text())
        if (previous.get("complete") and previous.get("checkpoint_sha256") == file_hash(checkpoint)
                and previous.get("environment", {}).get("source_sha256") == source_hash()
                and previous.get("config", {}).get("sampling") == expected
                and previous.get("num_samples") == args.samples
                and len(list((folder / "png").glob("*.png"))) == args.samples):
            return
    if folder.exists() and any(folder.iterdir()):
        # Preserve incomplete or differently configured samples before retrying.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = folder.with_name(folder.name + ".previous-" + stamp)
        folder.rename(archived)
        print(f"Preserved previous samples: {archived}", flush=True)
    options = []
    for key, value in expected.items():
        options += ["--set", f"sampling.{key}={json.dumps(value)}"]
    command("sample.py", "-r", checkpoint, "-o", folder, "--device", args.device, *options)


def write_summary(rows, output):
    json_write(rows, output / "summary.json")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mnist.json")
    parser.add_argument("--output", type=Path, default=ROOT / "saved/mnist_remaining_100k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--only", nargs="+", choices=[a.name for a in ARMS])
    parser.add_argument("--reuse-roots", type=Path, nargs="+",
                        default=[ROOT / "saved", ROOT / "saved/recovered"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--with-fid", action="store_true", help="Also generate images and evaluate FID/IS")
    parser.add_argument("--eval-images", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--sample-batch", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (min(args.steps, args.eval_images, args.samples, args.sample_batch, args.sampling_steps) < 1
            or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Use positive budgets and distinct nonnegative seeds")
    changes = ["data_loader.args.download=true"] if args.download else []
    base = load_config(args.config, changes)
    if base["data_loader"]["args"]["dataset"] != "mnist" or base["data_loader"]["args"]["validation_size"] < 2:
        parser.error("This runner requires MNIST with a held-out validation split")
    output = args.output.resolve()
    arms = [a for a in ARMS if args.only is None or a.name in args.only]
    runs = []
    for seed in args.seeds:
        for arm in arms:
            cfg = run_config(base, arm, seed, output, args.steps, args.device)
            run = plan_run(cfg, [p.resolve() for p in args.reuse_roots], args.steps)
            run.update(method=arm.name, seed=seed)
            runs.append(run)
            print(f"{run['action'].upper():6} seed={seed} {arm.name}: {run['checkpoint'] or run['output']}", flush=True)
    print(f"Queue: {sum(r['action'] == 'train' for r in runs)} new, "
          f"{sum(r['action'] == 'resume' for r in runs)} resume, "
          f"{sum(r['action'] == 'reuse' for r in runs)} completed; "
          f"{args.steps} updates each; DSM + {'FID' if args.with_fid else 'no sampling'}", flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        json_write(dict(source_sha256=source_hash(), with_fid=args.with_fid,
                        runs=[{**r, "checkpoint": str(r["checkpoint"]) if r["checkpoint"] else None,
                               "output": str(r["output"])} for r in runs]), output / "plan.json")
        rows = []
        real = ensure_real_images(base, output) if args.with_fid else None
        # Train/evaluate new models first; existing controls are reevaluated last.
        for run in sorted(runs, key=lambda r: r["action"] == "reuse"):
            cfg = run["config"]
            if run["action"] == "train":
                config_path = output / "configs" / f"{run['method']}_s{run['seed']}.json"
                json_write(cfg, config_path)
                command("train.py", "-c", config_path)
            elif run["action"] == "resume":
                command("train.py", "-r", run["checkpoint"], "--device", args.device,
                        "--set", f"trainer.iterations={args.steps}")
            checkpoint = (run["checkpoint"] if run["action"] == "reuse" else
                          run["output"] / f"ema_{args.steps:09d}.pt")
            result = output / "evaluation" / f"{run['method']}_s{run['seed']}"
            command("evaluate.py", "dsm", "-r", checkpoint, "-o", result / "dsm.json",
                    "--device", args.device, "--set", f"evaluation.max_images={args.eval_images}",
                    "--set", "evaluation.batch_size=128", "--set", "evaluation.seed=17001",
                    "--set", "evaluation.noise_bins=20", "--set", "evaluation.frequency_bins=4")
            metrics = json.loads((result / "dsm.json").read_text())
            fid = None
            if args.with_fid:
                samples = result / f"samples{args.samples}"
                ensure_samples(checkpoint, cfg, samples, args)
                command("evaluate.py", "fid", "--real", real, "--generated", samples / "png",
                        "--device", args.device, "--batch-size", 64, "-o", result / "fid.json")
                fid = json.loads((result / "fid.json").read_text())["metrics"]["frechet_inception_distance"]
            rows.append(dict(method=run["method"], seed=run["seed"], step=metrics["step"],
                             origin=run["action"], checkpoint=str(checkpoint),
                             checkpoint_sha256=file_hash(checkpoint),
                             training_source_sha256=load_checkpoint(checkpoint)["source_sha256"],
                             evaluation_source_sha256=source_hash(),
                             dsm_pixel_mean=metrics["dsm_pixel_mean"],
                             dsm_standard_error=metrics["standard_error"], fid=fid))
            write_summary(rows, output)


if __name__ == "__main__":
    main()
