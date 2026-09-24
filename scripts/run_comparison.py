"""Run paired-seed ablations under one configured loss objective."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourier_score.config import load_config, apply_overrides, validate, experiment_name
from fourier_score.method import OBJECTIVES, COMPARISON_OBJECTIVES


def comparison_runs(config, changes=(), objectives=COMPARISON_OBJECTIVES, seeds=None):
    cfg = load_config(config, changes)
    if (
        not objectives
        or len(set(objectives)) != len(objectives)
        or any(x not in OBJECTIVES for x in objectives)
    ):
        raise ValueError("Choose distinct supported objectives")
    seeds = [cfg["seed"]] if seeds is None else seeds
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")
    runs = []
    outputs = set()
    for seed in seeds:
        for objective in objectives:
            arm_changes = ["loss.type=" + objective, f"seed={seed}"]
            if cfg["name"] != "auto" and "{parameterization}" not in cfg["name"]:
                arm_changes += ["name=" + cfg["name"] + f"_{objective}_s{seed}"]
            run_cfg = validate(apply_overrides(cfg, arm_changes))
            output = (
                Path(run_cfg["trainer"]["save_dir"]) / experiment_name(run_cfg)
            ).resolve()
            if output in outputs:
                raise ValueError(f"Comparison output collision: {output}")
            outputs.add(output)
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "train.py"),
                "-c",
                str(config),
            ]
            for option in [*changes, *arm_changes]:
                command += ["--set", option]
            runs.append({"config": run_cfg, "command": command, "output": output})
    return runs


def check_outputs(runs):
    # Check every destination before spending compute on the first run.
    for run in runs:
        path = run["output"]
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(
                f"Run exists: {path}; use a new name or resume it with train.py"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default="configs/mnist.json")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--device")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--parameterizations",
        "--objectives",
        dest="objectives",
        nargs="+",
        choices=OBJECTIVES,
        default=COMPARISON_OBJECTIVES,
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Paired training seeds; defaults to the config seed",
    )
    p.add_argument(
        "--download", action="store_true", help="Download MNIST/CIFAR-10 if missing"
    )
    a = p.parse_args()
    changes = list(a.set)
    if a.device:
        changes.append("device=" + a.device)
    if a.download:
        changes.append("data_loader.args.download=true")
    runs = comparison_runs(a.config, changes, a.objectives, a.seeds)
    if not a.dry_run:
        check_outputs(runs)
        # Prepare once so every arm starts with the same statistics cache.
        from fourier_score.data import build_data, prepare_stats

        cfg = runs[0]["config"]
        prepare_stats(cfg, build_data(cfg))
    for run in runs:
        print(json.dumps(run["command"]), flush=True)
        if not a.dry_run:
            subprocess.run(run["command"], check=True)


if __name__ == "__main__":
    main()
