"""Paired, fixed-protocol GMM experiment for an exactly saturating noise gate.

The default experiment fixes sigma_switch=1.5, p=4 and transition [0.8, 1.0]
before evaluation. It compares five baselines, two sigmoid gates and two
plateau gates. Compatible baseline checkpoints can be audited and reused;
every method is evaluated on new common test observations.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fourier_score.gmm import (
    BASELINE_ARMS, GMMArm, GMMConfig, MatchedMomentFamily, evaluate_model,
    gated_arm, make_bank, make_model, plateau_arm, tensor_state_hash, train_arm,
)
from fourier_score.utils import json_write, source_hash
from scripts.run_gmm_comparison import export_report, mean_sd, verify_pairing, write_csv


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_arm(task):
    cfg, lam, seed, arm, output, reuse, provenance = task
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(True)
    family = MatchedMomentFamily(cfg, "gmm", lam)
    validation = make_bank(family, "validation")
    relative = Path(family.case_id) / f"{arm.name}_seed{seed}" / "checkpoint.pt"
    if reuse is not None:
        checkpoint = reuse / relative
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["state"]
        # These fields schedule runs or choose output names; all population,
        # optimizer, architecture, stream and validation settings must match.
        administrative = {"run_tag", "methods", "seeds", "spectrum_lambdas", "distributions"}
        old = {k: v for k, v in payload["config"].items() if k not in administrative}
        new = {k: v for k, v in asdict(cfg).items() if k not in administrative}
        if old != new or GMMArm(**state["arm"]) != arm:
            raise ValueError(f"Incompatible baseline configuration: {checkpoint}")
        if (not state["completed"] or state["step"] != cfg.steps or state["seed"] != seed
                or state["spectrum_lambda"] != lam or state["distribution"] != "gmm"):
            raise ValueError(f"Incomplete or mismatched baseline: {checkpoint}")
        model = make_model(family, arm, seed)
        if tensor_state_hash(model.backbone.state_dict()) != state["initial_backbone_sha256"]:
            raise ValueError(f"Baseline initialization differs: {checkpoint}")
        reference = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("reference.")}
        model.load_state_dict(payload["ema"], strict=True)
        if tensor_state_hash(model.backbone.state_dict()) != state["final_ema_backbone_sha256"]:
            raise ValueError(f"Baseline EMA digest differs: {checkpoint}")
        if any(not torch.equal(v, model.state_dict()[k]) for k, v in reference.items()):
            raise ValueError(f"Baseline statistics differ: {checkpoint}")
        reproduced = evaluate_model(model, family, validation)
        archived = state["validation"][-1]
        if (reproduced["bank_sha256"] != archived["bank_sha256"]
                or reproduced["score_error"] != archived["score_error"]
                or reproduced["per_noise"] != archived["per_noise"]):
            raise ValueError(f"Baseline validation is not exactly reproduced: {checkpoint}")
        origin = "reused; final EMA and every validation noise bin reproduced exactly"
        print(f"AUDITED {family.case_id} {arm.name} seed={seed}", flush=True)
    else:
        state = train_arm(family, arm, seed, validation, output, provenance)
        checkpoint = output / relative
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        origin = "trained"
    return {**state, "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
            "training_provenance": payload["provenance"], "origin": origin}


def evaluate_group(task):
    cfg, lam, seed, results, output, transition_sigmas = task
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(True)
    family = MatchedMomentFamily(cfg, "gmm", lam)
    bank = make_bank(family, "test")
    # A separate RNG namespace and grid for the transition diagnostic. Its
    # observations never enter the nine-bin primary mean.
    transition_bank = None
    if lam == max(cfg.spectrum_lambdas):
        diagnostic = MatchedMomentFamily(replace(cfg, bank_version=cfg.bank_version + "-transition",
                                                 test_per_noise=min(1024, cfg.test_per_noise)), "gmm", lam)
        transition_bank = make_bank(diagnostic, "test", sigmas=transition_sigmas)
    tested = []
    for state in results:
        if state["spectrum_lambda"] != lam or state["seed"] != seed:
            continue
        if file_hash(state["checkpoint"]) != state["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during comparison")
        payload = torch.load(state["checkpoint"], map_location="cpu", weights_only=True)
        arm = GMMArm(**state["arm"])
        model = make_model(family, arm, seed)
        model.load_state_dict(payload["ema"], strict=True)
        if tensor_state_hash(model.backbone.state_dict()) != state["final_ema_backbone_sha256"]:
            raise ValueError("Test EMA differs from final training EMA")
        metric = evaluate_model(model, family, bank)
        folder = output / "evaluation" / family.case_id / f"{arm.name}_seed{seed}"
        json_write(metric, folder / "test.json")
        row = {**state, "test": metric}
        if transition_bank is not None:
            row["transition_test"] = evaluate_model(model, family, transition_bank)
            json_write(row["transition_test"], folder / "transition_test.json")
        tested.append(row)
    print(f"TESTED {family.case_id} seed={seed} methods={len(tested)}", flush=True)
    return tested


def export_diagnostics(report, tested, cfg, arms):
    regions, transition = [], []
    for result in tested:
        common = {key: result[key] for key in ("spectrum_lambda", "seed", "method")}
        for region, lo, hi in (("low", 0., .3), ("middle", .3, 1.), ("high", 1., float("inf"))):
            bins = [r for r in result["test"]["per_noise"] if lo <= r["sigma"] < hi]
            if bins:
                n = sum(r["n"] for r in bins)
                regions.append({**common, "region": region, "n": n,
                                "score_error": sum(r["n"] * r["score_error"] for r in bins) / n})
        if "transition_test" in result:
            transition.extend({**common, **row} for row in result["transition_test"]["per_noise"])
    summary = []
    for lam in cfg.spectrum_lambdas:
        for arm in arms:
            for region in ("low", "middle", "high"):
                values = [r["score_error"] for r in regions if r["spectrum_lambda"] == lam
                          and r["method"] == arm.name and r["region"] == region]
                if values:
                    mean, sd = mean_sd(values)
                    summary.append(dict(spectrum_lambda=lam, method=arm.name, region=region,
                                        score_error_mean=mean, score_error_sd=sd, n_seeds=len(values)))
    write_csv(report / "noise_regions_per_seed.csv", regions)
    write_csv(report / "noise_regions.csv", summary)
    write_csv(report / "transition_resolved.csv", transition)
    audit = [{key: r[key] for key in ("spectrum_lambda", "seed", "method", "origin", "checkpoint",
              "checkpoint_sha256", "initial_backbone_sha256", "training_stream_first_batch_sha256",
              "final_data_rng_sha256", "final_ema_backbone_sha256", "training_provenance")} for r in tested]
    json_write(audit, report / "checkpoint_audit.json")
    return summary, transition


def plot_transition(report, cfg, arms, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    family = MatchedMomentFamily(cfg, "gmm", max(cfg.spectrum_lambdas))
    selected = [arms[4], arms[6], arms[8]]
    labels = ["Fourier / normalized", "Gated Fourier", "Plateau Fourier"]
    colors = ["#d77b29", "#863daf", "#c13958"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sigma = torch.logspace(np.log10(.1), np.log10(3.), 400)
    for arm, label, color in zip(selected, labels, colors):
        model = make_model(family, arm, cfg.seeds[0])
        gate = model.reference.gate_value(sigma).flatten().numpy()
        axes[0].plot(sigma.numpy(), gate, label=label, color=color)
        group = [r for r in rows if r["method"] == arm.name]
        xs = sorted({r["sigma"] for r in group})
        stats = [mean_sd([r["score_error"] for r in group if r["sigma"] == x]) for x in xs]
        means = np.array([s[0] for s in stats])
        sd = np.array([s[1] or 0 for s in stats])
        axes[1].plot(xs, means, label=label, color=color, marker=".")
        axes[1].fill_between(xs, means-sd, means+sd, color=color, alpha=.12)
    for ax in axes:
        ax.axvspan(arms[-1].sigma_lo, arms[-1].sigma_hi, color="#777777", alpha=.1)
        ax.grid(alpha=.18)
        ax.set_xlabel("Noise sigma")
    axes[0].set(xscale="log", ylabel="Gaussian reference gate g", title="Fixed gate before training")
    axes[1].set(ylabel="Test scaled true-score MSE", title=f"Transition diagnostic: lambda = {family.lam:g}")
    axes[0].set_xticks([.1, .3, .8, 1., 3.], labels=["0.1", "0.3", "0.8", "1", "3"])
    axes[0].minorticks_off()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for extension in ("png", "svg", "pdf"):
        path = report / f"gmm_plateau_transition.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def parallel_map(function, tasks, workers):
    if workers == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(function, tasks))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reuse-baselines", type=Path,
                        help="Read-only reuse after matching configs, EMA hashes and validation outputs")
    parser.add_argument("--preset", choices=("smoke", "experiment"), default="experiment")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--sigma-switch", type=float, default=1.5)
    parser.add_argument("--sharpness", type=float, default=4.)
    parser.add_argument("--sigma-lo", type=float, default=.8)
    parser.add_argument("--sigma-hi", type=float, default=1.)
    parser.add_argument("--test-bank-version", default="gmm-plateau-v1")
    args = parser.parse_args(argv)
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Choose positive workers and distinct nonnegative seeds")
    if not args.test_bank_version or args.test_bank_version == "gmm-gated-v1":
        parser.error("Choose an independent test bank version")
    cfg = GMMConfig(preset=args.preset, steps=5000, eval_every=500, width=192,
                    seeds=tuple(args.seeds), spectrum_lambdas=(0., .5, 1.), distributions=("gmm",),
                    n_noise_levels=9, val_per_noise=512, test_per_noise=2048,
                    cpu_threads=1, device="cpu", bank_version="gmm-gated-v1")
    if args.preset == "smoke":
        cfg = replace(cfg, image_size=4, width=32, depth=2, steps=12, eval_every=6,
                      batch_size=32, n_noise_levels=3, val_per_noise=32, test_per_noise=64,
                      spectrum_lambdas=(0., 1.))
    if not cfg.sigma_min <= args.sigma_lo < args.sigma_hi <= cfg.sigma_max:
        parser.error("Transition must lie within the training sigma range")
    arms = (*BASELINE_ARMS,
            *(gated_arm(cov, args.sigma_switch, args.sharpness) for cov in ("scalar", "fourier")),
            *(plateau_arm(cov, args.sigma_switch, args.sharpness, args.sigma_lo, args.sigma_hi)
              for cov in ("scalar", "fourier")))
    cfg = replace(cfg, methods=tuple(arm.name for arm in arms), run_tag=args.output.name)
    # Include both exact endpoints and a dense interior, plus nearby context.
    transition_sigmas = sorted(set([max(cfg.sigma_min, args.sigma_lo * .8),
                                   max(cfg.sigma_min, args.sigma_lo * .95),
                                   *np.linspace(args.sigma_lo, args.sigma_hi, 11).tolist(),
                                   min(cfg.sigma_max, args.sigma_hi * 1.05),
                                   min(cfg.sigma_max, args.sigma_hi * 1.2)]))
    output = args.output.resolve()
    report = (args.report or output / "report").resolve()
    reuse = args.reuse_baselines.resolve() if args.reuse_baselines else None
    if reuse == output:
        parser.error("Reuse checkpoints must be outside the new output directory")
    provenance = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                      source_sha256=source_hash(), runner_sha256=file_hash(__file__),
                      reporting_helpers_sha256=file_hash(ROOT / "scripts/run_gmm_comparison.py"),
                      git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)))
    selection = dict(criterion="Fixed proposal before training; no threshold search or test selection",
                     sigma_switch=args.sigma_switch, sharpness=args.sharpness,
                     sigma_lo=args.sigma_lo, sigma_hi=args.sigma_hi,
                     candidates=[dict(sigma_switch=args.sigma_switch, sharpness=args.sharpness,
                                      sigma_lo=args.sigma_lo, sigma_hi=args.sigma_hi, selection="fixed")])
    protocol = dict(config=asdict(cfg), arms=[asdict(a) for a in arms], provenance=provenance,
                    selection=selection, test_bank_version=args.test_bank_version,
                    transition_sigmas=transition_sigmas, reuse_baselines=str(reuse) if reuse else None)
    plan = output / "protocol.json"
    invocation = dict(provenance)
    if plan.exists():
        previous = json.loads(plan.read_text())
        for key in ("git_revision", "git_dirty"):
            provenance[key] = previous["provenance"][key]
        if json.loads(json.dumps(protocol)) != previous:
            raise ValueError("Existing plateau protocol differs; choose a new output directory")
    json_write(protocol, plan)
    json_write(selection, output / "selection.json")
    history_path = output / "execution_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(started_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers,
                        provenance=invocation))
    json_write(history, history_path)
    # Schedule new training first, allowing baseline audits to fill spare cores.
    ordered_arms = (*arms[-2:], *arms[:-2])
    tasks = [(cfg, lam, seed, arm, output,
              reuse if arm.gate_mode != "log_sigma_plateau" else None, provenance)
             for arm in ordered_arms for lam in cfg.spectrum_lambdas for seed in cfg.seeds]
    results = parallel_map(prepare_arm, tasks, args.workers)
    pairing = verify_pairing(results, cfg, arms)
    test_cfg = replace(cfg, bank_version=args.test_bank_version)
    tasks = [(test_cfg, lam, seed, results, output, transition_sigmas)
             for lam in cfg.spectrum_lambdas for seed in cfg.seeds]
    tested = [r for group in parallel_map(evaluate_group, tasks, args.workers) for r in group]
    torch.set_num_threads(1)
    payload = export_report(report, cfg, provenance, results, tested, selection, pairing, arms,
                            figure_stem="gmm_plateau_comparison")
    regions, transition = export_diagnostics(report, tested, cfg, arms)
    payload.update(n_training_runs=sum(r["origin"] == "trained" for r in results),
                   n_reused_runs=sum(r["origin"] != "trained" for r in results),
                   test_bank_version=args.test_bank_version,
                   transition_diagnostic=dict(spectrum_lambda=max(cfg.spectrum_lambdas),
                                              sigmas=transition_sigmas,
                                              n_per_noise=min(1024, cfg.test_per_noise),
                                              bank_version=args.test_bank_version + "-transition",
                                              included_in_primary_mean=False),
                   noise_regions=regions)
    json_write(payload, report / "summary.json")
    json_write(protocol, report / "protocol.json")
    json_write(history, report / "execution_history.json")
    plot_transition(report, cfg, arms, transition)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
