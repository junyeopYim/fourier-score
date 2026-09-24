"""Train paired GMM baselines/gate candidates, select on validation, then test.

The default protocol has 99 training runs and 63 final test evaluations.
Checkpoints stay in --output; compact tables and figures go in --report.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
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
    gated_arm, make_bank, make_model, select_gate, tensor_state_hash, train_arm,
)
from fourier_score.utils import json_write, source_hash


def train_group(task):
    cfg, lam, seed, arms, output, provenance = task
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(True)
    family = MatchedMomentFamily(cfg, "gmm", lam)
    validation = make_bank(family, "validation")
    return [train_arm(family, arm, seed, validation, output, provenance) for arm in arms]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def verify_pairing(results, cfg, arms):
    expected = {(lam, seed, arm.name) for lam in cfg.spectrum_lambdas
                for seed in cfg.seeds for arm in arms}
    indexed = {(r["spectrum_lambda"], r["seed"], r["method"]): r for r in results}
    if len(results) != len(indexed) or set(indexed) != expected:
        raise ValueError("Missing or duplicate planned GMM runs")
    checks = []
    for lam in cfg.spectrum_lambdas:
        for seed in cfg.seeds:
            group = [indexed[(lam, seed, arm.name)] for arm in arms]
            if not all(r["completed"] and r["step"] == cfg.steps for r in group):
                raise ValueError("Incomplete GMM training")
            check = dict(spectrum_lambda=lam, seed=seed)
            for key in ("initial_backbone_sha256", "training_stream_first_batch_sha256", "final_data_rng_sha256"):
                if len({r[key] for r in group}) != 1:
                    raise ValueError(f"GMM pairing failed: {key}")
                check[key] = group[0][key]
            if len({r["validation"][-1]["bank_sha256"] for r in group}) != 1:
                raise ValueError("Unpaired validation banks")
            checks.append(check)
    return checks


def test_selected(cfg, output, results, arms):
    """Called only after selection.json has been written."""
    indexed = {(r["spectrum_lambda"], r["seed"], r["method"]): r for r in results}
    tested = []
    for lam in cfg.spectrum_lambdas:
        family = MatchedMomentFamily(cfg, "gmm", lam)
        bank = make_bank(family, "test")
        for seed in cfg.seeds:
            for arm in arms:
                folder = output / family.case_id / f"{arm.name}_seed{seed}"
                payload = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                model = make_model(family, arm, seed)
                model.load_state_dict(payload["ema"], strict=True)
                state = indexed[(lam, seed, arm.name)]
                if tensor_state_hash(model.backbone.state_dict()) != state["final_ema_backbone_sha256"]:
                    raise ValueError("Final EMA checkpoint mismatch")
                metric = evaluate_model(model, family, bank)
                json_write(metric, folder / "test.json")
                tested.append({**state, "test": metric})
    return tested


def export_report(report, cfg, provenance, results, tested, selection, pairing, arms,
                  *, figure_stem="gmm_gated_comparison"):
    report.mkdir(parents=True, exist_ok=True)
    per_seed, noise, frequency, curves = [], [], [], []
    for result in tested:
        metric = result["test"]
        common = {key: result[key] for key in ("spectrum_lambda", "seed", "method")}
        per_seed.append({**common, "steps": result["step"],
                         "score_error": metric["score_error"],
                         "unweighted_score_error": sum(r["unweighted_score_error"] * r["n"] for r in metric["per_noise"]) / metric["n_observations"],
                         "dsm_pixel_mean": metric["dsm_pixel_mean"],
                         "gradient_norm_mean": result["gradient_norm_sum"] / result["step"],
                         "gradient_clip_fraction": result["gradient_clip_count"] / result["step"],
                         "test_bank_sha256": metric["bank_sha256"]})
        noise.extend({**common, **row} for row in metric["per_noise"])
        frequency.extend({**common, "band": band, "score_error": value}
                         for band, value in enumerate(metric["score_error_by_frequency"]))
    for result in results:
        curves.extend({"spectrum_lambda": result["spectrum_lambda"], "seed": result["seed"],
                       "method": result["method"], "step": row["step"],
                       "validation_score_error": row["score_error"]}
                      for row in result["validation"])
    summary = []
    for lam in cfg.spectrum_lambdas:
        for arm in arms:
            rows = [r for r in per_seed if r["spectrum_lambda"] == lam and r["method"] == arm.name]
            row = dict(spectrum_lambda=lam, method=arm.name, n_seeds=len(rows))
            for metric in ("score_error", "unweighted_score_error"):
                row[metric + "_mean"], row[metric + "_sd"] = mean_sd([r[metric] for r in rows])
            summary.append(row)
        hashes = {r["test_bank_sha256"] for r in per_seed if r["spectrum_lambda"] == lam}
        if len(hashes) != 1:
            raise ValueError("Final comparisons must share a test bank")
    indexed = {(r["spectrum_lambda"], r["seed"], r["method"]): r for r in per_seed}
    paired = []
    for lam in cfg.spectrum_lambdas:
        for arm in arms[-2:]:
            for baseline in arms:
                if baseline.name == arm.name:
                    continue
                a = [indexed[(lam, seed, arm.name)]["score_error"] for seed in cfg.seeds]
                b = [indexed[(lam, seed, baseline.name)]["score_error"] for seed in cfg.seeds]
                delta = np.asarray(a) - np.asarray(b)
                avg, sd = mean_sd(delta)
                paired.append(dict(spectrum_lambda=lam, method=arm.name, baseline=baseline.name,
                                   paired_delta_mean=avg, paired_delta_sd=sd,
                                   relative_change_percent=float(100 * (np.mean(a) / np.mean(b) - 1)),
                                   wins=int((delta < 0).sum()), n_seeds=len(a)))
    for name, rows in (("summary", summary), ("per_seed", per_seed), ("paired_comparisons", paired),
                       ("noise_resolved", noise), ("frequency_resolved", frequency),
                       ("validation_curves", curves), ("validation_selection", selection["candidates"])):
        write_csv(report / f"{name}.csv", rows)
    payload = dict(config=asdict(cfg), provenance=provenance, selection=selection,
                   n_training_runs=len(results), n_test_runs=len(tested),
                   test_observations_per_spectrum=cfg.n_noise_levels * cfg.test_per_noise,
                   arms=[asdict(a) for a in arms], summary=summary, paired=paired, pairing_checks=pairing)
    json_write(payload, report / "summary.json")
    plot_report(report, cfg, arms, summary, noise, figure_stem=figure_stem)
    return payload


def plot_report(report, cfg, arms, summary, noise, *, figure_stem="gmm_gated_comparison"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Score / DSM", "Scalar / DSM", "Fourier / DSM", "Scalar / normalized",
              "Fourier / normalized", "Gated Scalar", "Gated Fourier"]
    colors = ["#777777", "#3178ad", "#d77b29", "#3178ad", "#d77b29", "#18866c", "#863daf"]
    styles = [":", "--", "--", "-", "-", "-", "-"]
    appearances = {arm.name: (label, color, style)
                   for arm, label, color, style in zip(BASELINE_ARMS, labels, colors, styles)}
    for arm in arms:
        if arm.gate_mode == "spectral_cap":
            scalar = arm.parameterization == "scalar_gaussian"
            appearances[arm.name] = ("Spectral " + ("Scalar" if scalar else "Fourier"),
                                     "#6e9b28" if scalar else "#111111", "-")
        if arm.gate_mode in ("log_sigma", "log_sigma_plateau"):
            scalar = arm.parameterization == "scalar_gaussian"
            plateau = arm.gate_mode == "log_sigma_plateau"
            appearances[arm.name] = (("Plateau " if plateau else "Gated ") + ("Scalar" if scalar else "Fourier"),
                                     ("#2382c0" if scalar else "#c13958") if plateau else
                                     ("#18866c" if scalar else "#863daf"), "-")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2 if len(arms) > 10 else 4.7))
    focus = max(cfg.spectrum_lambdas)
    for arm in arms:
        label, color, style = appearances[arm.name]
        rows = [r for r in summary if r["method"] == arm.name]
        axes[0].errorbar([r["spectrum_lambda"] for r in rows], [r["score_error_mean"] for r in rows],
                         yerr=[r["score_error_sd"] or 0 for r in rows], label=label,
                         color=color, ls=style, marker="o", capsize=3)
        rows = [r for r in noise if r["method"] == arm.name and r["spectrum_lambda"] == focus]
        xs = sorted({r["sigma"] for r in rows})
        stats = [mean_sd([r["score_error"] for r in rows if r["sigma"] == x]) for x in xs]
        means = np.array([s[0] for s in stats])
        sds = np.array([s[1] or 0 for s in stats])
        axes[1].plot(xs, means, color=color, ls=style, label=label, lw=1.8)
        axes[1].fill_between(xs, np.maximum(0, means-sds), means+sds, color=color, alpha=.10)
    axes[0].set(xlabel="Spectrum strength lambda (0 = flat)", ylabel="Test scaled true-score MSE",
                title=f"Final EMA after {cfg.steps:,} updates", xticks=cfg.spectrum_lambdas)
    axes[1].set(xlabel="Noise sigma", ylabel="Test scaled true-score MSE", xscale="log",
                title=f"Noise-resolved error: lambda = {focus:g}")
    axes[1].set_xticks([.12, .25, .5, 1., 2.5], labels=["0.12", "0.25", "0.5", "1", "2.5"])
    axes[1].minorticks_off()
    for ax in axes:
        ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=5 if len(arms) > 7 else 4, frameon=False)
    fig.tight_layout(rect=(0, .19 if len(arms) > 10 else .13, 1, 1))
    for extension in ("png", "svg", "pdf"):
        path = report / f"{figure_stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preset", choices=("smoke", "experiment"), default="experiment")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent single-threaded runs; each arm is scheduled independently")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--switches", type=float, nargs="+", default=[.5, 1., 1.5])
    parser.add_argument("--sharpness", type=float, default=4.)
    parser.add_argument("--bank-version", default="gmm-gated-v1")
    args = parser.parse_args(argv)
    if args.workers < 1 or not args.bank_version or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Choose positive workers, distinct nonnegative seeds, and a bank version")
    if len(set(args.switches)) != len(args.switches):
        parser.error("Gate switches must be distinct")
    cfg = GMMConfig(preset=args.preset, steps=5000, eval_every=500, width=192,
                    seeds=tuple(args.seeds), spectrum_lambdas=(0., .5, 1.), distributions=("gmm",),
                    n_noise_levels=9, val_per_noise=512, test_per_noise=2048,
                    cpu_threads=1, device="cpu", bank_version=args.bank_version)
    if args.preset == "smoke":
        cfg = replace(cfg, image_size=4, width=32, depth=2, steps=12, eval_every=6,
                      batch_size=32, n_noise_levels=3, val_per_noise=32, test_per_noise=64,
                      spectrum_lambdas=(0., 1.), seeds=tuple(args.seeds))
    arms = (*BASELINE_ARMS, *(gated_arm(cov, switch, args.sharpness)
                             for switch in args.switches for cov in ("scalar", "fourier")))
    cfg = replace(cfg, methods=tuple(arm.name for arm in arms), run_tag=args.output.name)
    provenance = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                      source_sha256=source_hash(), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)))
    output = args.output.resolve()
    protocol = dict(config=asdict(cfg), arms=[asdict(a) for a in arms], provenance=provenance,
                    selection="One common switch minimizes final validation mean over both covariances, all spectra and seeds; test only after selection")
    plan = output / "protocol.json"
    if plan.exists():
        previous = json.loads(plan.read_text())
        # Numerical training/evaluation lives in fourier_score and remains
        # guarded by source_sha256. A launcher-only scheduling change may
        # resume the same protocol; retain its original checkpoint provenance.
        current_invocation = dict(provenance)
        for key in ("git_revision", "git_dirty", "runner_sha256"):
            protocol["provenance"][key] = previous["provenance"][key]
        if json.loads(json.dumps(protocol)) != previous:
            raise ValueError("Existing comparison protocol differs; choose a new output directory")
    else:
        current_invocation = dict(provenance)
    json_write(protocol, plan)
    history_path = output / "execution_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(started_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers,
                        provenance=current_invocation))
    json_write(history, history_path)
    tasks = [(cfg, lam, seed, (arm,), output, provenance)
             for lam in cfg.spectrum_lambdas for seed in cfg.seeds for arm in arms]
    if args.workers == 1:
        groups = [train_group(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            groups = list(pool.map(train_group, tasks))
    results = [result for group in groups for result in group]
    pairing = verify_pairing(results, cfg, arms)
    selection = select_gate(results, args.switches)
    # Persist selection BEFORE creating or evaluating any test observations.
    json_write(selection, output / "selection.json")
    print("SELECTED " + json.dumps(selection), flush=True)
    final_arms = (*BASELINE_ARMS, *(gated_arm(cov, selection["sigma_switch"], args.sharpness)
                                  for cov in ("scalar", "fourier")))
    torch.set_num_threads(cfg.cpu_threads)
    tested = test_selected(cfg, output, results, final_arms)
    payload = export_report(args.report or output / "report", cfg, provenance, results, tested,
                            selection, pairing, final_arms)
    json_write(history, (args.report or output / "report") / "execution_history.json")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
