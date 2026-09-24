"""Fixed GMM experiment: one backbone, one normalized MSE, spectral gate.

Fix delta=0.5 and a log-noise transition [1,2] before training. Compare to
the five original arms, sigmoid gates and the earlier [0.8,1] plateau gates.
Previously trained controls may be audited and reused for evaluation only.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fourier_score.gmm import (
    BASELINE_ARMS, GMMConfig, MatchedMomentFamily, gated_arm, make_model,
    plateau_arm, spectral_cap_arm,
)
from fourier_score.utils import json_write, source_hash
from scripts.run_gmm_comparison import export_report, mean_sd, verify_pairing, write_csv
from scripts.run_gmm_plateau import (
    evaluate_group, export_diagnostics, file_hash, parallel_map, prepare_arm,
)


def reuse_directory(roots, cfg, lam, seed, arm):
    if arm.gate_mode == "spectral_cap" or not roots:
        return None
    case_id = f"gmm_lambda{lam:g}".replace(".", "p")
    relative = Path(case_id) / f"{arm.name}_seed{seed}" / "checkpoint.pt"
    matches = [root for root in roots if (root / relative).is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one reusable checkpoint for {relative}; found {len(matches)}")
    return matches[0]


def plot_diagnostics(report, cfg, arms, transition):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    family = MatchedMomentFamily(cfg, "gmm", max(cfg.spectrum_lambdas))
    selected = [arms[4], arms[6], arms[8], arms[10]]
    labels = ["Fourier / normalized", "Gated Fourier", "Plateau Fourier", "Spectral Fourier"]
    colors = ["#d77b29", "#863daf", "#c13958", "#111111"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    sigma = torch.logspace(np.log10(cfg.sigma_min), np.log10(cfg.sigma_max), 300)
    profile = []
    for arm, label, color in zip(selected, labels, colors):
        ref = make_model(family, arm, cfg.seeds[0]).reference
        gate, _, scale, total = ref._coefficients(sigma, ref.power[None])
        b = (ref.power[None] / total).sqrt()
        lower = gate.flatten(1).min(1).values.numpy()
        upper = gate.flatten(1).max(1).values.numpy()
        ratio = (scale / b).flatten(1).max(1).values.numpy()
        axes[0].plot(sigma.numpy(), upper, label=label, color=color)
        if arm.gate_mode == "spectral_cap":
            axes[0].plot(sigma.numpy(), lower, color=color, ls="--", lw=1)
            axes[0].fill_between(sigma.numpy(), lower, upper, color=color, alpha=.1)
        axes[1].plot(sigma.numpy(), ratio, color=color)
        profile.extend(dict(method=arm.name, sigma=float(s), gate_min=float(lo), gate_max=float(hi),
                            max_scale_ratio=float(r)) for s, lo, hi, r in zip(sigma, lower, upper, ratio))
        group = [r for r in transition if r["method"] == arm.name]
        xs = sorted({r["sigma"] for r in group})
        stats = [mean_sd([r["score_error"] for r in group if r["sigma"] == x]) for x in xs]
        means = np.array([s[0] for s in stats])
        sd = np.array([s[1] or 0 for s in stats])
        axes[2].plot(xs, means, color=color, marker=".")
        axes[2].fill_between(xs, means-sd, means+sd, color=color, alpha=.12)
    for ax in axes:
        ax.axvspan(arms[-1].sigma_lo, arms[-1].sigma_hi, color="#777777", alpha=.08)
        ax.grid(alpha=.18)
        ax.set_xlabel("Noise sigma")
    axes[0].set(xscale="log", ylabel="Gate g", title="Gate range across frequencies")
    axes[1].set(xscale="log", ylabel="Maximum c / b", title="Residual scale relative to Fourier")
    axes[1].hlines((1+arms[-1].delta**2)**.5, arms[-1].sigma_hi, cfg.sigma_max,
                   colors="black", linestyles=":", lw=1.5)
    axes[2].set(ylabel="Test scaled true-score MSE", title=f"Transition diagnostic: lambda = {family.lam:g}")
    for ax in axes[:2]:
        ax.set_xticks([.1, .3, 1., 2., 3.], labels=["0.1", "0.3", "1", "2", "3"])
        ax.minorticks_off()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for extension in ("png", "svg", "pdf"):
        path = report / f"gmm_spectral_diagnostics.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    write_csv(report / "gate_profile.csv", profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reuse-baselines", type=Path, nargs="*", default=[],
                        help="Checkpoint roots for evaluation controls only; never used as teachers")
    parser.add_argument("--preset", choices=("smoke", "experiment"), default="experiment")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--sigma-switch", type=float, default=1.5)
    parser.add_argument("--sharpness", type=float, default=4.)
    parser.add_argument("--sigma-lo", type=float, default=1.)
    parser.add_argument("--sigma-hi", type=float, default=2.)
    parser.add_argument("--delta", type=float, default=.5)
    parser.add_argument("--test-bank-version", default="gmm-spectral-cap-v1")
    args = parser.parse_args(argv)
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Choose positive workers and distinct nonnegative seeds")
    if not args.test_bank_version or args.test_bank_version in ("gmm-oracle-v1", "gmm-gated-v1", "gmm-plateau-v1"):
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
            *(plateau_arm(cov, args.sigma_switch, args.sharpness, .8, 1.) for cov in ("scalar", "fourier")),
            *(spectral_cap_arm(cov, args.sigma_switch, args.sharpness, args.sigma_lo, args.sigma_hi, args.delta)
              for cov in ("scalar", "fourier")))
    cfg = replace(cfg, methods=tuple(arm.name for arm in arms), run_tag=args.output.name)
    transition_sigmas = sorted(set([max(cfg.sigma_min, args.sigma_lo * .6),
                                   max(cfg.sigma_min, args.sigma_lo * .8),
                                   *np.geomspace(args.sigma_lo, args.sigma_hi, 13).tolist(),
                                   min(cfg.sigma_max, args.sigma_hi * 1.2)]))
    output = args.output.resolve()
    report = (args.report or output / "report").resolve()
    roots = [p.resolve() for p in args.reuse_baselines]
    if output in roots or len(set(roots)) != len(roots):
        parser.error("Reuse roots must be distinct and outside the new output directory")
    provenance = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                      source_sha256=source_hash(), runner_sha256=file_hash(__file__),
                      comparison_helpers_sha256=file_hash(ROOT / "scripts/run_gmm_comparison.py"),
                      experiment_helpers_sha256=file_hash(ROOT / "scripts/run_gmm_plateau.py"),
                      git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)))
    fixed = dict(sigma_switch=args.sigma_switch, sharpness=args.sharpness,
                 sigma_lo=args.sigma_lo, sigma_hi=args.sigma_hi, delta=args.delta)
    selection = dict(criterion="Fixed proposal before training; no search or test selection",
                     **fixed, candidates=[{**fixed, "selection": "fixed"}])
    protocol = dict(config=asdict(cfg), arms=[asdict(a) for a in arms], provenance=provenance,
                    selection=selection, test_bank_version=args.test_bank_version,
                    transition_sigmas=transition_sigmas, reuse_baselines=[str(p) for p in roots],
                    constraints=dict(backbones_per_model=1, loss="normalized_residual for new arms",
                                     teachers=False, auxiliary_losses=False, pretrained_initialization=False))
    plan = output / "protocol.json"
    invocation = dict(provenance)
    if plan.exists():
        previous = json.loads(plan.read_text())
        for key in ("git_revision", "git_dirty"):
            provenance[key] = previous["provenance"][key]
        if json.loads(json.dumps(protocol)) != previous:
            raise ValueError("Existing spectral-gate protocol differs; choose a new output directory")
    json_write(protocol, plan)
    json_write(selection, output / "selection.json")
    history_path = output / "execution_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(started_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers, provenance=invocation))
    json_write(history, history_path)
    tasks = [(cfg, lam, seed, arm, output, reuse_directory(roots, cfg, lam, seed, arm), provenance)
             for arm in (*arms[-2:], *arms[:-2]) for lam in cfg.spectrum_lambdas for seed in cfg.seeds]
    results = parallel_map(prepare_arm, tasks, args.workers)
    pairing = verify_pairing(results, cfg, arms)
    test_cfg = replace(cfg, bank_version=args.test_bank_version)
    tasks = [(test_cfg, lam, seed, results, output, transition_sigmas)
             for lam in cfg.spectrum_lambdas for seed in cfg.seeds]
    tested = [r for group in parallel_map(evaluate_group, tasks, args.workers) for r in group]
    torch.set_num_threads(1)
    payload = export_report(report, cfg, provenance, results, tested, selection, pairing, arms,
                            figure_stem="gmm_spectral_comparison")
    regions, transition = export_diagnostics(report, tested, cfg, arms)
    family = MatchedMomentFamily(cfg, "gmm", max(cfg.spectrum_lambdas))
    counts = {a.name: sum(p.numel() for p in make_model(family, a, cfg.seeds[0]).parameters()) for a in arms}
    if len(set(counts.values())) != 1:
        raise AssertionError("Comparison must use identical parameter counts")
    payload.update(n_training_runs=sum(r["origin"] == "trained" for r in results),
                   n_reused_runs=sum(r["origin"] != "trained" for r in results),
                   test_bank_version=args.test_bank_version, constraints=protocol["constraints"], parameter_counts=counts,
                   transition_diagnostic=dict(spectrum_lambda=max(cfg.spectrum_lambdas), sigmas=transition_sigmas,
                                              n_per_noise=min(1024, cfg.test_per_noise),
                                              bank_version=args.test_bank_version + "-transition",
                                              included_in_primary_mean=False), noise_regions=regions)
    json_write(payload, report / "summary.json")
    json_write(protocol, report / "protocol.json")
    json_write(history, report / "execution_history.json")
    plot_diagnostics(report, cfg, arms, transition)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
