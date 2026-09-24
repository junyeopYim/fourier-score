"""Paired GMM experiment for the approved log-linear and bounded S gates.

Keep the prior design fixed: linear over [0.1,3], bounded sigmoid over
[0.1,1.5], center 0.75, exponent 2. Use one backbone and one normalized MSE.
Existing checkpoints are comparison controls, never teachers or initializers.
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
    BASELINE_ARMS, GMMConfig, MatchedMomentFamily, gated_arm, log_gate_arm,
    make_model, plateau_arm, shaped_gate_arm, spectral_cap_arm,
)
from fourier_score.utils import json_write, source_hash
from scripts.run_gmm_comparison import export_report, verify_pairing
from scripts.run_gmm_gate_shapes import plot_diagnostics, reuse_directory
from scripts.run_gmm_plateau import evaluate_group, export_diagnostics, file_hash, parallel_map, prepare_arm

NEW_MODES = ("linear_log_sigma", "bounded_log_sigmoid")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reuse-baselines", type=Path, nargs="*", default=[],
                        help="Checkpoint roots for audited comparison controls only")
    parser.add_argument("--preset", choices=("smoke", "experiment"), default="experiment")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--sigma-lo", type=float, default=.1)
    parser.add_argument("--linear-hi", type=float, default=3.)
    parser.add_argument("--sigmoid-hi", type=float, default=1.5)
    parser.add_argument("--sigma-switch", type=float, default=.75)
    parser.add_argument("--sharpness", type=float, default=2.)
    parser.add_argument("--test-bank-version", default="gmm-log-gates-v1")
    args = parser.parse_args(argv)
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Choose positive workers and distinct nonnegative seeds")
    if not args.test_bank_version or args.test_bank_version in (
            "gmm-oracle-v1", "gmm-gated-v1", "gmm-plateau-v1", "gmm-spectral-cap-v1", "gmm-linear-tanh-v1"):
        parser.error("Choose an independent test bank version")
    cfg = GMMConfig(preset=args.preset, steps=5000, eval_every=500, width=192,
                    seeds=tuple(args.seeds), spectrum_lambdas=(0., .5, 1.), distributions=("gmm",),
                    n_noise_levels=9, val_per_noise=512, test_per_noise=2048,
                    cpu_threads=1, device="cpu", bank_version="gmm-gated-v1")
    if args.preset == "smoke":
        cfg = replace(cfg, image_size=4, width=32, depth=2, steps=12, eval_every=6,
                      batch_size=32, n_noise_levels=3, val_per_noise=32, test_per_noise=64,
                      spectrum_lambdas=(0., 1.))
    if any(not cfg.sigma_min <= args.sigma_lo < hi <= cfg.sigma_max
           for hi in (args.linear_hi, args.sigmoid_hi)):
        parser.error("Gate bounds must lie within the training sigma range")
    controls = (*BASELINE_ARMS,
                *(gated_arm(cov, 1.5, 4.) for cov in ("scalar", "fourier")),
                *(plateau_arm(cov) for cov in ("scalar", "fourier")),
                *(spectral_cap_arm(cov) for cov in ("scalar", "fourier")),
                *(shaped_gate_arm(cov, mode) for mode in ("linear_sigma", "tanh_sigma")
                  for cov in ("scalar", "fourier")))
    new_arms = tuple(log_gate_arm(cov, mode, sigma_lo=args.sigma_lo,
                                 sigma_hi=args.linear_hi if mode == "linear_log_sigma" else args.sigmoid_hi,
                                 sigma_switch=args.sigma_switch, sharpness=args.sharpness)
                     for mode in NEW_MODES for cov in ("scalar", "fourier"))
    arms = (*controls, *new_arms)
    cfg = replace(cfg, methods=tuple(arm.name for arm in arms), run_tag=args.output.name)
    transition_sigmas = sorted({s for s in [.1, .2, .3, .4, .5, .6, .7, .75, .8, .9, 1., 1.1,
                                           1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2., 2.5, 3.,
                                           args.sigma_lo, args.sigma_switch, args.sigmoid_hi, args.linear_hi]
                                if cfg.sigma_min <= s <= cfg.sigma_max})
    output = args.output.resolve()
    report = (args.report or output / "report").resolve()
    roots = [p.resolve() for p in args.reuse_baselines]
    if output in roots or len(set(roots)) != len(roots):
        parser.error("Reuse roots must be distinct and outside the new output directory")
    proposal = ROOT / "assets/log_gate_design/design.json"
    provenance = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                      source_sha256=source_hash(), runner_sha256=file_hash(__file__),
                      comparison_helpers_sha256=file_hash(ROOT / "scripts/run_gmm_comparison.py"),
                      experiment_helpers_sha256=file_hash(ROOT / "scripts/run_gmm_plateau.py"),
                      gate_shape_helpers_sha256=file_hash(ROOT / "scripts/run_gmm_gate_shapes.py"),
                      design_proposal_sha256=file_hash(proposal),
                      git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)))
    fixed = dict(sigma_lo=args.sigma_lo, linear_hi=args.linear_hi, sigmoid_hi=args.sigmoid_hi,
                 sigma_switch=args.sigma_switch, sharpness=args.sharpness)
    selection = dict(criterion="Prior plotted proposal fixed before training; no search or test selection", **fixed,
                     candidates=[dict(mode=mode, **fixed, selection="fixed") for mode in NEW_MODES])
    matches_proposal = ([asdict(a) for a in new_arms if a.parameterization == "fourier_gaussian"]
                        == json.loads(proposal.read_text())["arms"])
    protocol = dict(config=asdict(cfg), arms=[asdict(a) for a in arms], provenance=provenance,
                    selection=selection, test_bank_version=args.test_bank_version,
                    transition_sigmas=transition_sigmas, reuse_baselines=[str(p) for p in roots],
                    matches_prior_design_proposal=matches_proposal,
                    gate_definitions=dict(
                        linear_log_sigma="clip(log(sigma/0.1)/log(3/0.1), 0, 1), with configurable bounds",
                        bounded_log_sigmoid="a^k/(a^k+b^k), a=z*(1-zc), b=(1-z)*zc; z is clipped normalized log noise",
                        sigmoid_tanh_equivalent=True, duplicate_tanh_training=False),
                    constraints=dict(backbones_per_model=1, loss="normalized_residual for new arms",
                                     teachers=False, auxiliary_losses=False, pretrained_initialization=False))
    plan = output / "protocol.json"
    invocation = dict(provenance)
    if plan.exists():
        previous = json.loads(plan.read_text())
        for key in ("git_revision", "git_dirty"):
            provenance[key] = previous["provenance"][key]
        if json.loads(json.dumps(protocol)) != previous:
            raise ValueError("Existing log-gate protocol differs; choose a new output directory")
    json_write(protocol, plan)
    json_write(selection, output / "selection.json")
    history_path = output / "execution_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(started_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers, provenance=invocation))
    json_write(history, history_path)
    tasks = [(cfg, lam, seed, arm, output, reuse_directory(roots, lam, seed, arm, new_modes=NEW_MODES), provenance)
             for arm in (*new_arms, *controls) for lam in cfg.spectrum_lambdas for seed in cfg.seeds]
    results = parallel_map(prepare_arm, tasks, args.workers)
    pairing = verify_pairing(results, cfg, arms)
    test_cfg = replace(cfg, bank_version=args.test_bank_version)
    tasks = [(test_cfg, lam, seed, results, output, transition_sigmas)
             for lam in cfg.spectrum_lambdas for seed in cfg.seeds]
    tested = [r for group in parallel_map(evaluate_group, tasks, args.workers) for r in group]
    torch.set_num_threads(1)
    payload = export_report(report, cfg, provenance, results, tested, selection, pairing, arms,
                            figure_stem="gmm_log_gate_comparison", paired_arms=new_arms)
    regions, transition = export_diagnostics(report, tested, cfg, arms)
    family = MatchedMomentFamily(cfg, "gmm", max(cfg.spectrum_lambdas))
    counts = {a.name: sum(p.numel() for p in make_model(family, a, cfg.seeds[0]).parameters()) for a in arms}
    if len(set(counts.values())) != 1:
        raise AssertionError("Comparison must use identical parameter counts")
    payload.update(n_training_runs=sum(r["origin"] == "trained" for r in results),
                   n_reused_runs=sum(r["origin"] != "trained" for r in results),
                   test_bank_version=args.test_bank_version, constraints=protocol["constraints"], parameter_counts=counts,
                   matches_prior_design_proposal=matches_proposal, gate_definitions=protocol["gate_definitions"],
                   transition_diagnostic=dict(spectrum_lambda=max(cfg.spectrum_lambdas), sigmas=transition_sigmas,
                                              n_per_noise=min(1024, cfg.test_per_noise),
                                              bank_version=args.test_bank_version + "-transition",
                                              included_in_primary_mean=False), noise_regions=regions)
    json_write(payload, report / "summary.json")
    json_write(protocol, report / "protocol.json")
    json_write(history, report / "execution_history.json")
    plot_diagnostics(report, cfg, arms, transition,
                     gate_modes=("none", "log_sigma", "log_sigma_plateau", "spectral_cap", *NEW_MODES),
                     figure_stem="gmm_log_gate_diagnostics")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
