"""How a GMM gate experiment runs: train or audit, select, test, report.

The task functions (``train_group``, ``prepare_arm``, ``evaluate_group`` and
the report-stage readers) are module-level so spawned workers can unpickle
them.  Their logic is the epoch-0 runner code (``scripts/run_gmm_*.py`` at tag
``pre-template-refactor``) with two deliberate changes: new checkpoints are
signed with numerics provenance only, and audit rows of reused checkpoints
name the output root and protocol they came from.

Stages: ``all`` plans, writes (or resumes) ``protocol.json``, trains, selects,
tests and reports.  ``report`` rebuilds the report directory of a finished
run from its checkpoints and test JSON without training or writing into the
output directory (reused controls are re-audited, which only reads them).
"""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from fourier_score.gmm import (
    GMMArm, MatchedMomentFamily, evaluate_model, make_bank, make_model, tensor_state_hash, train_arm,
)
from fourier_score.utils import json_write
from experiments.common import (
    append_history, file_sha256, history_entry, numerics_provenance, open_protocol,
    orchestration_provenance, parallel_map, read_history, read_protocol,
)
from experiments.gmm import registry
from experiments.gmm.report import export_diagnostics, export_report, plot_diagnostics

REUSED = "reused; final EMA and every validation noise bin reproduced exactly"


@lru_cache(maxsize=None)
def case_id(cfg, lam):
    return MatchedMomentFamily(cfg, "gmm", lam).case_id


def run_folder(root, case, arm, seed):
    """``root/<case id>/<arm>_seed<seed>``: the layout of fourier_score.gmm.train_arm outputs."""
    return Path(root) / case / f"{arm.name}_seed{seed}"


# ------------------------------------------------------------------ tasks


def train_group(task):
    cfg, lam, seed, arms, output, provenance = task
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(True)
    family = MatchedMomentFamily(cfg, "gmm", lam)
    validation = make_bank(family, "validation")
    return [train_arm(family, arm, seed, validation, output, provenance) for arm in arms]


def reuse_directory(roots, cfg, lam, seed, arm, *, new_modes):
    """The one root holding a reusable checkpoint of ``arm``, or None to train it here."""
    if arm.gate_mode in new_modes or not roots:
        return None
    relative = run_folder("", case_id(cfg, lam), arm, seed) / "checkpoint.pt"
    matches = [root for root in roots if (root / relative).is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one reusable checkpoint for {relative}; found {len(matches)}")
    return matches[0]


def load_verified_ema(family, arm, seed, checkpoint, expected, message):
    """Model with the checkpoint's EMA weights; ``message`` if they are not ``expected``."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = make_model(family, arm, seed)
    model.load_state_dict(payload["ema"], strict=True)
    if tensor_state_hash(model.backbone.state_dict()) != expected:
        raise ValueError(message)
    return model


def prepare_arm(task):
    cfg, lam, seed, arm, output, reuse, provenance = task
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(True)
    family = MatchedMomentFamily(cfg, "gmm", lam)
    validation = make_bank(family, "validation")
    relative = run_folder("", family.case_id, arm, seed) / "checkpoint.pt"
    origin_fields = {}
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
        origin = REUSED
        # Traceability of the reused run (its checkpoint signature no longer names the runner).
        protocol = reuse / "protocol.json"
        origin_fields = dict(origin_output_root=str(reuse),
                             origin_protocol_sha256=file_sha256(protocol) if protocol.is_file() else None)
        print(f"AUDITED {family.case_id} {arm.name} seed={seed}", flush=True)
    else:
        state = train_arm(family, arm, seed, validation, output, provenance)
        checkpoint = output / relative
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        origin = "trained"
    return {**state, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "training_provenance": payload["provenance"], "origin": origin, **origin_fields}


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


def select_gate(results, switches):
    """Choose ONE gate for both covariances using final-step validation only.

    Average equally over both covariances, all planned spectra and all seeds.
    All candidates must cover the same complete set of training conditions.
    """
    groups = {}
    conditions = set()
    for result in results:
        arm = result["arm"]
        if not result["completed"]:
            raise ValueError("Model selection requires completed runs")
        if arm["gate_mode"] != "log_sigma":
            continue
        key = (result["spectrum_lambda"], result["seed"], arm["parameterization"])
        conditions.add(key)
        values = groups.setdefault(arm["sigma_switch"], {})
        if key in values:
            raise ValueError("Duplicate gate validation condition")
        metric = result["validation"][-1]
        if metric["split"] != "validation" or metric["step"] != result["step"]:
            raise ValueError("Gate selection requires final-step validation")
        values[key] = metric["score_error"]
    if set(groups) != set(switches) or not conditions:
        raise ValueError("Missing gate candidates")
    rows = []
    for switch in switches:
        if set(groups[switch]) != conditions:
            raise ValueError("Unpaired gate validation conditions")
        rows.append(dict(sigma_switch=switch, n_runs=len(conditions),
                         validation_score_error=float(np.mean(list(groups[switch].values())))))
    chosen = min(rows, key=lambda row: (row["validation_score_error"], row["sigma_switch"]))
    return dict(sigma_switch=chosen["sigma_switch"], candidates=rows,
                criterion="Mean final EMA validation scaled-score MSE across both covariances, spectra and seeds")


def test_selected(cfg, output, results, arms):
    """Called only after selection.json has been written."""
    indexed = {(r["spectrum_lambda"], r["seed"], r["method"]): r for r in results}
    tested = []
    for lam in cfg.spectrum_lambdas:
        family = MatchedMomentFamily(cfg, "gmm", lam)
        bank = make_bank(family, "test")
        for seed in cfg.seeds:
            for arm in arms:
                folder = run_folder(output, family.case_id, arm, seed)
                state = indexed[(lam, seed, arm.name)]
                model = load_verified_ema(family, arm, seed, folder / "checkpoint.pt",
                                          state["final_ema_backbone_sha256"], "Final EMA checkpoint mismatch")
                metric = evaluate_model(model, family, bank)
                json_write(metric, folder / "test.json")
                tested.append({**state, "test": metric})
    return tested


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
        if file_sha256(state["checkpoint"]) != state["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during comparison")
        arm = GMMArm(**state["arm"])
        model = load_verified_ema(family, arm, seed, state["checkpoint"], state["final_ema_backbone_sha256"],
                                  "Test EMA differs from final training EMA")
        metric = evaluate_model(model, family, bank)
        folder = run_folder(output / "evaluation", family.case_id, arm, seed)
        json_write(metric, folder / "test.json")
        row = {**state, "test": metric}
        if transition_bank is not None:
            row["transition_test"] = evaluate_model(model, family, transition_bank)
            json_write(row["transition_test"], folder / "transition_test.json")
        tested.append(row)
    print(f"TESTED {family.case_id} seed={seed} methods={len(tested)}", flush=True)
    return tested


# ------------------------------------------------------------------ report-stage readers


def finished_state(cfg, lam, seed, arm, output):
    """(checkpoint, payload) of a completed run of this protocol; never trains."""
    checkpoint = run_folder(output, case_id(cfg, lam), arm, seed) / "checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state"]
    if (payload["config"] != asdict(cfg) or GMMArm(**state["arm"]) != arm
            or not state["completed"] or state["step"] != cfg.steps):
        raise ValueError(f"Report stage needs a completed run of this protocol: {checkpoint}")
    return checkpoint, payload


def collect_arm(task):
    """``prepare_arm`` for a finished run: read trained arms, re-audit reused ones."""
    cfg, lam, seed, arm, output, reuse, provenance = task
    if reuse is not None:
        return prepare_arm(task)
    checkpoint, payload = finished_state(cfg, lam, seed, arm, output)
    return {**payload["state"], "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "training_provenance": payload["provenance"], "origin": "trained"}


def read_selected(cfg, output, results, arms):
    """``test_selected`` for a finished run: the test JSON it wrote."""
    indexed = {(r["spectrum_lambda"], r["seed"], r["method"]): r for r in results}
    return [{**indexed[(lam, seed, arm.name)],
             "test": json.loads((run_folder(output, case_id(cfg, lam), arm, seed) / "test.json").read_text())}
            for lam in cfg.spectrum_lambdas for seed in cfg.seeds for arm in arms]


def read_group(task):
    """``evaluate_group`` for a finished run: the test JSON it wrote."""
    cfg, lam, seed, results, output, transition_sigmas = task
    tested = []
    for state in results:
        if state["spectrum_lambda"] != lam or state["seed"] != seed:
            continue
        folder = run_folder(output / "evaluation", case_id(cfg, lam), GMMArm(**state["arm"]), seed)
        row = {**state, "test": json.loads((folder / "test.json").read_text())}
        if lam == max(cfg.spectrum_lambdas):
            row["transition_test"] = json.loads((folder / "transition_test.json").read_text())
        tested.append(row)
    return tested


# ------------------------------------------------------------------ experiments


def run(spec, args, parser):
    if args.stage == "report":
        # Read-only on the finished run: the rebuilt report never replaces OUTPUT/report.
        output, report = args.output.resolve(), args.report and args.report.resolve()
        if report is None or report.is_relative_to(output):
            parser.error("--stage report needs a --report directory outside --output")
    runner = run_selection if isinstance(spec, registry.SelectionExperiment) else run_fixed
    return runner(spec, args, parser)


def entry_points(spec):
    return () if spec.stub is None else (spec.stub,)


def open_run(spec, args, output, protocol):
    """(protocol, execution history) of this invocation.

    Stage ``all`` writes or resumes ``OUTPUT/protocol.json`` (``open_protocol``)
    and appends to ``OUTPUT/execution_history.json``.  Stage ``report`` only
    reads the finished run's protocol; its history entry goes to the report.
    """
    provenance = protocol["provenance"]
    if args.stage == "report":
        return (read_protocol(output, protocol, label=spec.label),
                [*read_history(output), history_entry(workers=args.workers, stage="report", provenance=provenance)])
    invocation = open_protocol(output, protocol, label=spec.label)
    return protocol, append_history(output, workers=args.workers, provenance=invocation)


def plan_selection(spec, args, parser):
    """Validate flags and build (cfg, controls, arms, output, protocol) without side effects."""
    if args.workers < 1 or not args.bank_version or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Choose positive workers, distinct nonnegative seeds, and a bank version")
    if len(set(args.switches)) != len(args.switches):
        parser.error("Gate switches must be distinct")
    cfg = registry.base_config(args.preset, args.seeds, args.bank_version)
    controls = spec.controls(args)
    arms = (*controls, *spec.new_arms(args))
    cfg = replace(cfg, methods=tuple(arm.name for arm in arms), run_tag=args.output.name)
    provenance = orchestration_provenance(spec.inputs, entry_points=entry_points(spec))
    protocol = dict(config=asdict(cfg), arms=[asdict(a) for a in arms], provenance=provenance,
                    selection=spec.criterion)
    return cfg, controls, arms, args.output.resolve(), protocol


def run_selection(spec, args, parser):
    """Train every candidate, select one gate on validation, then test the selection."""
    cfg, controls, arms, output, protocol = plan_selection(spec, args, parser)
    provenance = protocol["provenance"]
    report = (args.report or output / "report").resolve()
    protocol, history = open_run(spec, args, output, protocol)
    if args.stage == "report":
        results = [finished_state(cfg, lam, seed, arm, output)[1]["state"]
                   for lam in cfg.spectrum_lambdas for seed in cfg.seeds for arm in arms]
    else:
        tasks = [(cfg, lam, seed, (arm,), output, numerics_provenance(provenance))
                 for lam in cfg.spectrum_lambdas for seed in cfg.seeds for arm in arms]
        results = [result for group in parallel_map(train_group, tasks, args.workers) for result in group]
    pairing = verify_pairing(results, cfg, arms)
    selection = select_gate(results, args.switches)
    if args.stage == "report":
        if json.loads(json.dumps(selection)) != json.loads((output / "selection.json").read_text()):
            raise ValueError("Recomputed selection differs from selection.json")
    else:
        # Persist selection BEFORE creating or evaluating any test observations.
        json_write(selection, output / "selection.json")
    print("SELECTED " + json.dumps(selection), flush=True)
    selected = spec.candidates(args, selection["sigma_switch"])
    final_arms = (*controls, *selected)
    torch.set_num_threads(cfg.cpu_threads)
    tested = (read_selected if args.stage == "report" else test_selected)(cfg, output, results, final_arms)
    payload = export_report(report, cfg, protocol["provenance"], results, tested, selection, pairing, final_arms,
                            figure_stem=spec.figure_stem, paired_arms=selected)
    json_write(history, report / "execution_history.json")
    print(json.dumps(payload["summary"], indent=2), flush=True)


def plan_fixed(spec, args, parser):
    """Validate flags and build (cfg, controls, new arms, roots, protocol) without side effects."""
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Choose positive workers and distinct nonnegative seeds")
    if not args.test_bank_version or args.test_bank_version in registry.forbidden_test_banks(spec):
        parser.error("Choose an independent test bank version")
    cfg = registry.base_config(args.preset, args.seeds)
    message = spec.check(cfg, args)
    if message:
        parser.error(message)
    controls, new_arms = spec.controls(args), spec.new_arms(args)
    arms = (*controls, *new_arms)
    cfg = replace(cfg, methods=tuple(arm.name for arm in arms), run_tag=args.output.name)
    transition_sigmas = spec.transition(cfg, args)
    output = args.output.resolve()
    if spec.reuse_nargs is None:
        roots = [args.reuse_baselines.resolve()] if args.reuse_baselines else []
        if output in roots:
            parser.error("Reuse checkpoints must be outside the new output directory")
        reuse_record = str(roots[0]) if roots else None
    else:
        roots = [p.resolve() for p in args.reuse_baselines]
        if output in roots or len(set(roots)) != len(roots):
            parser.error("Reuse roots must be distinct and outside the new output directory")
        reuse_record = [str(p) for p in roots]
    provenance = orchestration_provenance(spec.inputs, entry_points=entry_points(spec))
    fixed = {key: getattr(args, key) for key in spec.fixed}
    candidates = ([dict(mode=mode, **fixed, selection="fixed") for mode in spec.new_modes]
                  if len(spec.new_modes) > 1 else [{**fixed, "selection": "fixed"}])
    selection = dict(criterion=spec.criterion, **fixed, candidates=candidates)
    protocol = dict(config=asdict(cfg), arms=[asdict(a) for a in arms], provenance=provenance,
                    selection=selection, test_bank_version=args.test_bank_version,
                    transition_sigmas=transition_sigmas, reuse_baselines=reuse_record,
                    **spec.extras(args, new_arms))
    return cfg, controls, new_arms, output, roots, protocol


def run_fixed(spec, args, parser):
    """Train the new arms (and any control not reused), then test every arm on a new bank."""
    cfg, controls, new_arms, output, roots, protocol = plan_fixed(spec, args, parser)
    arms = (*controls, *new_arms)
    selection, transition_sigmas = protocol["selection"], protocol["transition_sigmas"]
    extra_keys = list(spec.extras(args, new_arms))  # protocol fields the summary repeats
    report = (args.report or output / "report").resolve()
    provenance = protocol["provenance"]
    finished = args.stage == "report"
    protocol, history = open_run(spec, args, output, protocol)
    if not finished:
        json_write(selection, output / "selection.json")
    # Schedule new training first, allowing baseline audits to fill spare cores.
    tasks = [(cfg, lam, seed, arm, output, reuse_directory(roots, cfg, lam, seed, arm, new_modes=spec.new_modes),
              numerics_provenance(provenance))
             for arm in (*new_arms, *controls) for lam in cfg.spectrum_lambdas for seed in cfg.seeds]
    results = parallel_map(collect_arm if finished else prepare_arm, tasks, args.workers)
    pairing = verify_pairing(results, cfg, arms)
    test_cfg = replace(cfg, bank_version=args.test_bank_version)
    tasks = [(test_cfg, lam, seed, results, output, transition_sigmas)
             for lam in cfg.spectrum_lambdas for seed in cfg.seeds]
    groups = [read_group(task) for task in tasks] if finished else parallel_map(evaluate_group, tasks, args.workers)
    tested = [r for group in groups for r in group]
    torch.set_num_threads(1)
    payload = export_report(report, cfg, protocol["provenance"], results, tested, selection, pairing, arms,
                            figure_stem=spec.figure_stem, paired_arms=new_arms)
    regions, transition = export_diagnostics(report, tested, cfg, arms)
    extra = {key: protocol[key] for key in ("constraints",) if key in extra_keys}
    if spec.parameter_counts:
        family = MatchedMomentFamily(cfg, "gmm", max(cfg.spectrum_lambdas))
        counts = {a.name: sum(p.numel() for p in make_model(family, a, cfg.seeds[0]).parameters()) for a in arms}
        if len(set(counts.values())) != 1:
            raise AssertionError("Comparison must use identical parameter counts")
        extra["parameter_counts"] = counts
    extra.update((key, protocol[key]) for key in extra_keys if key != "constraints")
    payload.update(n_training_runs=sum(r["origin"] == "trained" for r in results),
                   n_reused_runs=sum(r["origin"] != "trained" for r in results),
                   test_bank_version=args.test_bank_version, **extra,
                   transition_diagnostic=dict(spectrum_lambda=max(cfg.spectrum_lambdas), sigmas=transition_sigmas,
                                              n_per_noise=min(1024, cfg.test_per_noise),
                                              bank_version=args.test_bank_version + "-transition",
                                              included_in_primary_mean=False),
                   noise_regions=regions)
    json_write(payload, report / "summary.json")
    json_write(protocol, report / "protocol.json")
    json_write(history, report / "execution_history.json")
    plot_diagnostics(spec.diagnostics, report, cfg, arms, transition)
    print(json.dumps(payload["summary"], indent=2), flush=True)
