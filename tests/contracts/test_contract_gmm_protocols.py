"""Contract: the experiments registry reproduces the archived GMM protocols.

Written against the A3 API (these tests skip until
``experiments.gmm.registry`` exists)::

    experiments.gmm.registry
        EXPERIMENTS: dict[str, SelectionExperiment | FixedGateExperiment]
            in chain order: gated, plateau, spectral, gate_shapes, log_gates.
            Spec fields used: options ((flag, argparse kwargs), ...),
            test_bank, reuse_nargs, controls(args), new_arms(args), criterion.
        base_config(preset, seeds, bank_version=TRAIN_BANK) -> GMMConfig
    experiments.gmm.pipeline
        plan_fixed(spec, args, parser) -> (cfg, controls, new_arms, output, roots, protocol)
        plan_selection(spec, args, parser) -> (cfg, controls, arms, output, protocol)
            side-effect free; ``protocol`` is what ``run_fixed`` /
            ``run_selection`` write as ``OUTPUT/protocol.json``.

``protocol`` builds the default-flag ``args`` from the spec (like the CLI
parser would) and returns the planned protocol of the spec's kind.
``provenance``, ``reuse_baselines`` and ``config.run_tag`` are ignored;
every other key must match exactly.

Compared against (values serialized with ``json.dumps(sort_keys=True)``, so
int vs float and arm order matter):

* ``assets/gmm_{plateau,spectral,gate_shape,log_gate}_comparison/protocol.json``
  for the default (``experiment``) preset (arms: see ``assert_arms`` for
  archives that predate a ``GMMArm`` field);
* gated, which archived no ``protocol.json``: ``config`` from
  ``assets/gmm_gated_comparison/summary.json``, the 11 training arms and the
  selection text from the epoch-0 runner's smoke golden
  (``tests/golden/gmm_smoke/gated/protocol.json``; arms and selection do not
  depend on the preset), the 7 archived final arms, and, with
  ``FOURIER_GOLDEN_ROOT``, the full ``saved/gmm_gated_20260924/protocol.json``;
* every spec's ``smoke`` preset against the smoke goldens;
* the log-gate design input ``assets/log_gate_design/design.json``.

``test_gated_reselects_archived_switch`` needs no registry: ``select_gate``
(from the experiments pipeline once moved there) must pick sigma_switch=1.5
again from the archived final validation errors.
"""

import argparse
import csv
from dataclasses import MISSING, asdict, fields, is_dataclass
import json
import os
from pathlib import Path

import pytest

import golden_probes as gp
from golden_probes import gmm_smoke_norm as norm

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"
ARCHIVED = {
    "plateau": "gmm_plateau_comparison",
    "spectral": "gmm_spectral_comparison",
    "gate_shapes": "gmm_gate_shape_comparison",
    "log_gates": "gmm_log_gate_comparison",
}
GATED = ASSETS / "gmm_gated_comparison"
GATED_SAVED = "saved/gmm_gated_20260924/protocol.json"
IGNORED = ("provenance", "reuse_baselines")
LOG_MODES = ("linear_log_sigma", "bounded_log_sigmoid")


def canonical(value):
    return json.dumps(value, sort_keys=True)


def plain(value):
    """Dataclasses/tuples -> JSON types (int and float stay distinct)."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def config_string(config):
    return canonical({k: v for k, v in plain(config).items() if k != "run_tag"})


def arm_strings(arms):
    return [canonical(arm) for arm in plain(arms)]


def scientific(protocol):
    return {k: v for k, v in plain(protocol).items() if k not in IGNORED}


def assert_arms(ours, archived, label):
    """Same arms in the same order, each with the archived ``json.dumps`` string.

    Archives written before a ``GMMArm`` field existed (plateau: ``delta``;
    gated: ``sigma_lo``/``sigma_hi``/``delta``) are compared on their keys, and
    the later fields must hold their dataclass defaults; with every field
    archived this is exact string equality.
    """
    arm_cls = gp.symbol("GMMArm", "fourier_score.gmm")
    defaults = {f.name: f.default for f in fields(arm_cls) if f.default is not MISSING}
    ours, archived = plain(ours), plain(archived)
    assert [a["name"] for a in ours] == [a["name"] for a in archived], f"{label}: arm names/order"
    for o, a in zip(ours, archived):
        assert set(a) <= set(o), f"{label}: {a['name']} lost {sorted(set(a) - set(o))}"
        assert canonical({k: o[k] for k in a}) == canonical(a), f"{label}: arm {a['name']}"
        later = {k: o[k] for k in set(o) - set(a)}
        assert canonical(later) == canonical({k: defaults.get(k, MISSING) for k in later}), (
            f"{label}: {a['name']} fields absent from the archive are not defaults: {later}")


def assert_protocol(ours, archived, label):
    ours, archived = scientific(ours), scientific(archived)
    assert sorted(ours) == sorted(archived), f"{label}: protocol keys differ"
    assert config_string(ours["config"]) == config_string(archived["config"]), f"{label}: config"
    assert_arms(ours["arms"], archived["arms"], label)
    for key in sorted(set(archived) - {"config", "arms"}):
        assert canonical(ours[key]) == canonical(archived[key]), f"{label}: {key}"


@pytest.fixture(scope="module")
def registry():
    return pytest.importorskip("experiments.gmm.registry", reason="experiments.gmm.registry does not exist yet (A3)")


class Parser:
    def error(self, message):
        raise AssertionError(f"default flags rejected: {message}")


def default_args(spec, preset, seeds, run_tag):
    """The argparse namespace of ``python -m experiments gmm <name>`` with default flags."""
    args = {flag.lstrip("-").replace("-", "_"): kwargs.get("default") for flag, kwargs in spec.options}
    fixed = hasattr(spec, "reuse_nargs")
    args.update(output=Path(run_tag), report=None, preset=preset, seeds=list(seeds), workers=spec.workers,
                stage="all")
    if fixed:
        args.update(test_bank_version=spec.test_bank, reuse_baselines=None if spec.reuse_nargs is None else [])
    return argparse.Namespace(**args)


def protocol(registry, name, preset="experiment", seeds=(42, 43, 44)):
    """Scientific protocol the registry spec ``name`` writes for its default flags."""
    assert name in registry.EXPERIMENTS, f"registry has no experiment {name!r}"
    spec = registry.EXPERIMENTS[name]
    args = default_args(spec, preset, seeds, run_tag=f"gmm_{name}")
    plan = gp.symbol("plan_fixed" if hasattr(spec, "reuse_nargs") else "plan_selection", "experiments.gmm.pipeline")
    return plain(plan(spec, args, Parser())[-1])


def smoke_golden_protocol(name):
    return norm.load_golden(name)["protocol.json"]["exact"]["protocol.json"]


def test_registry_names_follow_the_chain(registry):
    # Later experiments are appended after the five recorded ones.
    assert list(registry.EXPERIMENTS)[:len(norm.CHAIN)] == [link.name for link in norm.CHAIN]


@pytest.mark.parametrize("name", sorted(ARCHIVED))
def test_archived_protocol(registry, name):
    archived = json.loads((ASSETS / ARCHIVED[name] / "protocol.json").read_text())
    assert_protocol(protocol(registry, name), archived, name)


@pytest.mark.parametrize("name", [link.name for link in norm.CHAIN])
def test_smoke_protocol(registry, name):
    assert_protocol(protocol(registry, name, preset="smoke", seeds=(42,)), smoke_golden_protocol(name), name)


def test_gated_protocol(registry):
    ours = scientific(protocol(registry, "gated"))
    summary = json.loads((GATED / "summary.json").read_text())
    runner = smoke_golden_protocol("gated")
    assert sorted(ours) == sorted(scientific(runner))
    assert config_string(ours["config"]) == config_string(summary["config"])
    assert arm_strings(ours["arms"]) == arm_strings(runner["arms"])
    assert [arm["name"] for arm in ours["arms"]] == summary["config"]["methods"]
    by_name = {arm["name"]: arm for arm in ours["arms"]}
    final = summary["arms"]
    assert all(arm["name"] in by_name for arm in final), "archived final arms not trained"
    assert_arms([by_name[arm["name"]] for arm in final], final, "gated final arms")
    assert canonical(ours["selection"]) == canonical(runner["selection"])


@pytest.mark.golden_local
def test_gated_protocol_saved(registry):
    root = os.environ.get("FOURIER_GOLDEN_ROOT")
    if not root:
        pytest.skip("local-only golden; set FOURIER_GOLDEN_ROOT to the original checkout")
    archived = json.loads((Path(root) / GATED_SAVED).read_text())
    assert_protocol(protocol(registry, "gated"), archived, "gated")


def test_log_gate_design_input(registry):
    design = json.loads((ASSETS / "log_gate_design" / "design.json").read_text())
    ours = protocol(registry, "log_gates")
    new = [arm for arm in ours["arms"]
           if arm["gate_mode"] in LOG_MODES and arm["parameterization"] == "fourier_gaussian"]
    assert arm_strings(new) == arm_strings(design["arms"])
    assert ours["matches_prior_design_proposal"] is True


def test_gated_reselects_archived_switch():
    """Final-step validation errors of the archived 99 runs select 1.5 again."""
    select_gate = gp.symbol("select_gate", "experiments.gmm.pipeline", "fourier_score.gmm")
    summary = json.loads((GATED / "summary.json").read_text())
    arms = {arm["name"]: arm for arm in smoke_golden_protocol("gated")["arms"]}
    assert list(arms) == summary["config"]["methods"]
    steps = summary["config"]["steps"]
    results = []
    with open(GATED / "validation_curves.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["step"]) != steps:
                continue
            results.append({
                "arm": arms[row["method"]], "completed": True, "step": steps,
                "spectrum_lambda": float(row["spectrum_lambda"]), "seed": int(row["seed"]),
                "validation": [{"split": "validation", "step": steps,
                                "score_error": float(row["validation_score_error"])}],
            })
    config = summary["config"]
    assert len(results) == len(config["methods"]) * len(config["seeds"]) * len(config["spectrum_lambdas"])
    switches = sorted({arm["sigma_switch"] for arm in arms.values() if arm["gate_mode"] == "log_sigma"})
    selection = select_gate(results, switches)
    assert selection["sigma_switch"] == 1.5
    assert canonical(selection) == canonical(summary["selection"])
