"""GMM orchestration: legacy flags, arm suites, resume rule, provenance, report stage.

Archived protocols and the smoke chain are checked by tests/contracts/test_contract_gmm_*.py.
"""

import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fourier_score.gmm import gated_arm, log_gate_arm
from experiments.__main__ import gmm_parser, main
from experiments.common import file_sha256, open_protocol, orchestration_provenance
from experiments.gmm import pipeline, registry
from experiments.gmm.pipeline import select_gate
from experiments.gmm.registry import EXPERIMENTS, SUITES

ROOT = Path(__file__).resolve().parents[1]
# Literal epoch-0 defaults of scripts/run_gmm_*.py (not imported from the code under test).
COMMON = dict(report=None, preset="experiment", seeds=[42, 43, 44], stage="all")
DEFAULTS = {
    "gated": dict(workers=3, switches=[.5, 1., 1.5], sharpness=4., bank_version="gmm-gated-v1"),
    "plateau": dict(workers=12, reuse_baselines=None, sigma_switch=1.5, sharpness=4., sigma_lo=.8, sigma_hi=1.,
                    test_bank_version="gmm-plateau-v1"),
    "spectral": dict(workers=12, reuse_baselines=[], sigma_switch=1.5, sharpness=4., sigma_lo=1., sigma_hi=2.,
                     delta=.5, test_bank_version="gmm-spectral-cap-v1"),
    "gate_shapes": dict(workers=16, reuse_baselines=[], sigma_switch=1.5, sharpness=4.,
                        test_bank_version="gmm-linear-tanh-v1"),
    "log_gates": dict(workers=16, reuse_baselines=[], sigma_lo=.1, linear_hi=3., sigmoid_hi=1.5, sigma_switch=.75,
                      sharpness=2., test_bank_version="gmm-log-gates-v1"),
}
FORBIDDEN = {  # each runner's hand-written blacklist at the tag, plus gmm-oracle-v1 for plateau
    "plateau": {"gmm-oracle-v1", "gmm-gated-v1"},
    "spectral": {"gmm-oracle-v1", "gmm-gated-v1", "gmm-plateau-v1"},
    "gate_shapes": {"gmm-oracle-v1", "gmm-gated-v1", "gmm-plateau-v1", "gmm-spectral-cap-v1"},
    "log_gates": {"gmm-oracle-v1", "gmm-gated-v1", "gmm-plateau-v1", "gmm-spectral-cap-v1", "gmm-linear-tanh-v1"},
}


def parse(name, *argv):
    parser = gmm_parser(EXPERIMENTS[name], "test")
    return parser, parser.parse_args(["--output", "out", *argv])


def new_spec():
    """A new fixed-gate experiment given only the fields experiments/README.md lists as required."""
    return registry.FixedGateExperiment(
        name="new_gate", description="A new gate.", report="reports/gmm/log-axis-gates.md",
        figure_stem="gmm_new_gate_comparison", options=(("--sigma-switch", dict(type=float, default=.6)),),
        new_modes=("bounded_log_sigmoid",),
        new_arms=lambda a: registry.pair(log_gate_arm, "bounded_log_sigmoid", sigma_switch=a.sigma_switch),
        controls=lambda a: SUITES["log_gates"], test_bank="gmm-new-gate-v1",
        criterion="Fixed before training", check=lambda cfg, a: None, transition=lambda cfg, a: [.5, 1.],
        fixed=("sigma_switch",), diagnostics=registry.Diagnostics("shapes", "gmm_new_gate_diagnostics"))


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_legacy_flags_and_defaults(name):
    _, args = parse(name)
    assert {k: v for k, v in vars(args).items() if k != "output"} == {**COMMON, **DEFAULTS[name]}


@pytest.mark.parametrize("name, argv, message", [
    ("gated", ["--switches", "1", "1"], "Gate switches must be distinct"),
    ("gated", ["--workers", "0"], "Choose positive workers, distinct nonnegative seeds, and a bank version"),
    ("plateau", ["--sigma-lo", "2"], "Transition must lie within the training sigma range"),
    ("plateau", ["--reuse-baselines", "out"], "Reuse checkpoints must be outside the new output directory"),
    ("spectral", ["--test-bank-version", "gmm-plateau-v1"], "Choose an independent test bank version"),
    ("gate_shapes", ["--sigma-switch", "3"], "Gate center must lie within the training sigma range"),
    ("log_gates", ["--sigmoid-hi", "4"], "Gate bounds must lie within the training sigma range"),
    ("log_gates", ["--reuse-baselines", "a", "a"], "Reuse roots must be distinct and outside the new output directory"),
    ("log_gates", ["--seeds", "1", "1"], "Choose positive workers and distinct nonnegative seeds"),
])
def test_legacy_validation_messages(name, argv, message, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser, args = parse(name, *argv)
    with pytest.raises(SystemExit) as stop:
        pipeline.run(EXPERIMENTS[name], args, parser)
    assert stop.value.code == 2 and capsys.readouterr().err.rstrip().endswith(f"error: {message}")
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("report", [None, "out/report", "out"])
def test_report_stage_never_writes_into_output(report, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("gated", "log_gates"):
        parser, args = parse(name, "--stage", "report", *(() if report is None else ("--report", report)))
        with pytest.raises(SystemExit) as stop:
            pipeline.run(EXPERIMENTS[name], args, parser)
        assert stop.value.code == 2 and capsys.readouterr().err.rstrip().endswith(
            "error: --stage report needs a --report directory outside --output")
    assert not any(tmp_path.iterdir())


def test_experiments_define_no_torch_modules():
    """Numerics stay in fourier_score (covered by source_sha256), never in orchestration."""
    for path in sorted((ROOT / "experiments").rglob("*.py")):
        classes = [node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.ClassDef)]
        assert not [c.name for c in classes if any(ast.unparse(b).split(".")[-1] == "Module" for b in c.bases)], path


def test_suites_chain_controls_in_order():
    names = list(EXPERIMENTS)
    assert list(SUITES) == ["baselines", *names] and len(SUITES["log_gates"]) == 19
    for previous, name in zip(["baselines", *names], names):
        spec, (_, args) = EXPERIMENTS[name], parse(name)
        assert spec.controls(args) == SUITES[previous]
        new = spec.candidates(args, 1.5) if name == "gated" else spec.new_arms(args)
        assert (*spec.controls(args), *new) == SUITES[name]
        assert {arm.gate_mode for arm in new} == set(spec.new_modes)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_spec_paths_exist(name):
    spec = EXPERIMENTS[name]
    assert (ROOT / spec.report).is_file() and (spec.stub is None or (ROOT / spec.stub).is_file())
    assert all((ROOT / path).is_file() for path in spec.inputs)


def test_new_spec_needs_no_stub_and_keeps_the_modern_defaults(monkeypatch, capsys):
    spec = new_spec()
    assert (spec.label, spec.stub, spec.workers, spec.reuse_nargs, spec.parameter_counts) == (
        "new_gate", None, 16, "*", True)
    monkeypatch.setitem(EXPERIMENTS, spec.name, spec)
    assert registry.forbidden_test_banks(spec) == FORBIDDEN["log_gates"] | {"gmm-log-gates-v1"}
    parser, args = parse(spec.name)
    protocol = pipeline.plan_fixed(spec, args, parser)[-1]
    assert all(path.startswith("experiments/") for path in protocol["provenance"]["orchestration_sha256"])
    assert protocol["selection"]["candidates"] == [dict(sigma_switch=.6, selection="fixed")]  # one new mode
    main(["list"])
    assert capsys.readouterr().out.splitlines()[-1].split()[:3] == ["gmm", "new_gate", "-"]
    with pytest.raises(SystemExit):
        main(["gmm", spec.name, "--help"])
    assert capsys.readouterr().out.startswith("usage: python -m experiments gmm new_gate ")
    with pytest.raises(FileNotFoundError):  # a declared stub is never silently left out
        orchestration_provenance(entry_points=("scripts/run_gmm_missing.py",))


def test_forbidden_test_banks_match_epoch0_runners():
    assert {name: registry.forbidden_test_banks(EXPERIMENTS[name]) for name in FORBIDDEN} == FORBIDDEN


def test_gate_selection_uses_validation_and_one_shared_switch():
    results = []
    for switch in (.5, 1.):
        for covariance in ("scalar", "fourier"):
            arm = gated_arm(covariance, switch)
            results.append(dict(arm=asdict(arm), completed=True, step=4, spectrum_lambda=1., seed=42,
                                validation=[dict(split="validation", step=4, score_error=2-switch)],
                                test=dict(score_error=switch)))
    selection = select_gate(results, [.5, 1.])
    assert selection["sigma_switch"] == 1.
    assert selection["candidates"][0]["n_runs"] == 2
    with pytest.raises(ValueError, match="Unpaired"):
        select_gate(results[:-1], [.5, 1.])


def copy(value):
    return json.loads(json.dumps(value))


def test_open_protocol_resumes_only_the_same_protocol(tmp_path):
    protocol = dict(config={"a": 1}, provenance=dict(
        python="3", torch="2", numpy="1", source_sha256="s", orchestration_sha256={"experiments/a.py": "o1"},
        input_sha256={"design.json": "d"}, git_revision="r1", git_dirty=False))
    assert open_protocol(tmp_path, copy(protocol), label="plateau") == protocol["provenance"]
    # Orchestration edits (any experiments/ file, the stub) and new commits resume the recorded protocol.
    resumed = copy(protocol)
    resumed["provenance"].update(orchestration_sha256={"experiments/a.py": "o2", "experiments/b.py": "o3"},
                                 git_revision="r2", git_dirty=True)
    invocation = open_protocol(tmp_path, resumed, label="plateau")
    assert invocation["orchestration_sha256"] == {"experiments/a.py": "o2", "experiments/b.py": "o3"}
    assert invocation["git_revision"] == "r2" and resumed == protocol
    assert json.loads((tmp_path / "protocol.json").read_text()) == protocol
    for key, value in (("source_sha256", "t"), ("torch", "3"), ("input_sha256", {"design.json": "e"})):
        changed = copy(protocol)
        changed["provenance"][key] = value
        with pytest.raises(ValueError, match="^Existing plateau protocol differs; choose a new output directory$"):
            open_protocol(tmp_path, changed, label="plateau")
    changed = copy(protocol)
    changed["config"]["a"] = 2
    with pytest.raises(ValueError, match="^Existing plateau protocol differs"):
        open_protocol(tmp_path, changed, label="plateau")
    assert json.loads((tmp_path / "protocol.json").read_text()) == protocol


def test_open_protocol_names_epoch0_outputs(tmp_path):
    epoch0 = dict(config={"a": 1}, provenance=dict(source_sha256="s", runner_sha256="x", git_revision="r",
                                                   git_dirty=False))
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(epoch0))
    protocol = dict(config={"a": 1}, provenance=dict(source_sha256="s", orchestration_sha256={}, git_revision="r",
                                                     git_dirty=False))
    with pytest.raises(ValueError, match=r"is an epoch-0 plateau output \(runner_sha256\); .*\.\./fs-epoch0"):
        open_protocol(tmp_path, protocol, label="plateau")
    assert json.loads(path.read_text()) == epoch0


def test_provenance_fingerprints_orchestration_and_inputs(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(3 * (1 << 20) + 5))
    assert file_sha256(big) == hashlib.sha256(big.read_bytes()).hexdigest()
    provenance = orchestration_provenance((registry.DESIGN,), entry_points=("scripts/run_gmm_log_gates.py",))
    code = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").rglob("*.py")}
    assert set(provenance["orchestration_sha256"]) == code | {"scripts/run_gmm_log_gates.py"}
    assert provenance["input_sha256"] == {registry.DESIGN: file_sha256(ROOT / registry.DESIGN)}
    assert list(provenance)[:4] == ["python", "torch", "numpy", "source_sha256"]


def test_experiments_list_runs_without_matplotlib():
    code = ("import sys, runpy; sys.modules['matplotlib'] = None; sys.argv = ['experiments', 'list'];"
            "runpy.run_module('experiments', run_name='__main__')")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert [line.split()[1] for line in result.stdout.splitlines()] == list(EXPERIMENTS)


@pytest.mark.slow
def test_report_stage_rebuilds_report_read_only(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}
    smoke = ["--preset", "smoke", "--seeds", "42", "--workers", "1"]
    gated, plateau = tmp_path / "gated", tmp_path / "plateau"
    runs = {"gated": ["--output", str(gated), "--switches", "1.5"],
            "plateau": ["--output", str(plateau), "--reuse-baselines", str(gated)]}

    def run(name, *extra):
        result = subprocess.run([sys.executable, "-m", "experiments", "gmm", name, *runs[name], *smoke, *extra],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert result.returncode == 0, result.stderr[-3000:]

    def files(folder):
        return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}

    for name, output in (("gated", gated), ("plateau", plateau)):
        run(name)
        before = files(output)
        run(name, "--stage", "report", "--report", str(tmp_path / f"{name}_report"))
        assert files(output) == before
        rebuilt = files(tmp_path / f"{name}_report")
        original = {k[len("report/"):]: v for k, v in before.items() if k.startswith("report/")}
        history = rebuilt.pop("execution_history.json"), original.pop("execution_history.json")
        assert rebuilt == original
        assert json.loads(history[0])[:-1] == json.loads(history[1])
