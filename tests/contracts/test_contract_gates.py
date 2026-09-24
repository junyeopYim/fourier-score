"""Contract: ``fourier_score.gates`` reproduces the epoch-0 gate code, torch-free.

``EPOCH0_GATES`` and ``EPOCH0_SIGNATURE`` are verbatim copies of
``fourier_score/method.py`` L12-79 and ``fourier_score/checkpoints.py`` L14-51
at tag ``pre-template-refactor`` (checked against the tag when git has it).
A grid denser than the config goldens compares objective names, validated
gates, error messages, run-name suffixes and resume signatures exactly.
Resume signatures are compared for known modes only: validate() never lets an
unknown mode reach a checkpoint.

Config modules import without torch (C20); that check fails at the tag.
"""

import copy
import itertools
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest

import golden_probes as gp

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES = ("fourier_score.gates",)
CHECKPOINTS = ("fourier_score.trainer.checkpoints",)
CONFIG = ("fourier_score.config",)

EPOCH0_GATES = r'''GAUSSIAN_OBJECTIVES = {
    "scalar_gaussian": {"covariance": "scalar"},
    "fourier_gaussian": {"covariance": "fourier"},
}
OBJECTIVES = ("score", "diffusion", *GAUSSIAN_OBJECTIVES)
# Score and diffusion are sign conventions, so only one is in the main ablation.
COMPARISON_OBJECTIVES = ("score", *GAUSSIAN_OBJECTIVES)

GATE_DEFAULTS = {"mode": "none", "sigma_switch": 1.0, "sharpness": 4.0, "value": 1.0,
                 "sigma_lo": 0.8, "sigma_hi": 1.0, "delta": 0.5}


def validate_gate(gate=None):
    """A disabled gate means the original reference coefficient g=1."""
    if gate is not None and (not isinstance(gate, dict) or set(gate) - GATE_DEFAULTS.keys()):
        raise ValueError("Invalid Gaussian gate configuration")
    result = {**GATE_DEFAULTS, **(gate or {})}
    if result["mode"] not in ("none", "constant", "log_sigma", "log_sigma_plateau", "spectral_cap",
                               "linear_sigma", "tanh_sigma", "linear_log_sigma", "bounded_log_sigmoid"):
        raise ValueError("gate.mode must be none, constant, log_sigma, log_sigma_plateau, "
                         "spectral_cap, linear_sigma, tanh_sigma, linear_log_sigma, or bounded_log_sigmoid")
    for key in ("sigma_switch", "sharpness", "value", "sigma_lo", "sigma_hi", "delta"):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"gate.{key} must be finite")
    if result["sigma_switch"] <= 0 or result["sharpness"] <= 0:
        raise ValueError("gate.sigma_switch and gate.sharpness must be positive")
    if not 0 <= result["value"] <= 1:
        raise ValueError("gate.value must be between zero and one")
    if not 0 < result["sigma_lo"] < result["sigma_hi"]:
        raise ValueError("gate requires 0 < sigma_lo < sigma_hi")
    if result["delta"] <= 0:
        raise ValueError("gate.delta must be positive")
    if result["mode"] == "bounded_log_sigmoid":
        if not result["sigma_lo"] < result["sigma_switch"] < result["sigma_hi"]:
            raise ValueError("bounded gate requires sigma_lo < sigma_switch < sigma_hi")
        if result["sharpness"] <= 1:
            raise ValueError("bounded gate sharpness must exceed one for flat endpoint slopes")
    return result


def gate_suffix(gate=None):
    gate = validate_gate(gate)
    def number(value):
        return format(value, ".17g").replace(".", "p").replace("-", "m").replace("+", "")
    if gate["mode"] == "none":
        return ""
    if gate["mode"] == "constant":
        return "_gate_constant" + number(gate["value"])
    if gate["mode"] in ("linear_log_sigma", "bounded_log_sigmoid"):
        bounds = [repr(float(gate[key])).removesuffix(".0").replace(".", "p")
                  .replace("-", "m").replace("+", "") for key in ("sigma_lo", "sigma_hi")]
        if gate["mode"] == "linear_log_sigma":
            return f"_gate_loglinear_lo{bounds[0]}_hi{bounds[1]}"
        return (f"_gate_logsigmoid_lo{bounds[0]}_hi{bounds[1]}"
                f"_s{number(gate['sigma_switch'])}_p{number(gate['sharpness'])}")
    suffix = f"_gate_s{number(gate['sigma_switch'])}_p{number(gate['sharpness'])}"
    if gate["mode"] in ("linear_sigma", "tanh_sigma"):
        return suffix + "_" + gate["mode"]
    if gate["mode"] in ("log_sigma_plateau", "spectral_cap"):
        # Shortest round-trip representations keep distinct bounds distinct.
        bounds = [repr(float(gate[key])).removesuffix(".0").replace(".", "p")
                  .replace("-", "m").replace("+", "") for key in ("sigma_lo", "sigma_hi")]
        kind = "plateau" if gate["mode"] == "log_sigma_plateau" else "cap"
        suffix += f"_{kind}_lo{bounds[0]}_hi{bounds[1]}"
        if gate["mode"] == "spectral_cap":
            suffix += f"_d{number(gate['delta'])}"
    return suffix
'''

EPOCH0_SIGNATURE = r'''def resume_signature(cfg):
    c = copy.deepcopy(cfg)
    # Missing objective in v1 checkpoints means DSM; retain their signatures.
    if c["loss"].get("objective", "dsm") == "dsm":
        c["loss"].pop("objective", None)
    # Adding disabled-gate defaults must preserve pre-gate v1 signatures.
    if c["fourier"].get("gate", {}).get("mode", "none") == "none":
        c["fourier"].pop("gate", None)
    else:
        gate = c["fourier"]["gate"]
        if gate["mode"] not in ("log_sigma_plateau", "spectral_cap", "linear_log_sigma", "bounded_log_sigmoid"):
            gate.pop("sigma_lo", None)
            gate.pop("sigma_hi", None)
        if gate["mode"] == "linear_log_sigma":
            gate.pop("sigma_switch", None)
            gate.pop("sharpness", None)
        # Optional defaults must not change older active-gate signatures.
        if gate["mode"] != "spectral_cap":
            gate.pop("delta", None)
    for k in ("name", "device", "evaluation", "sampling"):
        c.pop(k)
    for k in (
        "iterations",
        "save_dir",
        "save_every",
        "snapshot_every",
        "log_every",
        "eval_every",
        "tensorboard",
    ):
        c["trainer"].pop(k)
    for k in ("console", "progress_every_seconds"):
        c["trainer"].pop(k, None)
    for k in ("root", "download", "num_workers"):
        c["data_loader"]["args"].pop(k)
    c["fourier"].pop("cache_dir")
    c["fourier"].pop("stats_batch_size")
    return c
'''

# Literal on purpose: the epoch-0 modes are not read back from the code.
MODES = ("none", "constant", "log_sigma", "log_sigma_plateau", "spectral_cap", "linear_sigma",
         "tanh_sigma", "linear_log_sigma", "bounded_log_sigmoid")
PARAMS = ("sigma_switch", "sharpness", "value", "sigma_lo", "sigma_hi", "delta")
GOOD = (0, 1, 2, 0.1, 0.3, 1 / 3, 0.5, 0.75, 0.8, 1.0, 1.1, 1.5, 3.0, 4.0, 1e-5, 1e20, -0.0, -1.0)
BAD = (math.nan, math.inf, -math.inf, True, None, "0.5", [1.0])


@pytest.fixture(scope="module")
def epoch0():
    namespace = {"copy": copy, "math": math}
    exec(compile(EPOCH0_GATES + "\n\n" + EPOCH0_SIGNATURE, "<epoch0 gates>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("path, first, last, frozen", [
    ("fourier_score/method.py", 12, 79, EPOCH0_GATES),
    ("fourier_score/checkpoints.py", 14, 51, EPOCH0_SIGNATURE),
])
def test_frozen_copy_is_the_tag_source(path, first, last, frozen):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    result = subprocess.run(
        ["git", "show", f"{gp.EPOCH0_TAG}:{path}"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode:
        pytest.skip(f"tag {gp.EPOCH0_TAG} is unavailable: {result.stderr.strip()}")
    assert "".join(result.stdout.splitlines(keepends=True)[first - 1:last]) == frozen


def test_objective_names_and_gate_defaults(epoch0):
    for name in ("GAUSSIAN_OBJECTIVES", "OBJECTIVES", "COMPARISON_OBJECTIVES", "GATE_DEFAULTS"):
        ours = gp.symbol(name, *GATES)
        assert type(ours) is type(epoch0[name]) and repr(ours) == repr(epoch0[name]), name


def gate_inputs():
    yield from (None, {}, "log_sigma", ["log_sigma"], 1, {"width": 1.0}, {"mode": "log_sigma", "width": 1})
    for mode in (*MODES, "", "sigmoid", "NONE", "log_sigma ", 1, None, ["log_sigma"], {"mode": 1}):
        yield {"mode": mode}
    for mode, key in itertools.product(MODES, PARAMS):
        for value in (*GOOD, *BAD):
            yield {"mode": mode, key: value}
    bounds = (0.1, 0.3, 1, 1.5, 3.0)
    for mode, lo, switch, hi, sharpness, delta in itertools.product(
        MODES, bounds, bounds, bounds, (1, 1.1, 4.0), (0.1, 0.5)
    ):
        yield {"mode": mode, "sigma_lo": lo, "sigma_switch": switch, "sigma_hi": hi,
               "sharpness": sharpness, "delta": delta}
    rng = random.Random(20260925)
    for mode in MODES:
        for _ in range(400):
            keys = rng.sample(PARAMS, rng.randint(0, len(PARAMS)))
            yield {"mode": mode, **{k: rng.choice(GOOD if rng.random() < 0.95 else BAD) for k in keys}}


def outcome(fn, value):
    try:
        return repr(fn(copy.deepcopy(value)))
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def test_validate_gate_and_suffix_match_epoch0(epoch0):
    validate_gate, gate_suffix = gp.symbol("validate_gate", *GATES), gp.symbol("gate_suffix", *GATES)
    checked = accepted = 0
    for gate in gate_inputs():
        expected = outcome(epoch0["validate_gate"], gate)
        assert outcome(validate_gate, gate) == expected, gate
        assert outcome(gate_suffix, gate) == outcome(epoch0["gate_suffix"], gate), gate
        checked += 1
        accepted += not expected.startswith(("ValueError", "TypeError"))
    assert checked > 10000 and accepted > 2000, (checked, accepted)


# Stored forms: keys an older checkpoint lacks, and partially written gates.
FORMS = [(), (("fourier", "gate"),), (("fourier", "gate", "mode"),), (("loss", "objective"),),
         tuple(("fourier", "gate", k) for k in ("sigma_lo", "sigma_hi", "delta")),
         *[(("fourier", "gate", k),) for k in PARAMS]]


def without(cfg, paths):
    out = copy.deepcopy(cfg)
    for path in paths:
        parent = out
        for key in path[:-1]:
            parent = parent.get(key, {})
        parent.pop(path[-1], None)
    return out


def test_resume_signature_matches_epoch0(epoch0):
    load_config = gp.symbol("load_config", *CONFIG)
    resume_signature = gp.symbol("resume_signature", *CHECKPOINTS)
    base = load_config(str(REPO_ROOT / "configs/smoke.json"), ["loss.type=fourier_gaussian"])
    rng = random.Random(20260926)
    gates = [g for g in gate_inputs() if isinstance(g, dict) and g.get("mode") in MODES]
    checked = 0
    for mode in MODES:
        pool = [g for g in gates if g["mode"] == mode]
        for gate, objective in itertools.product(rng.sample(pool, 40), ("dsm", "normalized_residual")):
            cfg = copy.deepcopy(base)
            cfg["loss"]["objective"] = objective
            cfg["fourier"]["gate"] = {**epoch0["GATE_DEFAULTS"], **gate}
            for form in FORMS:
                stored = without(cfg, form)
                assert outcome(resume_signature, stored) == outcome(epoch0["resume_signature"], stored), (
                    mode, gate, objective, form)
                checked += 1
    assert checked == len(MODES) * 40 * 2 * len(FORMS)


@pytest.mark.parametrize("module", ["fourier_score.gates", "fourier_score.config", "fourier_score.ldm.config"])
def test_config_modules_import_without_torch(module):
    code = f"import {module} as m, sys; assert 'torch' not in sys.modules, sorted(sys.modules); print(m.__file__)"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve().is_relative_to(REPO_ROOT / "fourier_score")
