"""Objective names and Gaussian gate definitions; standard library only.

GATES holds one row per gate mode: its run-name suffix, the keys
resume_signature drops, and any constraint beyond the shared checks. A mode's
parameters are exactly the fields its suffix prints, so distinct settings get
distinct run directories; every mode shares GATE_DEFAULTS.

Adding a new gate mode should now mean: one table entry in gates.py + one
branch in FourierGaussian._gate_weights.
"""

from collections import namedtuple
import math


GAUSSIAN_OBJECTIVES = {
    "scalar_gaussian": {"covariance": "scalar"},
    "fourier_gaussian": {"covariance": "fourier"},
}
OBJECTIVES = ("score", "diffusion", *GAUSSIAN_OBJECTIVES)
# Score and diffusion are sign conventions, so only one is in the main ablation.
COMPARISON_OBJECTIVES = ("score", *GAUSSIAN_OBJECTIVES)

GATE_DEFAULTS = {"mode": "none", "sigma_switch": 1.0, "sharpness": 4.0, "value": 1.0,
                 "sigma_lo": 0.8, "sigma_hi": 1.0, "delta": 0.5}
GATE_PARAMS = tuple(key for key in GATE_DEFAULTS if key != "mode")


def _bounded(gate):
    if not gate["sigma_lo"] < gate["sigma_switch"] < gate["sigma_hi"]:
        raise ValueError("bounded gate requires sigma_lo < sigma_switch < sigma_hi")
    if gate["sharpness"] <= 1:
        raise ValueError("bounded gate sharpness must exceed one for flat endpoint slopes")


Gate = namedtuple("Gate", "suffix drops check", defaults=(None,))
# drops: keys resume_signature ignores (all unread by the mode, some newer
# than it); None drops the whole gate, as pre-gate v1 runs had none. Editing
# an existing row changes the signatures its recorded runs resume against.
GATES = {
    "none": Gate("", None),
    "constant": Gate("_gate_constant{value}", ("sigma_lo", "sigma_hi", "delta")),
    "log_sigma": Gate("_gate_s{sigma_switch}_p{sharpness}", ("sigma_lo", "sigma_hi", "delta")),
    "log_sigma_plateau": Gate("_gate_s{sigma_switch}_p{sharpness}_plateau_lo{sigma_lo}_hi{sigma_hi}",
                              ("delta",)),
    "spectral_cap": Gate("_gate_s{sigma_switch}_p{sharpness}_cap_lo{sigma_lo}_hi{sigma_hi}_d{delta}", ()),
    "linear_sigma": Gate("_gate_s{sigma_switch}_p{sharpness}_linear_sigma", ("sigma_lo", "sigma_hi", "delta")),
    "tanh_sigma": Gate("_gate_s{sigma_switch}_p{sharpness}_tanh_sigma", ("sigma_lo", "sigma_hi", "delta")),
    "linear_log_sigma": Gate("_gate_loglinear_lo{sigma_lo}_hi{sigma_hi}", ("sigma_switch", "sharpness", "delta")),
    "bounded_log_sigmoid": Gate("_gate_logsigmoid_lo{sigma_lo}_hi{sigma_hi}_s{sigma_switch}_p{sharpness}",
                                ("delta",), _bounded),
}
GATE_MODES = tuple(GATES)


def validate_gate(gate=None):
    """A disabled gate means the original reference coefficient g=1."""
    if gate is not None and (not isinstance(gate, dict) or set(gate) - GATE_DEFAULTS.keys()):
        raise ValueError("Invalid Gaussian gate configuration")
    result = {**GATE_DEFAULTS, **(gate or {})}
    # A tuple, not the dict: unhashable modes must fail with this message.
    if result["mode"] not in GATE_MODES:
        raise ValueError(f"gate.mode must be {', '.join(GATE_MODES[:-1])}, or {GATE_MODES[-1]}")
    for key in GATE_PARAMS:
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
    check = GATES[result["mode"]].check
    if check is not None:
        check(result)
    return result


def _printed(key, value):
    # Bounds use the shortest round-trip repr so distinct bounds stay distinct.
    text = repr(float(value)).removesuffix(".0") if key in ("sigma_lo", "sigma_hi") else format(value, ".17g")
    return text.replace(".", "p").replace("-", "m").replace("+", "")


def gate_suffix(gate=None):
    gate = validate_gate(gate)
    return GATES[gate["mode"]].suffix.format(**{key: _printed(key, gate[key]) for key in GATE_PARAMS})


def gate_signature_pops(gate):
    """Gate keys resume_signature drops; None drops the whole gate."""
    return GATES[gate.get("mode", "none")].drops
