"""What each GMM gate experiment runs: arm suites, presets, figure styles, experiment specs.

Every experiment compares its new gate arms with the arms of the experiments
before it (``SUITES``): the controls of experiment ``k`` are suite ``k - 1``,
in the same order, and are reused read-only from earlier outputs when
``--reuse-baselines`` names them.  Everything numerical lives in
``fourier_score.gmm``; a spec only records how one experiment differs from
the others.  Adding an experiment is described in ``experiments/README.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from typing import Callable, NamedTuple

import numpy as np

from fourier_score.gmm import (
    BASELINE_ARMS, GMMConfig, gated_arm, log_gate_arm, plateau_arm, shaped_gate_arm, spectral_cap_arm,
)
from experiments.common import ROOT

COVARIANCES = ("scalar", "fourier")
SHAPED_MODES = ("linear_sigma", "tanh_sigma")
LOG_MODES = ("linear_log_sigma", "bounded_log_sigmoid")
TRAIN_BANK = "gmm-gated-v1"    # training/validation banks of every experiment below
ORACLE_BANK = "gmm-oracle-v1"  # notebooks/gmm_oracle.ipynb
DESIGN = "assets/log_gate_design/design.json"  # proposal the log gates were fixed to
CONSTRAINTS = dict(backbones_per_model=1, loss="normalized_residual for new arms",
                   teachers=False, auxiliary_losses=False, pretrained_initialization=False)


def pair(factory, *args, **kwargs):
    """``factory(covariance, ...)`` for the scalar and the Fourier covariance."""
    return tuple(factory(cov, *args, **kwargs) for cov in COVARIANCES)


# ------------------------------------------------------------------ arm suites

SUITES = {"baselines": BASELINE_ARMS}
SUITES["gated"] = (*SUITES["baselines"], *pair(gated_arm, 1.5, 4.))  # the switch the gated run selected
SUITES["plateau"] = (*SUITES["gated"], *pair(plateau_arm))
SUITES["spectral"] = (*SUITES["plateau"], *pair(spectral_cap_arm))
SUITES["gate_shapes"] = (*SUITES["spectral"],
                         *(shaped_gate_arm(cov, mode) for mode in SHAPED_MODES for cov in COVARIANCES))
SUITES["log_gates"] = (*SUITES["gate_shapes"],
                       *(log_gate_arm(cov, mode) for mode in LOG_MODES for cov in COVARIANCES))


def base_config(preset, seeds, bank_version=TRAIN_BANK):
    """The GMMConfig of every runner; ``methods`` and ``run_tag`` are set per run."""
    cfg = GMMConfig(preset=preset, steps=5000, eval_every=500, width=192,
                    seeds=tuple(seeds), spectrum_lambdas=(0., .5, 1.), distributions=("gmm",),
                    n_noise_levels=9, val_per_noise=512, test_per_noise=2048,
                    cpu_threads=1, device="cpu", bank_version=bank_version)
    if preset == "smoke":
        cfg = replace(cfg, image_size=4, width=32, depth=2, steps=12, eval_every=6,
                      batch_size=32, n_noise_levels=3, val_per_noise=32, test_per_noise=64,
                      spectrum_lambdas=(0., 1.))
    return cfg


# ------------------------------------------------------------------ figure styles

# Baselines by arm name: (legend label, colour, line style).
BASELINE_STYLES = {
    "score": ("Score / DSM", "#777777", ":"),
    "scalar_gaussian": ("Scalar / DSM", "#3178ad", "--"),
    "fourier_gaussian": ("Fourier / DSM", "#d77b29", "--"),
    "scalar_normalized": ("Scalar / normalized", "#3178ad", "-"),
    "fourier_normalized": ("Fourier / normalized", "#d77b29", "-"),
}
# Gated arms by gate_mode: (label, (scalar, Fourier) colours, (scalar, Fourier) line styles).
STYLES = {
    "log_sigma": ("Gated", ("#18866c", "#863daf"), ("-", "-")),
    "log_sigma_plateau": ("Plateau", ("#2382c0", "#c13958"), ("-", "-")),
    "spectral_cap": ("Spectral", ("#6e9b28", "#111111"), ("-", "-")),
    "linear_sigma": ("Linear", ("#936b45", "#703000"), ("--", "-")),
    "tanh_sigma": ("Tanh", ("#cf72af", "#bc087e"), ("--", "-")),
    "linear_log_sigma": ("Log linear", ("#999999", "#111111"), ("--", "-")),
    "bounded_log_sigmoid": ("Bounded S", ("#e5bd47", "#b98b00"), ("--", "-")),
}
# Fourier colour used instead when another gate in the same figure owns the default one.
ALTERNATE_COLORS = {"spectral_cap": "#657078"}
# The gate-shape diagnostic figures of gate_shapes and log_gates were drawn with their
# own legend; kept (Diagnostics.styles of those two specs only) so regenerated figures
# match assets/.  gate_mode -> (label or None, colour or None) overriding style().
SHAPE_DIAGNOSTIC_STYLES = {"none": ("Fourier normalized", None), "log_sigma": ("Sigmoid Fourier", None),
                           "bounded_log_sigmoid": (None, "#d8a000")}


def style(arm, arms=()):
    """(label, colour, line style) of ``arm`` in a figure that also draws ``arms``."""
    if arm.gate_mode == "none":
        return BASELINE_STYLES[arm.name]
    label, colors, styles = STYLES[arm.gate_mode]
    fourier = arm.parameterization == "fourier_gaussian"
    color = colors[fourier]
    if fourier and arm.gate_mode in ALTERNATE_COLORS and any(
            a.gate_mode != arm.gate_mode and STYLES.get(a.gate_mode, ("", ("", "")))[1][1] == color
            for a in arms):
        color = ALTERNATE_COLORS[arm.gate_mode]
    return f"{label} {'Fourier' if fourier else 'Scalar'}", color, styles[fourier]


# ------------------------------------------------------------------ experiments

FLOAT = dict(type=float)
REUSE_CONTROLS = "Checkpoint roots for evaluation controls only; never used as teachers"


class Diagnostics(NamedTuple):
    """The diagnostic figure of a fixed-gate experiment (``report.plot_diagnostics``)."""
    layout: str                   # "transition" (plateau), "spectral" or "shapes": see report.LAYOUTS
    figure_stem: str
    gate_modes: tuple | None = None  # Fourier normalized arms drawn, by gate mode; None: all
    styles: dict | None = None    # gate_mode -> (label, colour) overrides; legacy figures only


@dataclass(frozen=True, kw_only=True)
class Experiment:
    name: str
    description: str              # --help text (the epoch-0 runner docstring for the legacy specs)
    report: str                   # prose write-up of the results
    figure_stem: str              # main comparison figure in the report directory
    options: tuple                # experiment flags: ((flag, argparse kwargs), ...), --help order
    new_modes: tuple              # gate modes trained here; never reused from earlier outputs
    new_arms: Callable            # args -> arms trained by this experiment
    controls: Callable            # args -> arms of earlier experiments (usually SUITES[previous])
    test_bank: str                # default test bank version
    criterion: str                # selection rule recorded in the protocol
    stub: str | None = None       # legacy scripts/ entry point (fingerprinted with the orchestration)
    label: str | None = None      # "Existing <label> protocol differs; ..."; defaults to name
    workers: int = 16             # default --workers
    workers_help: str | None = None
    inputs: tuple = ()            # repository files the protocol depends on (input_sha256)

    def __post_init__(self):
        if self.label is None:
            object.__setattr__(self, "label", self.name)


@dataclass(frozen=True, kw_only=True)
class SelectionExperiment(Experiment):
    """Train several candidate gates, select one on validation, then test the selection."""
    candidates: Callable          # (args, sigma_switch) -> gated arms of one candidate


@dataclass(frozen=True, kw_only=True)
class FixedGateExperiment(Experiment):
    """Gates fixed before training; every arm is tested on a new common test bank.

    The protocol's selection lists one candidate per new gate mode when there
    are several, else one candidate.
    """
    check: Callable               # (cfg, args) -> parser error message or None
    transition: Callable          # (cfg, args) -> sigma grid of the transition diagnostic
    fixed: tuple                  # argument names recorded as the fixed proposal
    diagnostics: Diagnostics
    extras: Callable = lambda args, new_arms: {}  # extra protocol fields, in order
    parameter_counts: bool = True  # assert identical parameter counts across arms
    reuse_nargs: str | None = "*"  # "*": several roots; None: one root (plateau)
    reuse_help: str = REUSE_CONTROLS


def within(lo, hi):
    return lambda cfg, args: (None if cfg.sigma_min <= getattr(args, lo) < getattr(args, hi) <= cfg.sigma_max
                              else "Transition must lie within the training sigma range")


def plateau_transition(cfg, args):
    # Include both exact endpoints and a dense interior, plus nearby context.
    return sorted(set([max(cfg.sigma_min, args.sigma_lo * .8),
                       max(cfg.sigma_min, args.sigma_lo * .95),
                       *np.linspace(args.sigma_lo, args.sigma_hi, 11).tolist(),
                       min(cfg.sigma_max, args.sigma_hi * 1.05),
                       min(cfg.sigma_max, args.sigma_hi * 1.2)]))


def spectral_transition(cfg, args):
    return sorted(set([max(cfg.sigma_min, args.sigma_lo * .6),
                       max(cfg.sigma_min, args.sigma_lo * .8),
                       *np.geomspace(args.sigma_lo, args.sigma_hi, 13).tolist(),
                       min(cfg.sigma_max, args.sigma_hi * 1.2)]))


def shape_transition(cfg, args):
    return sorted({s for s in [*np.linspace(.5, 2.5, 17).tolist(), .8, 1.,
                               args.sigma_switch,
                               args.sigma_switch*(1-2/args.sharpness),
                               args.sigma_switch*(1+2/args.sharpness)]
                   if cfg.sigma_min <= s <= cfg.sigma_max})


def log_transition(cfg, args):
    return sorted({s for s in [.1, .2, .3, .4, .5, .6, .7, .75, .8, .9, 1., 1.1,
                               1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2., 2.5, 3.,
                               args.sigma_lo, args.sigma_switch, args.sigmoid_hi, args.linear_hi]
                   if cfg.sigma_min <= s <= cfg.sigma_max})


def shape_extras(args, new_arms):
    return dict(gate_definitions=dict(
                    linear_sigma="clip(1/2 + (p/4)*(sigma/sigma_c - 1), 0, 1)",
                    tanh_sigma="(1 + tanh((p/2)*(sigma/sigma_c - 1)))/2",
                    center_value=.5, center_slope="p/(4*sigma_c)",
                    log_tanh_identity="(1+tanh((p/2)*log(sigma/sigma_c)))/2 equals existing sigmoid; no duplicate training"),
                constraints=dict(CONSTRAINTS))


def log_extras(args, new_arms):
    proposal = json.loads((ROOT / DESIGN).read_text())["arms"]
    return dict(matches_prior_design_proposal=[asdict(a) for a in new_arms
                                               if a.parameterization == "fourier_gaussian"] == proposal,
                gate_definitions=dict(
                    linear_log_sigma="clip(log(sigma/0.1)/log(3/0.1), 0, 1), with configurable bounds",
                    bounded_log_sigmoid="a^k/(a^k+b^k), a=z*(1-zc), b=(1-z)*zc; z is clipped normalized log noise",
                    sigmoid_tanh_equivalent=True, duplicate_tanh_training=False),
                constraints=dict(CONSTRAINTS))


GATED = SelectionExperiment(
    name="gated", stub="scripts/run_gmm_comparison.py", report="reports/gmm/gated-comparison.md",
    description="""Train paired GMM baselines/gate candidates, select on validation, then test.

The default protocol has 99 training runs and 63 final test evaluations.
Checkpoints stay in --output; compact tables and figures go in --report.
""",
    label="comparison", figure_stem="gmm_gated_comparison",
    workers=3, workers_help="Concurrent single-threaded runs; each arm is scheduled independently",
    options=(("--switches", dict(type=float, nargs="+", default=[.5, 1., 1.5])),
             ("--sharpness", dict(type=float, default=4.)),
             ("--bank-version", dict(default=TRAIN_BANK))),
    new_modes=("log_sigma",),
    candidates=lambda args, switch: pair(gated_arm, switch, args.sharpness),
    new_arms=lambda args: tuple(arm for switch in args.switches for arm in pair(gated_arm, switch, args.sharpness)),
    controls=lambda args: SUITES["baselines"],
    test_bank=TRAIN_BANK,
    criterion="One common switch minimizes final validation mean over both covariances, all spectra and seeds; test only after selection",
)

PLATEAU = FixedGateExperiment(
    name="plateau", stub="scripts/run_gmm_plateau.py", report="reports/gmm/plateau-comparison.md",
    description="""Paired, fixed-protocol GMM experiment for an exactly saturating noise gate.

The default experiment fixes sigma_switch=1.5, p=4 and transition [0.8, 1.0]
before evaluation. It compares five baselines, two sigmoid gates and two
plateau gates. Compatible baseline checkpoints can be audited and reused;
every method is evaluated on new common test observations.
""",
    figure_stem="gmm_plateau_comparison", workers=12,
    reuse_nargs=None, reuse_help="Read-only reuse after matching configs, EMA hashes and validation outputs",
    options=(("--sigma-switch", dict(FLOAT, default=1.5)), ("--sharpness", dict(FLOAT, default=4.)),
             ("--sigma-lo", dict(FLOAT, default=.8)), ("--sigma-hi", dict(FLOAT, default=1.))),
    new_modes=("log_sigma_plateau",),
    new_arms=lambda a: pair(plateau_arm, a.sigma_switch, a.sharpness, a.sigma_lo, a.sigma_hi),
    # The sigmoid control follows --sigma-switch/--sharpness (SUITES["gated"] at the defaults).
    controls=lambda a: (*SUITES["baselines"], *pair(gated_arm, a.sigma_switch, a.sharpness)),
    test_bank="gmm-plateau-v1", check=within("sigma_lo", "sigma_hi"), transition=plateau_transition,
    fixed=("sigma_switch", "sharpness", "sigma_lo", "sigma_hi"),
    criterion="Fixed proposal before training; no threshold search or test selection",
    parameter_counts=False, diagnostics=Diagnostics("transition", "gmm_plateau_transition"),
)

SPECTRAL = FixedGateExperiment(
    name="spectral", stub="scripts/run_gmm_spectral_gate.py", report="reports/gmm/spectral-gate.md",
    description="""Fixed GMM experiment: one backbone, one normalized MSE, spectral gate.

Fix delta=0.5 and a log-noise transition [1,2] before training. Compare to
the five original arms, sigmoid gates and the earlier [0.8,1] plateau gates.
Previously trained controls may be audited and reused for evaluation only.
""",
    label="spectral-gate", figure_stem="gmm_spectral_comparison", workers=12,
    options=(("--sigma-switch", dict(FLOAT, default=1.5)), ("--sharpness", dict(FLOAT, default=4.)),
             ("--sigma-lo", dict(FLOAT, default=1.)), ("--sigma-hi", dict(FLOAT, default=2.)),
             ("--delta", dict(FLOAT, default=.5))),
    new_modes=("spectral_cap",),
    new_arms=lambda a: pair(spectral_cap_arm, a.sigma_switch, a.sharpness, a.sigma_lo, a.sigma_hi, a.delta),
    # Sigmoid and plateau controls follow --sigma-switch/--sharpness (SUITES["plateau"] at the defaults).
    controls=lambda a: (*SUITES["baselines"], *pair(gated_arm, a.sigma_switch, a.sharpness),
                        *pair(plateau_arm, a.sigma_switch, a.sharpness, .8, 1.)),
    test_bank="gmm-spectral-cap-v1", check=within("sigma_lo", "sigma_hi"), transition=spectral_transition,
    fixed=("sigma_switch", "sharpness", "sigma_lo", "sigma_hi", "delta"),
    criterion="Fixed proposal before training; no search or test selection",
    extras=lambda args, new_arms: dict(constraints=dict(CONSTRAINTS)),
    diagnostics=Diagnostics("spectral", "gmm_spectral_diagnostics"),
)

GATE_SHAPES = FixedGateExperiment(
    name="gate_shapes", stub="scripts/run_gmm_gate_shapes.py", report="reports/gmm/linear-tanh-gates.md",
    description="""Fixed paired GMM comparison of sigma-linear and sigma-tanh gates.

Match the sigmoid's center and local slope (sigma_c=1.5, p=4). New arms
use one randomly initialized backbone and one normalized residual MSE.
Existing controls can be audited and reused for evaluation only.
""",
    label="gate-shape", figure_stem="gmm_gate_shape_comparison",
    options=(("--sigma-switch", dict(FLOAT, default=1.5)), ("--sharpness", dict(FLOAT, default=4.))),
    new_modes=SHAPED_MODES,
    new_arms=lambda a: tuple(shaped_gate_arm(cov, mode, a.sigma_switch, a.sharpness)
                             for mode in SHAPED_MODES for cov in COVARIANCES),
    controls=lambda a: SUITES["spectral"],
    test_bank="gmm-linear-tanh-v1",
    check=lambda cfg, a: (None if cfg.sigma_min < a.sigma_switch < cfg.sigma_max
                          else "Gate center must lie within the training sigma range"),
    transition=shape_transition, fixed=("sigma_switch", "sharpness"),
    criterion="Fixed before training; no search or test selection",
    extras=shape_extras,
    diagnostics=Diagnostics("shapes", "gmm_gate_shape_diagnostics", styles=SHAPE_DIAGNOSTIC_STYLES),
)

LOG_GATES = FixedGateExperiment(
    name="log_gates", stub="scripts/run_gmm_log_gates.py", report="reports/gmm/log-axis-gates.md",
    description="""Paired GMM experiment for the approved log-linear and bounded S gates.

Keep the prior design fixed: linear over [0.1,3], bounded sigmoid over
[0.1,1.5], center 0.75, exponent 2. Use one backbone and one normalized MSE.
Existing checkpoints are comparison controls, never teachers or initializers.
""",
    label="log-gate", figure_stem="gmm_log_gate_comparison",
    reuse_help="Checkpoint roots for audited comparison controls only",
    options=(("--sigma-lo", dict(FLOAT, default=.1)), ("--linear-hi", dict(FLOAT, default=3.)),
             ("--sigmoid-hi", dict(FLOAT, default=1.5)), ("--sigma-switch", dict(FLOAT, default=.75)),
             ("--sharpness", dict(FLOAT, default=2.))),
    new_modes=LOG_MODES,
    new_arms=lambda a: tuple(log_gate_arm(cov, mode, sigma_lo=a.sigma_lo,
                                          sigma_hi=a.linear_hi if mode == "linear_log_sigma" else a.sigmoid_hi,
                                          sigma_switch=a.sigma_switch, sharpness=a.sharpness)
                             for mode in LOG_MODES for cov in COVARIANCES),
    controls=lambda a: SUITES["gate_shapes"],
    test_bank="gmm-log-gates-v1", inputs=(DESIGN,),
    check=lambda cfg, a: (None if all(cfg.sigma_min <= a.sigma_lo < hi <= cfg.sigma_max
                                      for hi in (a.linear_hi, a.sigmoid_hi))
                          else "Gate bounds must lie within the training sigma range"),
    transition=log_transition, fixed=("sigma_lo", "linear_hi", "sigmoid_hi", "sigma_switch", "sharpness"),
    criterion="Prior plotted proposal fixed before training; no search or test selection",
    extras=log_extras,
    diagnostics=Diagnostics("shapes", "gmm_log_gate_diagnostics",
                            ("none", "log_sigma", "log_sigma_plateau", "spectral_cap", *LOG_MODES),
                            SHAPE_DIAGNOSTIC_STYLES),
)

# In chronological order: each experiment's controls are the previous suite.
EXPERIMENTS = {spec.name: spec for spec in (GATED, PLATEAU, SPECTRAL, GATE_SHAPES, LOG_GATES)}


def forbidden_test_banks(spec):
    """Bank versions whose observations an experiment must not reuse as its test bank:
    the test banks of every earlier experiment and the oracle notebook's bank."""
    earlier = list(EXPERIMENTS)[:list(EXPERIMENTS).index(spec.name)]
    return {EXPERIMENTS[name].test_bank for name in earlier} | {ORACLE_BANK}
