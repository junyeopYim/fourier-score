"""Experiment command line.

    python -m experiments list
    python -m experiments gmm NAME [flags ...] [--stage all|report]

NAME is a GMM experiment of experiments/gmm/registry.py; its flags are those
of the scripts/run_gmm_*.py runner it replaces (which now calls this main).
``--stage report`` rebuilds the report of a finished run without training,
into a --report directory outside --output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

STAGES = ("all", "report")


def gmm_parser(spec, prog):
    """The epoch-0 runner's flags (same names, defaults and order) plus --stage."""
    from experiments.gmm.registry import FixedGateExperiment

    parser = argparse.ArgumentParser(prog=prog, description=spec.description)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    fixed = isinstance(spec, FixedGateExperiment)
    if fixed and spec.reuse_nargs is None:
        parser.add_argument("--reuse-baselines", type=Path, help=spec.reuse_help)
    elif fixed:
        parser.add_argument("--reuse-baselines", type=Path, nargs=spec.reuse_nargs, default=[], help=spec.reuse_help)
    parser.add_argument("--preset", choices=("smoke", "experiment"), default="experiment")
    parser.add_argument("--workers", type=int, default=spec.workers, help=spec.workers_help)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    for flag, options in spec.options:
        parser.add_argument(flag, **options)
    if fixed:
        parser.add_argument("--test-bank-version", default=spec.test_bank)
    parser.add_argument("--stage", choices=STAGES, default="all",
                        help="report: rebuild the tables and figures of a finished run into a new --report "
                             "directory; nothing under --output is written")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from experiments.gmm.registry import EXPERIMENTS

    if argv == ["list"]:
        for spec in EXPERIMENTS.values():
            print(f"gmm {spec.name:<12} {spec.stub or '-':<34} {spec.description.splitlines()[0]}")
        return
    if len(argv) >= 2 and argv[0] == "gmm" and argv[1] in EXPERIMENTS:
        spec = EXPERIMENTS[argv[1]]
        invoked = Path(sys.argv[0]).name
        prog = (invoked if spec.stub and invoked == Path(spec.stub).name
                else f"python -m experiments gmm {spec.name}")
        parser = gmm_parser(spec, prog)
        args = parser.parse_args(argv[2:])
        from experiments.gmm.pipeline import run
        return run(spec, args, parser)
    stream = sys.stdout if argv[:1] in (["-h"], ["--help"]) else sys.stderr
    print(__doc__.strip() + "\n\nGMM experiments: " + ", ".join(EXPERIMENTS), file=stream)
    raise SystemExit(0 if stream is sys.stdout else 2)


if __name__ == "__main__":
    main()
