"""GMM smoke chain: the commands and the output normalizer.

Shared by ``tests/golden/record_gmm_smoke.py`` (runs the epoch-0 runners and
writes ``tests/golden/gmm_smoke/``) and
``tests/contracts/test_contract_gmm_smoke.py`` (runs the current tree and
compares).  Import only the standard library here: the recorder runs this
module under the epoch-0 interpreter before any project code is imported.

The chain reproduces the reuse chain of the archived study
(``assets/gmm_*_comparison/protocol.json`` ``reuse_baselines``): each fixed
protocol reuses the controls of every earlier experiment, so only its new
arms are trained, and each reused checkpoint must be found in exactly one
earlier output (``reuse_directory``)::

    gated        (trains all 11 arms, selects one switch on validation)
    plateau      --reuse-baselines gated
    spectral     --reuse-baselines gated plateau
    gate_shapes  --reuse-baselines gated plateau spectral
    log_gates    --reuse-baselines gated plateau spectral gate_shapes

Normalization of one output directory (see :func:`normalize`) yields, per
golden file under ``tests/golden/gmm_smoke/<name>/``, either ``{"exact": v}``
(must match on every machine) or ``{"numeric": v}`` (bit-for-bit on the
recording machine, ``rtol=1e-4`` and digests skipped elsewhere; see
``golden_probes.exact_numeric``):

``files.json`` (exact)
    Every relative file path in the output directory, figures included.
``protocol.json`` (exact)
    ``protocol.json`` and ``report/protocol.json`` without ``provenance``.
``report.json`` (numeric)
    Every CSV (header + typed rows), ``selection.json``, ``summary.json``
    without ``provenance`` and ``report/checkpoint_audit.json`` without
    ``checkpoint_sha256`` (the payload embeds provenance),
    ``training_provenance`` and any ``origin_*`` field; other report JSON
    without ``provenance``.
``runs.json`` (numeric)
    Every per-arm ``metrics.json`` (training state without ``signature``,
    which digests the provenance, and without ``optimizer_seconds``),
    ``test.json`` and ``transition_test.json``.
``figures.json`` (numeric, only if deterministic)
    sha256 of every SVG after :func:`normalize_svg` (drops ``<dc:date>`` and
    renames matplotlib's salted ``m``/``p`` hash ids in order of first use).
    The recorder writes it only when two recordings agree; raw SVG and PDF
    bytes are never deterministic (date, random id salt).

Dropped entirely: ``execution_history.json`` (wall-clock time, provenance,
worker count), ``*.pt`` payloads (they embed provenance; their EMA digests
are in ``metrics.json`` and the audit), PNG/PDF bytes.

Every string has the chain's scratch directory replaced by ``<OUT>`` and the
repository root by ``<ROOT>``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

GOLDEN = Path(__file__).resolve().parents[1] / "gmm_smoke"
OUT, ROOT = "<OUT>", "<ROOT>"
SMOKE = ("--preset", "smoke", "--seeds", "42")
FIGURES = (".png", ".svg", ".pdf")
GOLDEN_FILES = ("files.json", "protocol.json", "report.json", "runs.json")
FIGURES_FILE = "figures.json"
PROTOCOL_FILES = ("protocol.json", "report/protocol.json")
SKIPPED_JSON = ("execution_history.json",)
AUDIT_DROP = ("checkpoint_sha256", "training_provenance")
METRICS_DROP = ("signature", "optimizer_seconds")


@dataclass(frozen=True)
class Link:
    """One experiment of the chain."""

    name: str      # experiments registry name and golden subdirectory
    script: str    # legacy runner (now a stub) under scripts/
    output: str    # output directory name; the runners store it as config.run_tag
    reuse: tuple   # earlier links whose outputs are the --reuse-baselines roots


CHAIN = (
    Link("gated", "run_gmm_comparison.py", "gmm_gated", ()),
    Link("plateau", "run_gmm_plateau.py", "gmm_plateau", ("gated",)),
    Link("spectral", "run_gmm_spectral_gate.py", "gmm_spectral", ("gated", "plateau")),
    Link("gate_shapes", "run_gmm_gate_shapes.py", "gmm_gate_shapes", ("gated", "plateau", "spectral")),
    Link("log_gates", "run_gmm_log_gates.py", "gmm_log_gates",
         ("gated", "plateau", "spectral", "gate_shapes")),
)
LINKS = {link.name: link for link in CHAIN}


# ---------------------------------------------------------------- commands


def link_args(link, base, *, workers=1):
    """CLI flags of ``link`` writing into ``base / link.output``."""
    args = ["--output", str(Path(base) / link.output)]
    if link.reuse:
        args += ["--reuse-baselines", *(str(Path(base) / LINKS[r].output) for r in link.reuse)]
    return [*args, *SMOKE, "--workers", str(workers)]


def stub_command(root, link, base, *, python=None, workers=1):
    """``python scripts/<runner>.py ...`` (the legacy entry point)."""
    return [python or sys.executable, str(Path(root) / "scripts" / link.script),
            *link_args(link, base, workers=workers)]


def experiments_command(link, base, *, python=None, workers=1):
    """``python -m experiments gmm <name> ...`` (run with cwd = repository root)."""
    return [python or sys.executable, "-m", "experiments", "gmm", link.name,
            *link_args(link, base, workers=workers)]


def placeholder_command(command, base, root, python=None):
    """``command`` with the interpreter, scratch directory and root as placeholders."""
    python = python or sys.executable
    return ["<PYTHON>" if part == python else _placeholders(part, base, root) for part in command]


def subprocess_env():
    """Environment for chain subprocesses: never touch a GPU, never write bytecode."""
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    return env


def run(command, cwd, log):
    """Run one link, appending its output to ``log``; raise with the log tail on failure."""
    start = time.perf_counter()
    with open(log, "a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        code = subprocess.run(command, cwd=cwd, env=subprocess_env(), stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    if code:
        tail = Path(log).read_text().splitlines()[-40:]
        raise RuntimeError(f"exit {code}: {' '.join(command)}\n" + "\n".join(tail))
    return time.perf_counter() - start


def run_chain(root, base, *, python=None, links=CHAIN, log=None):
    """Run the stub chain from ``root`` into ``base``; return seconds per link."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    log = log or base / "chain.log"
    return {link.name: run(stub_command(root, link, base, python=python), root, log) for link in links}


# ---------------------------------------------------------------- normalizer


def _placeholders(text, base, root):
    for path, token in sorted(((str(base), OUT), (str(root), ROOT)), key=lambda p: -len(p[0])):
        if path:
            text = text.replace(path, token)
    return text


def paths(obj, base, root):
    """Replace ``base``/``root`` in every string of a JSON value."""
    if isinstance(obj, str):
        return _placeholders(obj, base, root)
    if isinstance(obj, list):
        return [paths(v, base, root) for v in obj]
    if isinstance(obj, dict):
        return {paths(k, base, root): paths(v, base, root) for k, v in obj.items()}
    return obj


def _is_digest(text):
    return len(text) in (40, 64) and all(c in "0123456789abcdef" for c in text)


def cell(text):
    """CSV cell -> int, float or str (csv writes floats with repr, so this round-trips)."""
    if text == "" or _is_digest(text):
        return text
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def read_csv(path):
    with open(path, newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return {"columns": [], "rows": []}
    return {"columns": rows[0], "rows": [[cell(v) for v in row] for row in rows[1:]]}


def _without(obj, keys):
    return {k: v for k, v in obj.items() if k not in keys} if isinstance(obj, dict) else obj


def audit_row(row):
    return {k: v for k, v in row.items() if k not in AUDIT_DROP and not k.startswith("origin_")}


def normalize_svg(text):
    """SVG text without the save date and with matplotlib's salted ids renumbered."""
    text = re.sub(r"<dc:date>[^<]*</dc:date>", "<dc:date/>", text)
    ids = {}
    return re.sub(r"\b([mp])[0-9a-f]{10}\b",
                  lambda m: ids.setdefault(m.group(0), f"{m.group(1)}{len(ids):04d}"), text)


def svg_digest(path):
    return hashlib.sha256(normalize_svg(Path(path).read_text()).encode()).hexdigest()


def normalize(out_dir, base, root):
    """``{golden file name: {"exact"|"numeric": value}}`` for one output directory."""
    out_dir, base, root = Path(out_dir), Path(base), Path(root)
    files = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())
    protocol, report, runs, figures = {}, {}, {}, {}
    for rel in files:
        path = out_dir / rel
        name = path.name
        if rel in PROTOCOL_FILES:
            protocol[rel] = _without(json.loads(path.read_text()), ("provenance",))
        elif path.suffix == ".csv":
            report[rel] = read_csv(path)
        elif path.suffix == ".svg":
            figures[rel] = svg_digest(path)
        elif path.suffix in FIGURES or path.suffix == ".pt" or name in SKIPPED_JSON:
            continue
        elif path.suffix == ".json":
            value = json.loads(path.read_text())
            if name == "metrics.json":
                runs[rel] = _without(value, METRICS_DROP)
            elif name in ("test.json", "transition_test.json"):
                runs[rel] = value
            elif name == "checkpoint_audit.json":
                report[rel] = [audit_row(row) for row in value]
            else:
                report[rel] = _without(value, ("provenance",))
        else:
            raise ValueError(f"Unexpected file in GMM output: {path}")
    return {
        "files.json": {"exact": files},
        "protocol.json": {"exact": paths(protocol, base, root)},
        "report.json": {"numeric": paths(report, base, root)},
        "runs.json": {"numeric": paths(runs, base, root)},
        FIGURES_FILE: {"numeric": figures},
    }


def normalize_chain(base, root, links=CHAIN):
    return {link.name: normalize(Path(base) / link.output, base, root) for link in links}


def _scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


def dumps(obj):
    """Golden file text: sorted keys, one line per CSV row / flat record.

    Deterministic (byte-identical across recordings) and ``json.loads``-able;
    floats use ``repr`` so values round-trip exactly.
    """
    def one_line(value):
        return json.dumps(value, sort_keys=True, allow_nan=True, ensure_ascii=False)

    def encode(value, level):
        pad = " " * (level + 1)
        if isinstance(value, dict) and value:
            if all(_scalar(v) for v in value.values()) and len(one_line(value)) <= 240:
                return one_line(value)
            items = (f"{pad}{json.dumps(k, ensure_ascii=False)}: {encode(value[k], level + 1)}"
                     for k in sorted(value))
            return "{\n" + ",\n".join(items) + "\n" + " " * level + "}"
        if isinstance(value, list) and value and not all(_scalar(v) for v in value):
            return "[\n" + ",\n".join(pad + encode(v, level + 1) for v in value) + "\n" + " " * level + "]"
        return one_line(value)

    return encode(obj, 0) + "\n"


def load_golden(name):
    """``{golden file name: section dict}`` of one recorded link."""
    folder = GOLDEN / name
    return {p.name: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def load_meta():
    return json.loads((GOLDEN / "_meta.json").read_text())
