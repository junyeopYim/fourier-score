"""Orchestration helpers shared by experiments: provenance, protocols, tables, figures, workers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from fourier_score.provenance import ROOT, file_sha256, source_hash
from fourier_score.utils import json_write

PACKAGE = Path(__file__).resolve().parent
# The only provenance that enters a checkpoint signature (fourier_score.gmm.train_arm).
NUMERICS = ("python", "torch", "numpy", "source_sha256")
# Provenance an existing protocol keeps on resume (see open_protocol).
RESUMABLE = ("git_revision", "git_dirty", "orchestration_sha256")


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def orchestration_provenance(extra_inputs=(), *, entry_points=()):
    """Numerics provenance plus sha256 of the orchestration code and of extra input files.

    ``orchestration_sha256`` covers every ``experiments/**/*.py`` and the
    ``entry_points`` (the ``scripts/`` stub of the experiment, recorded whether
    or not it was the invoking command; a missing one is an error); ``extra_inputs``
    are repository files the protocol depends on, such as a design proposal.
    """
    code = [*PACKAGE.rglob("*.py"), *(ROOT / p for p in entry_points)]
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                source_sha256=source_hash(),
                orchestration_sha256=dict(sorted((relative(p), file_sha256(p)) for p in code)),
                input_sha256={relative(ROOT / p): file_sha256(ROOT / p) for p in extra_inputs},
                git_revision=git("rev-parse", "HEAD").strip(),
                git_dirty=bool(git("status", "--porcelain")))


def numerics_provenance(provenance):
    return {key: provenance[key] for key in NUMERICS}


def open_protocol(output, protocol, *, label):
    """Write ``output/protocol.json``, or resume the identical protocol already there.

    An existing protocol must equal ``protocol`` after copying its ``RESUMABLE``
    provenance keys into ``protocol["provenance"]`` (in place); otherwise the
    output directory belongs to another protocol.  So the science fields,
    ``source_sha256`` (numerics), the python/torch/numpy versions and
    ``input_sha256`` must be unchanged, while orchestration edits (a runner,
    a table, a figure, another experiment's spec) and new commits may resume:
    checkpoint signatures hold numerics provenance only.  Returns this
    invocation's provenance as it was before the copy, for
    ``execution_history.json``.
    """
    path = Path(output) / "protocol.json"
    invocation = dict(protocol["provenance"])
    if path.exists():
        previous = json.loads(path.read_text())
        recorded = previous.get("provenance", {})
        if "runner_sha256" in recorded and "orchestration_sha256" not in recorded:
            raise ValueError(f"{path} is an epoch-0 {label} output (runner_sha256); resume or re-evaluate it "
                             "from the pre-template-refactor tree (../fs-epoch0), or choose a new output directory")
        for key in RESUMABLE:
            if key in recorded:
                protocol["provenance"][key] = recorded[key]
        if json.loads(json.dumps(protocol)) != previous:
            raise ValueError(f"Existing {label} protocol differs; choose a new output directory")
    json_write(protocol, path)
    return invocation


def read_protocol(output, protocol, *, label):
    """Report stage: the existing protocol, which must match ``protocol`` except for provenance."""
    path = Path(output) / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"No {label} protocol at {path}; run the experiment first")
    previous = json.loads(path.read_text())
    science = lambda p: {k: v for k, v in p.items() if k != "provenance"}  # noqa: E731
    if science(json.loads(json.dumps(protocol))) != science(previous):
        raise ValueError(f"Existing {label} protocol differs; pass the flags of the original run")
    return previous


def history_entry(**entry):
    return dict(started_utc=datetime.now(timezone.utc).isoformat(), **entry)


def read_history(output):
    path = Path(output) / "execution_history.json"
    return json.loads(path.read_text()) if path.exists() else []


def append_history(output, **entry):
    """Append one invocation to ``output/execution_history.json`` and return the history."""
    history = [*read_history(output), history_entry(**entry)]
    json_write(history, Path(output) / "execution_history.json")
    return history


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, directory, stem):
    """Save ``directory/stem.{png,svg,pdf}`` reproducibly and close ``fig``.

    SVG ids use a fixed salt and SVG/PDF dates come from ``SOURCE_DATE_EPOCH``
    (default 0), so reruns are byte-identical; PNG keeps matplotlib's default
    metadata.  SVG lines lose trailing whitespace, as in every earlier figure.
    """
    import matplotlib
    plt = pyplot()
    stamp = datetime.fromtimestamp(int(os.environ.get("SOURCE_DATE_EPOCH", "0")), timezone.utc)
    metadata = dict(svg={"Date": stamp.isoformat()}, pdf={"CreationDate": stamp})
    with matplotlib.rc_context({"svg.hashsalt": stem}):
        for extension in ("png", "svg", "pdf"):
            path = Path(directory) / f"{stem}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata.get(extension))
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def parallel_map(function, tasks, workers):
    """``map`` over spawned single-task processes; ``workers == 1`` runs inline."""
    if workers == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(function, tasks))
