"""Bitwise comparison of two run directories (or two files), ignoring provenance.

    python scripts/compare_runs.py A B [--ignore-key PATTERN ...] [--ignore-file PATTERN ...]

* ``*.pt``/``*.pth``/``*.ckpt``: ``torch.load(weights_only=True)``, compared
  recursively; tensors must match dtype, shape and bytes (NaN-safe).
* ``*.json`` / ``*.jsonl``: parsed and compared line by line.
* everything else (``.npz``, ``.npy``, ``.png``, ...): raw bytes.

Dict keys matching an ignore pattern (``fnmatch``, at any depth, in ``.pt``
and JSON alike) are dropped first: provenance (source/git/environment) and
wall-clock/throughput fields.  Prints one line per difference and a summary;
exits 1 when anything differs or exists on one side only.
"""

import argparse
from fnmatch import fnmatch
import hashlib
import json
import math
from pathlib import Path
import sys

TENSOR_SUFFIXES = {".pt", ".pth", ".ckpt"}
IGNORE_KEYS = (
    "source_sha256", "runner_sha256", "orchestration_sha256", "environment", "provenance", "git_*",
    "*_seconds", "*_utc", "time", "*_time", "timestamp", "eta_*", "*_eta", "*per_second*",
    "throughput*", "*_gib", "hostname",
)
IGNORE_FILES = ("events.out.tfevents.*",)


def load(path):
    if path.suffix in TENSOR_SUFFIXES:
        import torch

        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - not weights_only-loadable: compare bytes
            return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diff(a, b, ignore, where="", out=None):
    """Append ``where: description`` for each difference between ``a`` and ``b``."""
    out, at = ([] if out is None else out), where or "."
    if type(a) is not type(b):
        out.append(f"{at}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        keys = lambda d: {k for k in d if not any(fnmatch(str(k), p) for p in ignore)}  # noqa: E731
        ka, kb = keys(a), keys(b)
        if ka != kb:
            out.append(f"{at}: keys only in A {sorted(map(str, ka - kb))}, only in B {sorted(map(str, kb - ka))}")
        for k in sorted(ka & kb, key=str):
            diff(a[k], b[k], ignore, f"{where}.{k}", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{at}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, ignore, f"{where}[{i}]", out)
    elif hasattr(a, "dtype") and hasattr(a, "untyped_storage"):  # torch.Tensor
        import torch

        if a.dtype != b.dtype or a.shape != b.shape:
            out.append(f"{at}: tensor {a.dtype}{list(a.shape)} != {b.dtype}{list(b.shape)}")
        else:
            raw = lambda t: t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)  # noqa: E731
            if not torch.equal(raw(a), raw(b)):
                changed = (a != b) & ~(a.isnan() & b.isnan()) if a.is_floating_point() else a != b
                out.append(f"{at}: tensor {a.dtype}{list(a.shape)} differs in {int(changed.sum())} elements")
    elif isinstance(a, float):
        if repr(a) != repr(b) and not (math.isnan(a) and math.isnan(b)):
            out.append(f"{at}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{at}: {a!r} != {b!r}")
    return out


def files(root, ignore_files):
    return {
        p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*"))
        if p.is_file() and not any(fnmatch(p.name, pattern) for pattern in ignore_files)
    }


def compare(a, b, ignore=IGNORE_KEYS, ignore_files=IGNORE_FILES):
    """``({relative path: [differences]}, number of files)`` for two files or two dirs."""
    if a.is_file() and b.is_file():
        fa, fb = {a.name: a}, {a.name: b}
    else:
        fa, fb = files(a, ignore_files), files(b, ignore_files)
    report = {rel: ["only in A"] for rel in fa.keys() - fb.keys()}
    report.update({rel: ["only in B"] for rel in fb.keys() - fa.keys()})
    for rel in sorted(fa.keys() & fb.keys()):
        found = diff(load(fa[rel]), load(fb[rel]), ignore)
        if found:
            report[rel] = found
    return dict(sorted(report.items())), len(fa.keys() | fb.keys())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--ignore-key", action="append", default=[], metavar="PATTERN",
                        help=f"Also drop dict keys matching PATTERN (default drops {', '.join(IGNORE_KEYS)})")
    parser.add_argument("--ignore-file", action="append", default=[], metavar="PATTERN",
                        help=f"Skip file names matching PATTERN (default {', '.join(IGNORE_FILES)})")
    parser.add_argument("--max-lines", type=int, default=20, help="Differences printed per file")
    args = parser.parse_args(argv)
    for path in (args.a, args.b):
        if not path.exists():
            parser.error(f"{path} does not exist")
    if args.a.is_file() != args.b.is_file():
        parser.error("Compare two files or two directories")
    report, total = compare(args.a, args.b, (*IGNORE_KEYS, *args.ignore_key), (*IGNORE_FILES, *args.ignore_file))
    for rel, found in report.items():
        print(f"DIFF {rel}")
        for line in found[: args.max_lines]:
            print(f"  {line}")
        if len(found) > args.max_lines:
            print(f"  ... {len(found) - args.max_lines} more")
    print(f"{total} files compared, {len(report)} differ")
    return 1 if report else 0


if __name__ == "__main__":
    sys.exit(main())
