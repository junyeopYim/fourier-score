"""Stdlib-only source fingerprint and file digests (no torch import)."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _anchor():
    here = Path(__file__).resolve()
    for folder in here.parents:
        package = folder / "fourier_score"
        complete = (folder / "pyproject.toml").is_file() and (package / "__init__.py").is_file()
        # Only the checkout of this file's own package, never another tree above it.
        if complete and here.is_relative_to(package):
            return folder
    raise RuntimeError(f"No pyproject.toml beside the fourier_score package holding {__file__}")


ROOT = _anchor()


def source_hash():
    paths = sorted((ROOT / "fourier_score").rglob("*.py"))
    if not paths:
        raise RuntimeError(f"No Python sources under {ROOT / 'fourier_score'}")
    if ROOT / "fourier_score" / "__init__.py" not in paths:
        raise RuntimeError(f"Missing {ROOT / 'fourier_score' / '__init__.py'}")
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
