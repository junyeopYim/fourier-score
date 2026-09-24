"""Plain-dict configs shared by the pixel and LDM pipelines (stdlib only).

The pytorch-template ``parse_config`` role without its ConfigParser object:
JSON files with ``extends`` inheritance, strict dotted ``KEY=VALUE``
overrides, a type walk against the defaults, ``CustomArgs`` flags that
expand into the same ``--set`` entries, and ``from_args`` choosing between a
config file and a resume checkpoint. Default messages are the pixel
pipeline's; callers pass their own where their dialect differs.
"""

from __future__ import annotations
from collections import namedtuple
import copy
import json
import math
from pathlib import Path

CustomArgs = namedtuple(
    "CustomArgs", "flags target type action choices help", defaults=(None,) * 4
)


def deep_merge(base: dict, patch: dict, prefix: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in out:
            raise ValueError(f"Unknown configuration key: {path}")
        if isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            out[key] = deep_merge(out[key], value, path)
        else:
            out[key] = value
    return out


def read_with_extends(
    path, defaults_fn, prepare=None, cycle="Configuration inheritance cycle: {path}"
) -> dict:
    """Merge each file onto its ``extends`` parent (relative to the declaring
    file); the chain's root goes onto ``defaults_fn(path, obj)``, or stays as
    is when that returns None. ``prepare`` rewrites every file before merging."""

    def read(path, stack):
        path = path.resolve()
        if path in stack:
            raise ValueError(cycle.format(path=path))
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("Configuration must be a JSON object")
        if prepare is not None:
            obj = prepare(obj)
        parent = obj.pop("extends", None)
        if parent is None:
            defaults = defaults_fn(path, obj)
            return obj if defaults is None else deep_merge(defaults, obj)
        if not isinstance(parent, str):
            raise ValueError("extends must be one JSON filename")
        return deep_merge(read(path.parent / parent, stack + (path,)), obj)

    return read(Path(path), ())


def apply_overrides(
    cfg: dict,
    entries,
    aliases=None,
    malformed="Expected key=value, got {entry!r}",
    unknown="Unknown configuration key: {path}",
) -> dict:
    """``KEY=VALUE`` with a JSON value (else the raw string) at an existing key."""
    cfg = copy.deepcopy(cfg)
    for entry in entries:
        if "=" not in entry:
            raise ValueError(malformed.format(entry=entry))
        path, raw = entry.split("=", 1)
        path = (aliases or {}).get(path, path)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parts = path.split(".")
        parent = cfg
        for key in parts[:-1]:
            if key not in parent or not isinstance(parent[key], dict):
                raise ValueError(unknown.format(path=path))
            parent = parent[key]
        if parts[-1] not in parent:
            raise ValueError(unknown.format(path=path))
        parent[parts[-1]] = value
    return cfg


def check_types(
    reference: dict,
    cfg: dict,
    invalid="{path}: invalid type {type}",
    nonfinite="{path} must be finite",
    prefix="",
):
    """Each non-null default fixes its leaf's type (int or float for a float).

    With ``nonfinite`` every float leaf must be finite (pixel); with None only
    leaves whose default is a float, reported as ``invalid`` (LDM).
    """
    for key, expected in reference.items():
        value = cfg[key]
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(expected, dict):
            check_types(expected, value, invalid, nonfinite, path)
            continue
        if isinstance(expected, float):
            good = type(value) in (int, float)
            if nonfinite is None:
                good = good and math.isfinite(value)
        else:
            good = expected is None or type(value) is type(expected)
        if not good:
            raise ValueError(invalid.format(path=path, type=type(value).__name__))
        if nonfinite and isinstance(value, float) and not math.isfinite(value):
            raise ValueError(nonfinite.format(path=path))


def add_options(parser, options):
    for option in options:
        kwargs = zip(option._fields[2:], option[2:])
        parser.add_argument(*option.flags, **{k: v for k, v in kwargs if v is not None})
    return parser


def cli_overrides(args, options) -> list[str]:
    """``--set`` entries, then each given option in order (so flags win)."""
    entries = list(args.set)
    for option in options:
        flag = next((f for f in option.flags if f.startswith("--")), option.flags[0])
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        if value is not None and value is not False:
            entries.append(f"{option.target}={'true' if value is True else value}")
    return entries


def from_args(parser, args, options, load, default, resume=None, conflict=None):
    """The template's ``from_args``; loaders are passed in, so torch stays out.

    ``changes`` are ``cli_overrides(args, options)``. Without ``-r`` this returns
    ``(load(config or default, changes), None)``; with it, what ``resume(path,
    changes)`` returns: ``(cfg, checkpoint)`` for training, the loaded model for
    inference. ``-c`` with ``-r`` is ``parser.error(conflict)``.
    """
    changes = cli_overrides(args, options)
    path = getattr(args, "resume", None)
    if path and args.config is not None:
        parser.error(conflict)
    if path:
        return resume(path, changes)
    return load(args.config or default, changes), None
