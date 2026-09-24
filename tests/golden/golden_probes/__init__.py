"""Golden contract probes shared by the recorder and the contract tests.

A probe computes JSON-serializable facts about the source tree it runs in.
``tests/golden/record_goldens.py`` runs the probes against the epoch-0 tree
(tag ``pre-template-refactor``) and writes ``tests/golden/<group>.json``;
``tests/contracts/test_contract_<group>.py`` recomputes them against the
working tree and compares.

Rules for probe authors:

* Import project code only through :func:`resolve` / :func:`symbol` with the
  NEW module path first and the epoch-0 path last, so one probe runs before
  and after modules move.
* Return ``{"exact": ..., "numeric": ...}`` (either key optional).  ``exact``
  holds facts that must match on every machine (strings, key sets, shapes,
  config digests, identities).  ``numeric`` holds float-derived facts (losses,
  tensor digests); they must match bit-for-bit on the recording machine and
  within ``rtol=1e-4`` elsewhere, where digest strings are skipped.
* Use ``ctx.root`` for repository paths and ``ctx.tmp`` for outputs.  Never
  write into ``ctx.root`` and never touch ``ctx.golden_root`` except to read.

Environment:

* ``FOURIER_GOLDEN_ROOT``: the original checkout (real checkpoints, datasets,
  caches; read-only).  ``local=True`` probes are skipped without it.
* ``FOURIER_GOLDEN_EXACT``: how ``numeric`` sections are compared.  Unset
  (default): bit-for-bit when :func:`machine` equals the recorded machine,
  else tolerance mode (digest strings skipped, floats within ``rtol=1e-4``).
  ``1`` forces bit-for-bit everywhere; ``0`` forces tolerance mode even on
  the recording machine (simulates CI / another machine).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

GOLDEN_DIR = Path(__file__).resolve().parents[1]
PAYLOAD_DIR = GOLDEN_DIR / "payloads"
EPOCH0_TAG = "pre-template-refactor"
EPOCH0_SOURCE_SHA256 = "f79717f3da852ffcde3ab8eea4997b3b44a3d0843e6cea040bde9afa145970a8"
GROUPS = ("config", "checkpoint", "numerics", "gmm", "local")

PROBES: dict[str, "Probe"] = {}
WRITERS: dict[str, "Probe"] = {}


@dataclass
class Probe:
    group: str
    name: str
    fn: object
    local: bool = False

    @property
    def key(self):
        return f"{self.group}.{self.name}"


@dataclass
class Context:
    """Where a probe runs: the tree under test, scratch space and payloads."""

    root: Path
    tmp: Path
    payload_dir: Path = PAYLOAD_DIR
    golden_root: Path | None = None
    recording: bool = False
    notes: list = field(default_factory=list)


def probe(group, name, *, local=False):
    """Register ``fn(ctx) -> {"exact": ..., "numeric": ...}``.

    ``local=True`` probes need ``FOURIER_GOLDEN_ROOT`` (real checkpoints,
    datasets or caches of the original checkout) and never run in CI.
    """

    def register(fn):
        item = Probe(group, name, fn, local)
        if item.key in PROBES:
            raise ValueError(f"Duplicate probe {item.key}")
        PROBES[item.key] = item
        return fn

    return register


def payload_writer(group, name, *, local=False):
    """Register ``fn(ctx)`` that writes epoch-0 payload files at record time.

    Writers run only in the recorder, before the probes, with
    ``ctx.payload_dir`` pointing at ``tests/golden/payloads``.  Probes then
    read those files in both modes (the recorder reads what it just wrote).
    """

    def register(fn):
        item = Probe(group, name, fn, local)
        if item.key in WRITERS:
            raise ValueError(f"Duplicate payload writer {item.key}")
        WRITERS[item.key] = item
        return fn

    return register


def load_group(group):
    """Import the probe module for ``group`` (registers its probes)."""
    importlib.import_module(f"golden_probes.{group}")
    return sorted(k for k, p in PROBES.items() if p.group == group)


def resolve(*candidates):
    """Import the first existing module among dotted ``candidates``.

    Only a missing *candidate* (or a missing parent package of it) moves on
    to the next candidate; a missing dependency inside an existing module
    propagates.
    """
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            missing = error.name or ""
            if missing and (name == missing or name.startswith(missing + ".")):
                continue
            raise
    raise ModuleNotFoundError(f"None of {candidates} can be imported")


def symbol(attr, *candidates):
    """``getattr(resolve(*candidates), attr)`` with the first module defining it."""
    for name in candidates:
        try:
            module = resolve(name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, attr):
            return getattr(module, attr)
    raise AttributeError(f"{attr} not found in any of {candidates}")


# ---------------------------------------------------------------- digests


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def tensor_digest(obj):
    """Order-independent sha256 of nested dicts/lists of tensors and primitives."""
    import torch

    h = hashlib.sha256()

    def walk(value):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(f"T{t.dtype}{tuple(t.shape)}".encode())
            if t.dtype == torch.bfloat16:
                t = t.view(torch.int16)
            h.update(t.numpy().tobytes())
        elif isinstance(value, dict):
            h.update(b"{")
            for k in sorted(value, key=str):
                h.update(str(k).encode() + b":")
                walk(value[k])
            h.update(b"}")
        elif isinstance(value, (list, tuple)):
            h.update(b"[")
            for v in value:
                walk(v)
            h.update(b"]")
        else:
            h.update(repr(value).encode())

    walk(obj)
    return h.hexdigest()


def shapes(state):
    """``{key: [dtype, shape]}`` of a state dict (key names and order matter)."""
    return [[k, str(v.dtype), list(v.shape)] for k, v in state.items()]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def floats(value, digits=None):
    """Tensor/number -> python floats (lists), for tolerance comparison."""
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().double().cpu().tolist()
    return value


# ---------------------------------------------------------------- machine


def machine():
    import torch

    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        model = platform.processor()
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": model,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
    }


def exact_numeric(recorded_machine):
    """Numeric facts are compared bit-for-bit only on the recording machine.

    ``FOURIER_GOLDEN_EXACT=1`` forces bit-for-bit and ``=0`` forces tolerance
    mode; any other value is an error so a typo cannot silently weaken a run.
    """
    forced = os.environ.get("FOURIER_GOLDEN_EXACT", "")
    if forced == "1":
        return True
    if forced == "0":
        return False
    if forced:
        raise ValueError(f"FOURIER_GOLDEN_EXACT must be 0, 1 or unset, not {forced!r}")
    return recorded_machine == machine()


# ---------------------------------------------------------------- running


def _torch_flags():
    """Process-global torch settings that project code (configure_runtime,
    the GMM runners) mutates; restored after every probe."""
    import torch

    flags = {
        "default_dtype": torch.get_default_dtype(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    # Mirror configure_runtime: never mix the legacy and fp32_precision APIs.
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        flags["matmul_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
        flags["conv_fp32_precision"] = torch.backends.cudnn.conv.fp32_precision
    else:
        flags["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        flags["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    return flags


def _restore_torch_flags(flags):
    import torch

    torch.set_default_dtype(flags["default_dtype"])
    torch.use_deterministic_algorithms(flags["deterministic"], warn_only=flags["deterministic_warn_only"])
    torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
    if "matmul_fp32_precision" in flags:
        torch.backends.cuda.matmul.fp32_precision = flags["matmul_fp32_precision"]
        torch.backends.cudnn.conv.fp32_precision = flags["conv_fp32_precision"]
    else:
        torch.backends.cuda.matmul.allow_tf32 = flags["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]


@contextmanager
def deterministic():
    """One CPU thread, restored global RNGs, cwd and torch flags around a probe."""
    import numpy as np
    import random
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("Golden probes require CUDA_VISIBLE_DEVICES= (empty)")
    threads = torch.get_num_threads()
    flags = _torch_flags()
    state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    cwd = os.getcwd()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        os.chdir(cwd)
        torch.set_num_threads(threads)
        _restore_torch_flags(flags)
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.random.set_rng_state(state[2])


def run(key, ctx):
    item = PROBES[key]
    with deterministic():
        scratch = Path(tempfile.mkdtemp(prefix=item.name + ".", dir=ctx.tmp))
        local = Context(ctx.root, scratch, ctx.payload_dir, ctx.golden_root, ctx.recording)
        result = item.fn(local)
    if not isinstance(result, dict) or not set(result) <= {"exact", "numeric"}:
        raise TypeError(f"Probe {key} must return a dict with exact/numeric keys")
    # Round-trip so live values compare like recorded JSON (tuples -> lists).
    return json.loads(json.dumps(result, allow_nan=True))


def write_payloads(group, ctx):
    for key, item in sorted(WRITERS.items()):
        if item.group != group or (item.local and ctx.golden_root is None):
            continue
        with deterministic():
            item.fn(ctx)


def golden_path(group):
    return GOLDEN_DIR / f"{group}.json"


def load_golden(group):
    return json.loads(golden_path(group).read_text())


def _looks_like_digest(value):
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(c in "0123456789abcdef" for c in value)
    )


def compare(expected, actual, *, exact, path="$"):
    """Raise AssertionError describing the first mismatch."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        assert sorted(expected) == sorted(actual), (
            f"{path}: keys differ; missing={sorted(set(expected) - set(actual))} "
            f"extra={sorted(set(actual) - set(expected))}"
        )
        for k in expected:
            compare(expected[k], actual[k], exact=exact, path=f"{path}.{k}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(expected) == len(actual), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            compare(e, a, exact=exact, path=f"{path}[{i}]")
        return
    if exact:
        same = expected == actual or (
            isinstance(expected, float) and isinstance(actual, float)
            and math.isnan(expected) and math.isnan(actual)
        )
        assert same and type(expected) is type(actual), f"{path}: {actual!r} != {expected!r}"
        return
    if _looks_like_digest(expected):
        return
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)), f"{path}: {actual!r} is not a number"
        if math.isnan(expected):
            assert math.isnan(actual), f"{path}: {actual!r} != nan"
            return
        assert math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-6), (
            f"{path}: {actual!r} != {expected!r} (rtol 1e-4)"
        )
        return
    assert expected == actual, f"{path}: {actual!r} != {expected!r}"


def check(group, key, ctx):
    """Compare one probe against its recorded golden (used by contract tests)."""
    golden = load_golden(group)
    assert key in golden["probes"], f"No golden recorded for {key}; rerun record_goldens.py"
    expected = golden["probes"][key]
    actual = run(key, ctx)
    assert sorted(expected) == sorted(actual), f"{key}: sections differ"
    if "exact" in expected:
        compare(expected["exact"], actual["exact"], exact=True, path=f"{key}.exact")
    if "numeric" in expected:
        compare(
            expected["numeric"],
            actual["numeric"],
            exact=exact_numeric(golden["_meta"]["machine"]),
            path=f"{key}.numeric",
        )
