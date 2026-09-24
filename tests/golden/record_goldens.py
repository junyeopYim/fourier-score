"""Record golden contract values from the epoch-0 source tree.

Run with the epoch-0 worktree's interpreter so the probes import epoch-0 code:

    CUDA_VISIBLE_DEVICES= ../fs-epoch0/.venv/bin/python tests/golden/record_goldens.py \
        --root ../fs-epoch0 [--group config ...] [--golden-root /path/to/original/checkout]

The tree under ``--root`` must be the ``pre-template-refactor`` tag with a
clean ``fourier_score/`` whose source hash is the epoch-0 hash.  Goldens and
payloads are written next to this script (``tests/golden``) in the tree that
contains it, never into ``--root``.

Recording is reproducible: two recordings on the same machine write
byte-identical ``*.json`` goldens and payload files.  For that the scratch
directory is the fixed ``/tmp/fourier-golden-record`` (payload checkpoints
embed the paths they were trained under; wiped before and after, guarded by
a lock so two recordings cannot share it).  Bytecode writing is disabled for
this process and every subprocess, so nothing (``__pycache__``) is created
inside the frozen ``--root`` tree.
"""

import os
import sys

# Before anything is imported from --root: never write __pycache__ there.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse  # noqa: E402
from contextlib import contextmanager  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

HERE = Path(__file__).resolve().parent
RECORD_TMP = Path("/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()) / "fourier-golden-record"


@contextmanager
def record_tmp():
    """The fixed, exclusively locked scratch directory of one recording."""
    lock = open(str(RECORD_TMP) + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another recording holds {RECORD_TMP}.lock") from None
    shutil.rmtree(RECORD_TMP, ignore_errors=True)
    RECORD_TMP.mkdir()
    try:
        yield RECORD_TMP
    finally:
        shutil.rmtree(RECORD_TMP, ignore_errors=True)
        lock.close()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="epoch-0 worktree")
    parser.add_argument("--group", action="append", help="Record only these groups (default: all)")
    parser.add_argument("--golden-root", type=Path, default=os.environ.get("FOURIER_GOLDEN_ROOT"),
                        help="Original checkout with real checkpoints/datasets for local-only probes (read-only)")
    parser.add_argument("--allow-any-tree", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        parser.error("Set CUDA_VISIBLE_DEVICES= (empty) so recording never touches a GPU")

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(root))
    import fourier_score

    if Path(fourier_score.__file__).resolve().parent != root / "fourier_score":
        parser.error(f"fourier_score imported from {fourier_score.__file__}, not {root}")
    import golden_probes as gp
    from fourier_score.utils import source_hash

    if not args.allow_any_tree:
        tag = git(root, "rev-parse", gp.EPOCH0_TAG + "^{commit}")
        if git(root, "rev-parse", "HEAD") != tag:
            parser.error(f"{root} is not at {gp.EPOCH0_TAG}")
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "fourier_score"], cwd=root).returncode:
            parser.error(f"{root}/fourier_score has uncommitted changes")
        if source_hash() != gp.EPOCH0_SOURCE_SHA256:
            parser.error("source_hash() differs from the epoch-0 hash")

    groups = args.group or list(gp.GROUPS)
    golden_root = args.golden_root.resolve() if args.golden_root else None
    with record_tmp() as tmp:
        ctx = gp.Context(root=root, tmp=tmp, golden_root=golden_root, recording=True)
        for group in groups:
            keys = gp.load_group(group)
            gp.PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
            gp.write_payloads(group, ctx)
            path = gp.golden_path(group)
            previous = json.loads(path.read_text()) if path.exists() else {"probes": {}}
            probes = dict(previous["probes"])
            for key in keys:
                item = gp.PROBES[key]
                if item.local and golden_root is None:
                    print(f"skip {key} (local-only; pass --golden-root)", flush=True)
                    continue
                print(f"record {key}", flush=True)
                probes[key] = gp.run(key, ctx)
            stale = sorted(set(probes) - set(keys))
            for key in stale:
                del probes[key]
            payload = {
                "_meta": {
                    "tag": gp.EPOCH0_TAG,
                    "source_sha256": gp.EPOCH0_SOURCE_SHA256,
                    "machine": gp.machine(),
                },
                "probes": dict(sorted(probes.items())),
            }
            path.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=True) + "\n")
            print(f"wrote {path} ({len(probes)} probes, removed {len(stale)} stale)", flush=True)


if __name__ == "__main__":
    main()
