"""Record the GMM smoke-chain goldens (tests/golden/gmm_smoke/) from the epoch-0 runners.

Run inside the epoch-0 worktree with its interpreter:

    cd ../fs-epoch0
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \\
        .venv/bin/python ../fs-refactor/tests/golden/record_gmm_smoke.py --root .

The five legacy runners ``--root/scripts/run_gmm_*.py`` run as the archived
study's reuse chain (``golden_probes.gmm_smoke_norm.CHAIN``) with
``--preset smoke --seeds 42 --workers 1`` in scratch directories under /tmp;
``--root`` is only read (``log_gates`` reads its
``assets/log_gate_design/design.json``).  Then:

1. The chain runs twice in independent scratch directories and both
   normalized outputs must be identical (determinism self-check).
2. The first link runs once more with ``--workers 2`` and must normalize
   identically (the new ``python -m experiments`` CLI is checked that way).
3. ``figures.json`` (normalized SVG digests) is written only when the two
   chains agree on it; otherwise only the figure file names (``files.json``)
   are recorded.
4. ``tests/golden/gmm_smoke/`` next to this script is rewritten (nothing
   else is touched).  Two recordings on one machine are byte-identical: no
   timestamps, scratch paths or durations are written.
"""

import os
import sys

# Before anything is imported from --root: never write __pycache__ there.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

HERE = Path(__file__).resolve().parent
FROZEN_INPUTS = ("fourier_score", "scripts", "assets/log_gate_design")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def first_difference(a, b, path="$"):
    """Path of the first difference between two JSON values (for error messages)."""
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if sorted(a) != sorted(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            found = first_difference(a[k], b[k], f"{path}.{k}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b or (a != a and b != b) else f"{path}: {a!r} != {b!r}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="epoch-0 worktree (default: cwd)")
    parser.add_argument("--keep", action="store_true", help="Keep the scratch directories")
    parser.add_argument("--allow-any-tree", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        parser.error("Set CUDA_VISIBLE_DEVICES= (empty) so recording never touches a GPU")

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(root))
    import golden_probes as gp
    from golden_probes import gmm_smoke_norm as norm

    if not args.allow_any_tree:
        import fourier_score
        from fourier_score.utils import source_hash

        if Path(fourier_score.__file__).resolve().parent != root / "fourier_score":
            parser.error(f"fourier_score imported from {fourier_score.__file__}, not {root}")
        if git(root, "rev-parse", "HEAD") != git(root, "rev-parse", gp.EPOCH0_TAG + "^{commit}"):
            parser.error(f"{root} is not at {gp.EPOCH0_TAG}")
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *FROZEN_INPUTS], cwd=root).returncode:
            parser.error(f"{root} has uncommitted changes in {FROZEN_INPUTS}")
        if source_hash() != gp.EPOCH0_SOURCE_SHA256:
            parser.error("source_hash() differs from the epoch-0 hash")

    python = sys.executable
    scratch = Path(tempfile.mkdtemp(prefix="fourier-gmm-smoke.", dir="/tmp")).resolve()
    try:
        chains = []
        for attempt in ("a", "b"):
            base = scratch / attempt
            seconds = norm.run_chain(root, base, python=python)
            print(f"chain {attempt}: " + ", ".join(f"{k} {v:.1f}s" for k, v in seconds.items()), flush=True)
            chains.append((base, norm.normalize_chain(base, root)))
        (base_a, first), (_, second) = chains
        figures_deterministic = all(first[n][norm.FIGURES_FILE] == second[n][norm.FIGURES_FILE] for n in first)
        for name in first:
            for file in norm.GOLDEN_FILES:
                found = first_difference(first[name][file], second[name][file])
                if found:
                    raise SystemExit(f"Nondeterministic smoke output {name}/{file}: {found}")

        gated = norm.CHAIN[0]
        base_w = scratch / "workers2"
        norm.run(norm.stub_command(root, gated, base_w, python=python, workers=2), root, scratch / "workers2.log")
        parallel = norm.normalize(base_w / gated.output, base_w, root)
        for file in norm.GOLDEN_FILES:
            found = first_difference(first[gated.name][file], parallel[file])
            if found:
                raise SystemExit(f"--workers 2 changes {gated.name}/{file}: {found}")
        print(f"{gated.name}: --workers 2 output identical", flush=True)

        if norm.GOLDEN.exists():
            shutil.rmtree(norm.GOLDEN)
        norm.GOLDEN.mkdir(parents=True)
        for name, sections in first.items():
            folder = norm.GOLDEN / name
            folder.mkdir()
            for file, section in sections.items():
                if file == norm.FIGURES_FILE and not figures_deterministic:
                    continue
                (folder / file).write_text(norm.dumps(section))
        meta = {
            "tag": gp.EPOCH0_TAG,
            "source_sha256": gp.EPOCH0_SOURCE_SHA256,
            "machine": gp.machine(),
            "chain": [
                {"name": link.name, "output": link.output, "reuse": list(link.reuse),
                 "command": norm.placeholder_command(norm.stub_command(root, link, base_a, python=python),
                                                     base_a, root, python)}
                for link in norm.CHAIN
            ],
            "figures_deterministic": figures_deterministic,
            "figures": ("sha256 of normalize_svg(svg) per figure (raw SVG/PDF embed a date and salted ids)"
                        if figures_deterministic else "figure file names only (files.json)"),
            "workers_invariant": {gated.name: 2},
        }
        (norm.GOLDEN / "_meta.json").write_text(norm.dumps(meta))
        count = sum(1 for _ in norm.GOLDEN.rglob("*.json"))
        size = sum(p.stat().st_size for p in norm.GOLDEN.rglob("*.json"))
        print(f"wrote {norm.GOLDEN} ({count} files, {size / 1e6:.2f} MB, "
              f"figures_deterministic={figures_deterministic})", flush=True)
    finally:
        if args.keep:
            print(f"kept {scratch}", flush=True)
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
