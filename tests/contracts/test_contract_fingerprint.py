"""Contract: every tracked Python file a GMM run executes is fingerprinted.

A tiny ``scripts/run_gmm_comparison.py`` smoke run (one gate switch, one
seed, in-process) is executed through ``runpy`` in a subprocess that then
dumps ``sys.modules`` file paths (no sitecustomize or import hooks).  Every
loaded file that ``git ls-files '*.py'`` tracks must be covered by the run's
``protocol.json`` provenance:

* ``fourier_score/**`` by ``source_sha256`` (checked against this tree), or
* every other file (the stub itself, ``experiments/**``) by
  ``orchestration_sha256`` ``{relpath: sha256}`` (checked against the files).

Until the stub delegates to ``experiments`` the runner is the epoch-0 script,
whose provenance names its helper scripts by key (``runner_sha256`` etc.);
those keys are mapped to their files so the same coverage rule is checked
(``LEGACY_KEYS``; dead once every stub writes ``orchestration_sha256``).
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import golden_probes as gp
from golden_probes import gmm_smoke_norm as norm

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/run_gmm_comparison.py"
# Epoch-0 runner provenance keys -> the file each one digests.
LEGACY_KEYS = {
    "reporting_helpers_sha256": "scripts/run_gmm_comparison.py",
    "comparison_helpers_sha256": "scripts/run_gmm_comparison.py",
    "experiment_helpers_sha256": "scripts/run_gmm_plateau.py",
    "gate_shape_helpers_sha256": "scripts/run_gmm_gate_shapes.py",
}
COLLECT = r"""
import json, runpy, sys
out, script, *argv = sys.argv[1:]
sys.argv = [script, *argv]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    files = {getattr(module, "__file__", None) for module in list(sys.modules.values())}
    with open(out, "w") as stream:
        json.dump(sorted(f for f in files | {script} if f), stream)
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprinted(provenance, script):
    """``{relpath: sha256}`` of the non-``fourier_score`` files a protocol digests."""
    if "orchestration_sha256" in provenance:
        return dict(provenance["orchestration_sha256"])
    files = {script: provenance["runner_sha256"]}
    for key, path in LEGACY_KEYS.items():
        if key in provenance:
            files[path] = provenance[key]
    return files


def test_loaded_sources_are_fingerprinted(tmp_path):
    output, listing = tmp_path / "gmm_fingerprint", tmp_path / "modules.json"
    command = [sys.executable, "-c", COLLECT, str(listing), str(REPO_ROOT / SCRIPT),
               "--output", str(output), *norm.SMOKE, "--workers", "1", "--switches", "1.5"]
    done = subprocess.run(command, cwd=REPO_ROOT, env=norm.subprocess_env(), capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]

    tracked = set(subprocess.check_output(["git", "ls-files", "*.py"], cwd=REPO_ROOT, text=True).split())
    loaded = set()
    for name in json.loads(listing.read_text()):
        path = (REPO_ROOT / name).resolve()
        if path.is_relative_to(REPO_ROOT) and path.relative_to(REPO_ROOT).as_posix() in tracked:
            loaded.add(path.relative_to(REPO_ROOT).as_posix())
    assert SCRIPT in loaded and any(p.startswith("fourier_score/") for p in loaded), sorted(loaded)

    provenance = json.loads((output / "protocol.json").read_text())["provenance"]
    source_hash = gp.symbol("source_hash", "fourier_score.provenance", "fourier_score.utils")
    assert provenance["source_sha256"] == source_hash()
    files = fingerprinted(provenance, SCRIPT)
    for path, digest in sorted(files.items()):
        assert digest == sha256(REPO_ROOT / path), f"stale fingerprint for {path}"
    uncovered = sorted(p for p in loaded if not p.startswith("fourier_score/") and p not in files)
    assert not uncovered, f"executed but not fingerprinted: {uncovered}"
    if any(p.startswith("experiments/") for p in loaded):
        assert "orchestration_sha256" in provenance, "experiments code ran without orchestration_sha256"
