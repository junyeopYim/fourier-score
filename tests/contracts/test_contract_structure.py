"""Structural contracts (no goldens): vendored bytes, source_hash coverage,
package layering and documented entrypoints.

These facts must hold on the epoch-0 tree (tag ``pre-template-refactor``) and
after every refactor step.  Paths are anchored on this file, so the module can
be copied into another checkout and run there unchanged.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

import golden_probes as gp

ROOT = Path(__file__).resolve().parents[2]


def subprocess_env(**extra):
    """CPU-only child environment that never writes bytecode into the tree."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.update(extra)
    return env


def git_ls_files(*pathspecs, root=ROOT):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *pathspecs], cwd=root, capture_output=True
    )
    if result.returncode:
        pytest.skip(f"{root} is not a git checkout: {result.stderr.decode().strip()}")
    return sorted(name for name in result.stdout.decode().split("\0") if name)


def python_files(directory):
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


# ------------------------------------------------------------ 1. vendored bytes


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# `git ls-tree -r pre-template-refactor fourier_score/backbones fourier_score/ldm/upstream`
BACKBONE_BLOBS = {
    "layers.py": "eb772b2e6606ed92295dd031cb43be8a82a992c7",
    "layerspp.py": "2eb4e5de372e799f0608272408718664c35719c0",
    "ncsnpp.py": "ea16eb35b5e6f2fa92db10be2552b3acdfe8fa7b",
    "up_or_down_sampling.py": "bf73196b571b2bdfcde5077a3978fcebcc99a539",
}
UPSTREAM_BLOBS = {
    "LICENSE": "be24ebeed0fb31665dc3c33d2610d35f77b12709",
    "LICENSE.taming": "57fb4153bafcd64b60377ba0ba2c79b7530efc1e",
    "PROVENANCE.json": "f13ac386a1d4fdea63381566fa1600287949d9fb",
    "__init__.py": "9f11b40c764c89a4ece4bd11979dc8e041218ffd",
    "attention.py": "f6a666bc2ee900e6839542cf3ce3c65e8b5d3d4d",
    "ema.py": "c8c75af43565f6e140287644aaaefa97dd6e67c5",
    "factory.py": "2642e5ccc11545a748b7b29a1d1103bc06450c41",
    "model.py": "1b6a416c5e0eaa94a64c509ec490c3cf0a074f72",
    "openaimodel.py": "31bbfcc1d40a7657cbba8a747674397d93fb338c",
    "util.py": "893f41ae66e26194b8d950dac2f93d0fa48c4bbd",
}
# The epoch-0 location comes first.  Another location is accepted only when
# the epoch-0 directory is gone, and it must be unique.
VENDORED = {
    "backbones": (
        ("fourier_score/backbones", "fourier_score/model/backbones", "fourier_score/third_party/backbones"),
        BACKBONE_BLOBS,
    ),
    "ldm_upstream": (
        (
            "fourier_score/ldm/upstream",
            "fourier_score/third_party/ldm_upstream",
            "fourier_score/third_party/ldm/upstream",
            "fourier_score/third_party/latent_diffusion",
        ),
        UPSTREAM_BLOBS,
    ),
}
# Third-party packages the vendored code imports at the tag (stdlib is always allowed).
VENDORED_THIRD_PARTY = {"torch", "numpy", "einops", "omegaconf"}


def vendored_dir(name, root=ROOT):
    """Directory of a vendored tree; a future ``third_party`` home only if the old one is absent."""
    candidates, pinned = VENDORED[name]
    current = root / candidates[0]
    if current.is_dir():
        return current
    found = {root / c for c in candidates[1:] if (root / c).is_dir()}
    third_party = root / "fourier_score" / "third_party"
    if third_party.is_dir():
        found |= {
            d for d in third_party.rglob("*")
            if d.is_dir() and all((d / f).is_file() for f in pinned)
        }
    assert len(found) == 1, f"{name}: {candidates[0]} is gone and candidates are {sorted(found)}"
    return found.pop()


def test_vendored_locator_prefers_epoch0_location(tmp_path):
    for name, (candidates, pinned) in VENDORED.items():
        for location in (candidates[0], "fourier_score/third_party/x/" + name):
            (tmp_path / location).mkdir(parents=True, exist_ok=True)
            for filename in pinned:
                (tmp_path / location / filename).touch()
        assert vendored_dir(name, tmp_path) == tmp_path / candidates[0]
        shutil.rmtree(tmp_path / candidates[0])
        assert vendored_dir(name, tmp_path) == tmp_path / "fourier_score/third_party/x" / name


@pytest.mark.parametrize("name", sorted(BACKBONE_BLOBS))
def test_backbone_git_blob_identity(name):
    data = (vendored_dir("backbones") / name).read_bytes()
    assert git_blob_sha1(data) == BACKBONE_BLOBS[name], f"vendored backbones/{name} changed"


def test_ldm_upstream_is_byte_identical():
    upstream = vendored_dir("ldm_upstream")
    present = sorted(
        str(p.relative_to(upstream)) for p in upstream.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    assert present == sorted(UPSTREAM_BLOBS)
    for name, blob in UPSTREAM_BLOBS.items():
        assert git_blob_sha1((upstream / name).read_bytes()) == blob, f"ldm/upstream/{name} changed"
    manifest = json.loads((upstream / "PROVENANCE.json").read_text())
    assert manifest["files"], "PROVENANCE.json lists no files"
    for name, item in manifest["files"].items():
        assert gp.file_sha256(upstream / name) == item["local_sha256"], name
    for name, item in manifest["configs"].items():
        assert gp.file_sha256(ROOT / name) == item["sha256"], name


# ------------------------------------------------------- 2. source_hash coverage


def provenance():
    modules = ("fourier_score.provenance", "fourier_score.utils")
    return gp.symbol("source_hash", *modules), Path(gp.symbol("ROOT", *modules))


def call(source_hash):
    getattr(source_hash, "cache_clear", lambda: None)()
    return source_hash()


def enumerated(root):
    """``source_hash``'s enumeration at the tag: every ``*.py`` below ``fourier_score/``."""
    return sorted((root / "fourier_score").rglob("*.py"))


def reference_digest(root, relative_paths):
    """The tag's algorithm: sorted paths, each contributing relpath then bytes."""
    h = hashlib.sha256()
    for path in sorted(root / name for name in relative_paths):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def test_source_hash_covers_exactly_the_tracked_package():
    source_hash, root = provenance()
    assert root.resolve() == ROOT, f"ROOT={root} but tests run in {ROOT}"
    assert (root / "pyproject.toml").is_file() and (root / "fourier_score/__init__.py").is_file()

    hashed = [str(p.relative_to(root)) for p in enumerated(root)]
    tracked = git_ls_files("fourier_score/**/*.py", "fourier_score/*.py", root=root)
    assert sorted(hashed) == tracked, (
        f"untracked={sorted(set(hashed) - set(tracked))} "
        f"unhashed={sorted(set(tracked) - set(hashed))}"
    )
    assert len(tracked) >= 38
    vendored = {
        str(p.relative_to(root))
        for name in VENDORED
        for p in python_files(vendored_dir(name, root))
    }
    assert {str((vendored_dir("backbones", root) / n).relative_to(root)) for n in BACKBONE_BLOBS} <= vendored
    assert vendored <= set(tracked), f"vendored files not hashed: {sorted(vendored - set(tracked))}"

    digest = call(source_hash)
    assert digest != hashlib.sha256(b"").hexdigest()
    assert digest == reference_digest(root, tracked)


def flip_one_byte(path):
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def mutate(package, how):
    """Apply one change to a copied ``fourier_score/`` (targets chosen by structure, not name)."""
    files = python_files(package)
    deepest = max(files, key=lambda p: (len(p.relative_to(package).parts), str(p)))
    if how == "edit":
        # A first-party module at the package top level (vendored trees are nested).
        flip_one_byte(next(
            p for p in files if p.parent == package and p.name != "__init__.py" and p.stat().st_size
        ))
    elif how == "edit_vendored":
        flip_one_byte(vendored_dir("backbones", package.parent) / "layers.py")
    elif how == "add":
        (deepest.parent / "contract_added_module.py").write_bytes(b"VALUE = 1\n")
    elif how == "rename":
        deepest.rename(deepest.with_name(deepest.stem + "_renamed.py"))
    else:
        raise ValueError(how)


@pytest.mark.parametrize("how", ["edit", "edit_vendored", "add", "rename"])
def test_source_hash_detects_changes_in_a_copy(how, tmp_path, monkeypatch):
    source_hash, root = provenance()
    original = call(source_hash)
    copy = tmp_path / "checkout"
    shutil.copytree(
        root / "fourier_score", copy / "fourier_score", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(root / "pyproject.toml", copy / "pyproject.toml")
    monkeypatch.setattr(sys.modules[source_hash.__module__], "ROOT", copy)
    # Relative paths only: the checkout location does not enter the digest.
    assert call(source_hash) == original
    mutate(copy / "fourier_score", how)
    assert call(source_hash) != original, f"source_hash ignored a {how}"


# ------------------------------------------------------------------ 3. layering


def imports(path):
    """``(module, level, lineno)`` for static imports plus string-literal dynamic imports."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, 0, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                found += [(alias.name, node.level, node.lineno) for alias in node.names]
            else:
                found.append((node.module, node.level, node.lineno))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.append((arg.value, 0, node.lineno))
    return found


def test_package_import_is_torch_free():
    code = (
        "import fourier_score, fourier_score.ldm, fourier_score.ldm.dataset_sources; import sys; "
        "assert 'torch' not in sys.modules, sorted(sys.modules); print(fourier_score.__file__)"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=ROOT,
        env=subprocess_env(PYTHONPATH=str(ROOT)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == ROOT / "fourier_score/__init__.py"


def test_package_never_imports_orchestration_code():
    forbidden = {"scripts", "experiments", "tests"}
    bad = [
        f"{path.relative_to(ROOT)}:{line} imports {module}"
        for path in python_files(ROOT / "fourier_score")
        for module, level, line in imports(path)
        if level == 0 and module.split(".")[0] in forbidden
    ]
    assert not bad, bad


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_vendored_imports_only_itself_stdlib_and_whitelist(name):
    directory = vendored_dir(name)
    allowed = set(sys.stdlib_module_names) | VENDORED_THIRD_PARTY
    bad = []
    for path in python_files(directory):
        for module, level, line in imports(path):
            where = f"{path.relative_to(ROOT)}:{line}"
            if level == 0:
                if module.split(".")[0] not in allowed:
                    bad.append(f"{where} imports {module}")
            elif level > 1:
                bad.append(f"{where} reaches outside the vendored tree ({'.' * level}{module})")
            else:
                head = module.split(".")[0]
                if not ((directory / f"{head}.py").is_file() or (directory / head).is_dir()):
                    bad.append(f"{where} imports .{module}, which is not in the vendored tree")
    assert not bad, bad


def test_tensorflow_evaluator_is_independent_of_the_package():
    path = ROOT / "scripts/evaluate_score_sde.py"
    tree = ast.parse(path.read_bytes())
    assert not [m for m, _, _ in imports(path) if m.split(".")[0] == "fourier_score"]
    mentions = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "fourier_score" in node.value
    ]
    assert not mentions, f"evaluate_score_sde.py mentions fourier_score on lines {mentions}"


TEMPLATE_NAMES = ("model", "utils", "base", "trainer", "logger", "data_loader")


def test_repo_root_has_no_template_package_directories():
    assert not [n for n in TEMPLATE_NAMES if (ROOT / n).is_dir()]
    assert not sorted({p.split("/")[0] for p in git_ls_files() if "/" in p} & set(TEMPLATE_NAMES))


def test_tracked_tree_avoids_gitignored_directory_names():
    bad = [
        path for path in git_ls_files()
        if {"pretrained", "local", "notes"} & set(path.split("/")[:-1])
    ]
    assert not bad, bad


# --------------------------------------------------------------- 4. entrypoints

DOCS = ("README.md", ".github/workflows/checks.yml", "scripts/reproduce_cifar10.sh", "notebooks/README.md")
URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s)\]>\"'`]+")
PY_REFERENCE = re.compile(r"[\w./-]*\w\.py(?!\w)")


def documents():
    extra = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "reports").rglob("*.md"))
    return [*DOCS, *extra]


def references(doc):
    return sorted(set(PY_REFERENCE.findall(URL.sub(" ", (ROOT / doc).read_text()))))


def resolves(reference, doc):
    """Relative to the document, the repository root, or (for prose) the package."""
    bases = ((ROOT / doc).parent, ROOT, ROOT / "fourier_score")
    return any((base / reference).is_file() for base in bases)


def test_reference_scanner_ignores_urls_and_other_suffixes():
    text = "run `scripts/doctor.py`, see https://github.com/x/y/blob/main/z.py and a.pyc, ../b/c.py."
    assert PY_REFERENCE.findall(URL.sub(" ", text)) == ["scripts/doctor.py", "../b/c.py"]


@pytest.mark.parametrize("doc", documents())
def test_documented_python_paths_exist(doc):
    assert (ROOT / doc).is_file(), f"{doc} is missing"
    found = references(doc)
    if doc == "README.md":
        assert {"train.py", "sample.py", "evaluate.py", "ldm.py"} <= set(found)
    missing = [ref for ref in found if not resolves(ref, doc)]
    assert not missing, f"{doc} references missing files: {missing}"


ROOT_ENTRYPOINTS = ("train.py", "sample.py", "evaluate.py", "ldm.py")
# CLIs at the tag; they keep their paths through the refactor.
EPOCH0_SCRIPT_CLIS = (
    "benchmark_ldm", "check_notebooks", "doctor", "download_ldm", "download_ldm_data",
    "download_score_sde", "export_real", "import_score_sde", "inspect_model",
    "plot_diagnostics", "plot_log_gate_design", "plot_method", "prepare", "run_comparison",
    "run_gmm_comparison", "run_gmm_gate_shapes", "run_gmm_log_gates", "run_gmm_plateau",
    "run_gmm_spectral_gate", "run_mnist_remaining", "verify_ldm_upstream",
)
HELP_SKIPS = {
    # Targets a separate TensorFlow environment (scripts/requirements-score-sde-eval.txt),
    # not the locked project environment; its independence is checked above instead.
    "scripts/evaluate_score_sde.py",
}


def script_clis():
    """``scripts/*.py`` files that define a command line (library modules are ignored)."""
    return sorted(
        f"scripts/{p.name}" for p in (ROOT / "scripts").glob("*.py")
        if "ArgumentParser" in p.read_text() and f"scripts/{p.name}" not in HELP_SKIPS
    )


ENTRYPOINTS = [*ROOT_ENTRYPOINTS, *script_clis()]


def test_every_epoch0_cli_is_still_swept():
    assert {f"scripts/{name}.py" for name in EPOCH0_SCRIPT_CLIS} <= set(ENTRYPOINTS)


def run_help(entrypoint):
    return subprocess.run(
        [sys.executable, str(ROOT / entrypoint), "--help"],
        cwd=ROOT,
        env=subprocess_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def help_results():
    # Mostly `import torch`; run concurrently so the sweep stays a few seconds of wall time.
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as pool:
        return dict(zip(ENTRYPOINTS, pool.map(run_help, ENTRYPOINTS)))


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_cli_help_exits_zero(entrypoint, help_results):
    result = help_results[entrypoint]
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout.lower(), result.stdout[:500]
