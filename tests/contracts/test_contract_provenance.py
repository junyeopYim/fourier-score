"""Contract: ``fourier_score.provenance`` is a stdlib-only, repository-anchored fingerprint.

``ROOT`` is the first directory above the module holding ``pyproject.toml``
and the ``fourier_score`` package (with ``__init__.py``) the module belongs to;
another tree's package above it is never used.  ``source_hash`` keeps
the epoch-0 algorithm (sorted relpaths, each followed by the file bytes) but
refuses to fingerprint a tree without the package.
"""

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

import golden_probes as gp
from fourier_score import provenance

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_root_is_the_repository_anchor():
    from fourier_score import config, utils

    assert provenance.ROOT == REPO_ROOT
    assert (provenance.ROOT / "pyproject.toml").is_file()
    assert (provenance.ROOT / "fourier_score/__init__.py").is_file()
    assert utils.ROOT is provenance.ROOT and config.ROOT is provenance.ROOT
    assert utils.source_hash is provenance.source_hash


def load_copy(path):
    spec = importlib.util.spec_from_file_location("contract_provenance_copy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_walks_up_to_the_nearest_complete_anchor(tmp_path):
    outer, inner = tmp_path / "outer", tmp_path / "outer/inner"
    module = inner / "fourier_score/deep/provenance.py"
    module.parent.mkdir(parents=True)
    shutil.copy2(provenance.__file__, module)
    with pytest.raises(RuntimeError, match="pyproject.toml"):
        load_copy(module)
    (inner / "pyproject.toml").write_text("")
    with pytest.raises(RuntimeError, match="pyproject.toml"):
        load_copy(module)
    (outer / "pyproject.toml").write_text("")
    (outer / "fourier_score").mkdir()
    (outer / "fourier_score/__init__.py").write_text("")
    with pytest.raises(RuntimeError, match="pyproject.toml"):
        load_copy(module)
    (inner / "fourier_score/__init__.py").write_text("")
    assert load_copy(module).ROOT == inner.resolve()


@pytest.mark.parametrize("tree", ["empty", "no_sources", "no_init"])
def test_source_hash_refuses_a_tree_without_the_package(tree, tmp_path, monkeypatch):
    if tree != "empty":
        (tmp_path / "fourier_score/sub").mkdir(parents=True)
        (tmp_path / "fourier_score/notes.txt").write_text("not python\n")
    if tree == "no_init":
        (tmp_path / "fourier_score/sub/module.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    with pytest.raises(RuntimeError):
        provenance.source_hash()


def test_source_hash_digests_sorted_relpaths_then_bytes(tmp_path, monkeypatch):
    init, module = b'"""Package."""\n', b"VALUE = 1\n"
    (tmp_path / "fourier_score/sub").mkdir(parents=True)
    (tmp_path / "fourier_score/sub/a.py").write_bytes(module)
    (tmp_path / "fourier_score/__init__.py").write_bytes(init)
    (tmp_path / "fourier_score/sub/data.json").write_text("{}")
    expected = hashlib.sha256(
        b"fourier_score/__init__.py" + init + b"fourier_score/sub/a.py" + module
    ).hexdigest()
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    assert provenance.source_hash() == expected


def test_source_hash_reproduces_the_epoch0_fingerprint(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    archive = tmp_path / "epoch0.tar"
    result = subprocess.run(
        ["git", "archive", "-o", str(archive), gp.EPOCH0_TAG, "fourier_score"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        pytest.skip(f"tag {gp.EPOCH0_TAG} is unavailable: {result.stderr.strip()}")
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "epoch0", filter="data")
    monkeypatch.setattr(provenance, "ROOT", tmp_path / "epoch0")
    assert provenance.source_hash() == gp.EPOCH0_SOURCE_SHA256


def test_import_is_stdlib_only():
    code = (
        "import fourier_score.provenance, sys; "
        "assert 'torch' not in sys.modules, sorted(sys.modules); "
        "print(fourier_score.provenance.__file__)"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == REPO_ROOT / "fourier_score/provenance.py"


@pytest.mark.parametrize("size", [0, 8 * 1024 * 1024 + 3])
def test_file_sha256_streams_whole_files(size, tmp_path):
    data = (bytes(range(256)) * (size // 256 + 1))[:size]
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert provenance.file_sha256(path) == hashlib.sha256(data).hexdigest()
