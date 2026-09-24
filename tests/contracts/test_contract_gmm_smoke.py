"""Golden contract: the GMM smoke reuse chain (A3 orchestration move).

Runs the five ``scripts/run_gmm_*.py`` entry points of this tree as the
archived study's reuse chain (``golden_probes.gmm_smoke_norm.CHAIN``:
gated -> plateau -> spectral -> gate_shapes -> log_gates, each reusing the
controls of every earlier output) with ``--preset smoke --seeds 42
--workers 1`` and compares the normalized outputs with
``tests/golden/gmm_smoke/``, recorded from the epoch-0 runners by
``tests/golden/record_gmm_smoke.py``.  ``exact`` sections must match
everywhere; ``numeric`` sections bit-for-bit on the recording machine and
within ``rtol=1e-4`` elsewhere (digests skipped), per
``golden_probes.exact_numeric``.

Then the first link runs once more through ``python -m experiments gmm
gated`` (``--workers 2``; the recorder proved the epoch-0 output does not
depend on the worker count) and must normalize identically to the stub run.
That part is skipped until the ``experiments`` package exists.

Runtime: about 25-30 s for the chain on the recording machine (single
CPU thread per run), so it is not marked ``slow``.
"""

from pathlib import Path
import time

import pytest

import golden_probes as gp
from golden_probes import gmm_smoke_norm as norm

REPO_ROOT = Path(__file__).resolve().parents[2]
NAMES = [link.name for link in norm.CHAIN]


@pytest.fixture(scope="module")
def meta():
    return norm.load_meta()


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    base = tmp_path_factory.mktemp("gmm_smoke").resolve()
    start = time.perf_counter()
    seconds = norm.run_chain(REPO_ROOT, base)
    total = time.perf_counter() - start
    print(f"\nGMM smoke chain: {total:.1f}s (" + ", ".join(f"{k} {v:.1f}s" for k, v in seconds.items()) + ")")
    return base, norm.normalize_chain(base, REPO_ROOT)


def compare_sections(expected, actual, *, exact_numeric, label):
    """Compare golden files ``{file: {"exact"|"numeric": value}}`` of one link."""
    wanted = set(expected)
    produced = {f for f in actual if f != norm.FIGURES_FILE or norm.FIGURES_FILE in expected}
    assert wanted == produced, f"{label}: golden files {sorted(wanted)} != {sorted(produced)}"
    for file, section in expected.items():
        (kind, value), = section.items()
        assert kind in actual[file], f"{label}/{file}: section {kind} missing"
        gp.compare(value, actual[file][kind], exact=kind == "exact" or exact_numeric, path=f"{label}/{file}")


def test_recorded_chain_is_the_current_chain(meta, tmp_path):
    """The recorder and this test build the same commands (else re-record)."""
    assert meta["tag"] == gp.EPOCH0_TAG and meta["source_sha256"] == gp.EPOCH0_SOURCE_SHA256
    assert [c["name"] for c in meta["chain"]] == NAMES
    for recorded, link in zip(meta["chain"], norm.CHAIN):
        assert (recorded["output"], tuple(recorded["reuse"])) == (link.output, link.reuse)
        command = norm.placeholder_command(norm.stub_command(REPO_ROOT, link, tmp_path), tmp_path, REPO_ROOT)
        assert recorded["command"] == command
    assert sorted(p.name for p in norm.GOLDEN.iterdir() if p.is_dir()) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_stub_chain_matches_epoch0(chain, meta, name):
    _, actual = chain
    compare_sections(norm.load_golden(name), actual[name],
                     exact_numeric=gp.exact_numeric(meta["machine"]), label=name)


def test_experiments_cli_matches_stub(chain, tmp_path):
    if not (REPO_ROOT / "experiments" / "__main__.py").is_file():
        pytest.skip("experiments package does not exist yet (A3); only the stub chain is checked")
    base, actual = chain
    link = norm.CHAIN[0]
    cli_base = tmp_path.resolve()
    start = time.perf_counter()
    norm.run(norm.experiments_command(link, cli_base, workers=2), REPO_ROOT, cli_base / "cli.log")
    print(f"\npython -m experiments gmm {link.name}: {time.perf_counter() - start:.1f}s")
    ours = norm.normalize(cli_base / link.output, cli_base, REPO_ROOT)
    compare_sections(actual[link.name], ours, exact_numeric=True, label=f"experiments:{link.name}")
