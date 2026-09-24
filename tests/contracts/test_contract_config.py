"""Golden contract: config. Values recorded from the epoch-0 tree."""

import pytest

import golden_probes as gp

KEYS = gp.load_group("config")


@pytest.mark.parametrize("key", KEYS)
def test_golden(key, golden_ctx):
    if gp.PROBES[key].local and golden_ctx.golden_root is None:
        pytest.skip("local-only golden; set FOURIER_GOLDEN_ROOT to the original checkout")
    gp.check("config", key, golden_ctx)
