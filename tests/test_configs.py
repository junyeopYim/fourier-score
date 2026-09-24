"""Every config under configs/ loads, validates and names its run.

Presets are added and edited freely during research; nothing here records
their contents, so a new or changed preset only has to stay loadable.
"""

from pathlib import Path

import pytest

from fourier_score.config import experiment_name, load_config, validate
from fourier_score.trainer.checkpoints import resume_signature

ROOT = Path(__file__).resolve().parents[1]
LDM = ROOT / "configs" / "ldm"
IMAGE = sorted(
    [ROOT / "config.json", *(p for p in (ROOT / "configs").rglob("*.json") if LDM not in p.parents)]
)


def relative(path):
    return path.relative_to(ROOT).as_posix()


@pytest.mark.parametrize("path", IMAGE, ids=relative)
def test_image_config_loads(path):
    cfg = load_config(str(path))
    assert validate(cfg) == cfg
    assert experiment_name(cfg)
    resume_signature(cfg)


@pytest.mark.parametrize("path", sorted(LDM.glob("*.json")), ids=relative)
def test_ldm_config_loads(path):
    pytest.importorskip("yaml")
    from fourier_score.ldm import config as ldm

    cfg = ldm.load_config(str(path))
    assert ldm.validate(cfg) == cfg
    assert ldm.experiment_name(cfg)
    ldm.load_spec(cfg)
