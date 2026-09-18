import torch
import pytest
from parse_config import ConfigParser


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def smoke_config(tmp_path):
    return ConfigParser.from_file('configs/smoke.json', [
        f'trainer.save_dir={tmp_path / "runs"}',
        f'data_loader.cache_dir={tmp_path / "cache"}']).config
