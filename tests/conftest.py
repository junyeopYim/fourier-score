import pytest
import torch
from fourier_score.config import load_config
from fourier_score.data_loader.statistics import placeholder_stats


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)

@pytest.fixture
def cfg(tmp_path):
    return load_config('configs/smoke.json',[f'trainer.save_dir={tmp_path}/runs',f'fourier.cache_dir={tmp_path}/stats','backend.cpu_threads=2','device=cpu'])

@pytest.fixture
def stats():
    return placeholder_stats((1, 8, 8))
