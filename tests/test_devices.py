import pytest
import torch
from scripts.doctor import run

@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='No MPS hardware in this test environment')
@pytest.mark.parametrize('backend',['auto','cpu'])
def test_mps(backend):
    assert run('mps',backend)['passed']

@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='No CUDA hardware in this test environment')
def test_cuda():
    assert run('cuda')['passed']
