import copy
import pytest
import torch
from model.model import build_model
from model.loss import build_loss
from model.gaussian import estimate_stats
from sde.sde_lib import make_sde
from sde.sampling import sample_batch, predictor_update
from model.backbones.up_or_down_sampling import upsample_2d, downsample_2d, upsample_conv_2d
from model.backbones.layerspp import Upsample
from utils.util import isolated_rng
from parse_config import ConfigParser
from sample import BatchedScore


@pytest.mark.parametrize('progressive', ['none', 'output_skip', 'residual'])
@pytest.mark.parametrize('input_mode', ['none', 'input_skip', 'residual'])
def test_ncsnpp_architecture_paths(smoke_config, progressive, input_mode):
    cfg = smoke_config
    cfg.data.image_size = 8
    cfg.model.attn_resolutions = [4]
    cfg.model.progressive, cfg.model.progressive_input = progressive, input_mode
    x = torch.randn(2, 1, 8, 8)
    stats = estimate_stats([torch.randn(8, 1, 8, 8)])
    model = build_model(cfg, stats)
    loss = build_loss(cfg, make_sde(cfg))(model, x)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('discrete', [True, False])
def test_ve_discrete_and_continuous(smoke_config, discrete):
    cfg = smoke_config
    if discrete:
        cfg.training.continuous = False
        cfg.model.embedding_type = 'positional'
    stats = estimate_stats([torch.randn(4, 1, 32, 32)])
    model = build_model(cfg, stats)
    loss = build_loss(cfg, make_sde(cfg))(model, torch.randn(2, 1, 32, 32))
    loss.backward()
    assert torch.isfinite(loss)
    x, nfe = sample_batch(model, cfg, (2,1,32,32), 'cpu')
    assert torch.isfinite(x).all() and nfe == cfg.model.num_scales
    if discrete:
        with pytest.raises(ValueError): make_sde(cfg, 7)
        assert model.backbone.sigmas.dtype == torch.float32


@pytest.mark.parametrize('name', ['vpsde', 'subvpsde'])
def test_vp_and_subvp(smoke_config, name):
    cfg = smoke_config
    cfg.reference.mode = 'baseline'
    cfg.training.sde = name
    cfg.model.embedding_type = 'positional'
    cfg.model.scale_by_sigma = False
    # VP continuous uses labels t*999; preserve source 1000-label embedding buffer.
    cfg.model.num_scales = 1000
    cfg.sampling.predictor = 'euler_maruyama'
    model = build_model(cfg)
    loss = build_loss(cfg, make_sde(cfg))(model, torch.randn(2,1,32,32))
    loss.backward()
    assert torch.isfinite(loss)
    x, nfe = sample_batch(model, cfg, (1,1,32,32), 'cpu', steps=10)
    assert torch.isfinite(x).all() and nfe == 10


def test_native_resampling_gradients():
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    assert upsample_2d(x).shape == (2,3,16,16)
    assert downsample_2d(x).shape == (2,3,4,4)
    torch.testing.assert_close(upsample_2d(x), x.repeat_interleave(2,-2).repeat_interleave(2,-1))
    torch.testing.assert_close(downsample_2d(x), torch.nn.functional.avg_pool2d(x, 2))
    w = torch.randn(4,3,3,3, requires_grad=True)
    y = upsample_conv_2d(x, w, k=[1,3,3,1])
    assert y.shape == (2,4,16,16)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    assert Upsample(in_ch=3, fir=False)(x).shape == (2,3,16,16)


def test_probability_flow_euler_scalar_diffusion(smoke_config):
    sde = make_sde(smoke_config)
    x, t = torch.randn(2,1,4,4), torch.tensor([.2,.6])
    y, mean = predictor_update(sde, lambda x,t: -x, x, t, 'euler_maruyama', True)
    torch.testing.assert_close(y, mean)


def test_forward_microbatch_preserves_langevin_effective_batch(smoke_config):
    cfg = smoke_config
    cfg.sampling.corrector = 'langevin'
    stats = estimate_stats([torch.randn(4,1,32,32)])
    model = build_model(cfg, stats).eval()
    with isolated_rng(12):
        x, _ = sample_batch(model, cfg, (4,1,32,32), 'cpu')
    with isolated_rng(12):
        y, _ = sample_batch(BatchedScore(model, 2), cfg, (4,1,32,32), 'cpu')
    torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-4)


def test_ode_finite(smoke_config):
    cfg = smoke_config
    cfg.sampling.method = 'ode'
    cfg.sampling.noise_removal = False
    cfg.sampling.rtol = cfg.sampling.atol = 1e-3
    cfg.model.sigma_max = 1.
    stats = estimate_stats([torch.randn(4,1,32,32)])
    x, calls = sample_batch(build_model(cfg,stats), cfg, (1,1,32,32), 'cpu')
    assert x.shape == (1,1,32,32) and torch.isfinite(x).all() and calls > 0
