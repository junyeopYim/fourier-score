import copy
import pytest
import torch
from fourier_score.model import build_model
from fourier_score.loss import training_loss
from fourier_score.diffusion import sample_batch


def test_score_epsilon_algebraic_equivalence(cfg,stats):
    cfg1=copy.deepcopy(cfg); cfg1['loss']['type']='score'; score=build_model(cfg1,stats)
    cfg2=copy.deepcopy(cfg); cfg2['loss']['type']='diffusion'; diffusion=build_model(cfg2,stats)
    lev=score.process.level(torch.tensor([0.2,0.8])); y=torch.randn(2,1,8,8); h=torch.randn_like(y); eps=torch.randn_like(y)
    a=score.scaled_from_raw(h,y,lev); b=diffusion.scaled_from_raw(-h,y,lev)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close((a+eps).square(),(-h-eps).square(),atol=0,rtol=0)


@pytest.mark.parametrize('name,process,objective',[
    ('score','ve','dsm'), ('diffusion','ddpm','dsm'), ('fourier_gaussian','ve','dsm'),
    ('scalar_gaussian','ve','normalized_residual'),
    ('fourier_gaussian','ve','normalized_residual'),
])
def test_loss_backward_and_sampling(cfg,stats,name,process,objective):
    cfg=copy.deepcopy(cfg); cfg['loss']['type']=name; cfg['process']['type']=process
    cfg['loss']['objective']=objective
    if process=='ddpm': cfg['sampling'].update(method='ddpm',steps=cfg['process']['num_scales'])
    model=build_model(cfg,stats); g=torch.Generator().manual_seed(0)
    level=model.process.sample(2,'cpu',g); clean=torch.randn(2,1,8,8,generator=g); eps=torch.randn(clean.shape,generator=g)
    loss=training_loss(model,clean,level,eps,objective=objective); loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval(); out,nfe=sample_batch(model,cfg,2,torch.device('cpu'),g)
    assert torch.isfinite(out).all() and nfe>0
