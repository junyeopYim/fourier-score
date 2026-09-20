import copy
import pytest
import torch
from fourier_score.model import build_model
from fourier_score.loss import training_loss
from fourier_score.diffusion import NoiseProcess
from fourier_score.diffusion import sample_batch
from fourier_score.method import OBJECTIVES


def test_score_epsilon_algebraic_equivalence(cfg,stats):
    cfg1=copy.deepcopy(cfg); cfg1['loss']['type']='score'; score=build_model(cfg1,stats)
    cfg2=copy.deepcopy(cfg); cfg2['loss']['type']='diffusion'; diffusion=build_model(cfg2,stats)
    lev=score.process.level(torch.tensor([0.2,0.8])); y=torch.randn(2,1,8,8); h=torch.randn_like(y); eps=torch.randn_like(y)
    a=score.scaled_from_raw(h,y,lev); b=diffusion.scaled_from_raw(-h,y,lev)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close((a+eps).square(),(-h-eps).square(),atol=0,rtol=0)


def test_ddpm_forward_coefficients(cfg):
    cfg['process']['type']='ddpm'; p=NoiseProcess(cfg['process'])
    lev=p.level(torch.tensor([0,p.N-1])); beta=p.betas
    expected=(1-beta.double()).cumprod(0).float()[[0,-1]]
    torch.testing.assert_close(lev.alpha.square(),expected)
    torch.testing.assert_close(lev.alpha.square()+lev.sigma.square(),torch.ones(2))

@pytest.mark.parametrize('name',OBJECTIVES)
@pytest.mark.parametrize('process',['ve','ddpm'])
def test_loss_backward_and_sampling(cfg,stats,name,process):
    cfg=copy.deepcopy(cfg); cfg['loss']['type']=name; cfg['process']['type']=process
    if process=='ddpm': cfg['sampling'].update(method='ddpm',steps=cfg['process']['num_scales'])
    model=build_model(cfg,stats); g=torch.Generator().manual_seed(0)
    level=model.process.sample(2,'cpu',g); clean=torch.randn(2,1,8,8,generator=g); eps=torch.randn(clean.shape,generator=g)
    loss=training_loss(model,clean,level,eps); loss.backward()
    model.eval(); out,nfe=sample_batch(model,cfg,2,torch.device('cpu'),g)
    assert torch.isfinite(out).all() and nfe>0


def test_ve_pc_gaussian_oracle(cfg,stats):
    # A zero learned residual makes the Fourier reference an analytic score.
    model=build_model(cfg,stats)
    class Zero(torch.nn.Module):
        def forward(self,x,t): return torch.zeros_like(x)
    model.backbone=Zero(); cfg['sampling'].update(method='pc',corrector_steps=1,steps=4)
    out,nfe=sample_batch(model,cfg,2,torch.device('cpu'),torch.Generator().manual_seed(1))
    assert nfe==8 and torch.isfinite(out).all()
