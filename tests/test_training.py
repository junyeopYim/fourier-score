import copy
import pytest
import torch
from fourier_score.trainer.trainer import Trainer
from fourier_score.data_loader.data_loaders import build_data
from fourier_score.data_loader.statistics import prepare_stats
from fourier_score.utils import load_checkpoint
from fourier_score.trainer.checkpoints import load_inference


@pytest.mark.parametrize('parameterization,objective,gated', [
    ('fourier_gaussian','dsm',False),
    ('fourier_gaussian','normalized_residual','log_sigma'),
])
def test_resume_matches_uninterrupted(cfg,parameterization,objective,gated):
    cfg['loss'].update(type=parameterization,objective=objective)
    if gated:
        cfg['fourier']['gate'].update(mode=gated,sigma_switch=.5)
    a=copy.deepcopy(cfg); a['name']='full'; a['trainer']['iterations']=4
    first=Trainer(a); first.train()
    b=copy.deepcopy(cfg); b['name']='split'; b['trainer']['iterations']=2
    second=Trainer(b); second.train(); ckpt=load_checkpoint(second.out/'last.pt')
    b['trainer']['iterations']=4
    third=Trainer(b,ckpt); third.train()
    for name,p in first.model.state_dict().items():
        torch.testing.assert_close(p,third.model.state_dict()[name],rtol=0,atol=0)
    assert first.stream.state_dict()==third.stream.state_dict()
    assert torch.equal(first.generator.get_state(),third.generator.get_state())
    for n,v in first.ema.shadow.items(): torch.testing.assert_close(v,third.ema.shadow[n],atol=0,rtol=0)
    last,loaded_cfg,_,_=load_inference(first.out/'last.pt')
    assert loaded_cfg['loss']['objective']==objective
    assert loaded_cfg['fourier']['gate']==cfg['fourier']['gate']
    snapshot,_,_,_=load_inference(first.out/f'ema_{first.step:09d}.pt')
    for name,value in last.state_dict().items():
        torch.testing.assert_close(value,snapshot.state_dict()[name],atol=0,rtol=0)


def test_reference_statistics_no_validation_leak(cfg):
    b=build_data(cfg); stats=prepare_stats(cfg,b)
    train=set(stats['train_indices'].tolist()); val=set(stats['val_indices'].tolist())
    assert not train&val and len(train)+len(val)==len(b.full)
    direct=torch.stack([b.full[i] for i in stats['train_indices'].tolist()])
    torch.testing.assert_close(stats['mean'],direct.mean(0),atol=1e-6,rtol=1e-5)
