import copy
import pytest
import torch
from trainer.trainer import Trainer
from data_loader.data_loaders import build_data,prepare_stats
from base.base_data_loader import BaseDataLoader
from model.metric import evaluate_dsm
from utils.util import load_checkpoint,capture_rng
from utils.inference import load_inference
from parse_config import experiment_name


def test_resume_matches_uninterrupted(cfg):
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


def test_ema_snapshot_matches_last(cfg):
    t=Trainer(cfg); t.train()
    a,_,_,_=load_inference(t.out/'last.pt'); b,_,_,_=load_inference(t.out/f'ema_{t.step:09d}.pt')
    for n,p in a.state_dict().items(): torch.testing.assert_close(p,b.state_dict()[n],atol=0,rtol=0)


def test_eval_isolated_and_repeatable(cfg):
    t=Trainer(cfg); before=capture_rng(t.device)
    a=evaluate_dsm(t.model,cfg,t.bundle.validation,t.device); after=capture_rng(t.device)
    b=evaluate_dsm(t.model,cfg,t.bundle.validation,t.device)
    assert a==b and torch.equal(before['torch'],after['torch'])
    assert before['python']==after['python']; t.log.close()

@pytest.mark.parametrize('change',[('loss','type','score'),('arch','args',None),('process','sigma_max',3.0)])
def test_resume_rejects_changes(cfg,change):
    t=Trainer(cfg); t.train(); state=load_checkpoint(t.out/'last.pt')
    c=copy.deepcopy(cfg); c['trainer']['iterations']=4
    group,key,val=change
    if group=='arch': c['arch']['args']['num_res_blocks']=2
    else: c[group][key]=val
    with pytest.raises(ValueError,match='Resume config mismatch'): Trainer(c,state)


def test_reference_statistics_no_validation_leak(cfg):
    b=build_data(cfg); stats=prepare_stats(cfg,b)
    train=set(stats['train_indices'].tolist()); val=set(stats['val_indices'].tolist())
    assert not train&val and len(train)+len(val)==len(b.full)
    direct=torch.stack([b.full[i] for i in stats['train_indices'].tolist()])
    torch.testing.assert_close(stats['mean'],direct.mean(0),atol=1e-6,rtol=1e-5)


def test_augmentation_mixture_stats(cfg):
    cfg['data_loader']['args']['random_flip']=True
    b=build_data(cfg); s=prepare_stats(cfg,b)
    assert s['n_effective']==2*s['n_train']
    torch.testing.assert_close(s['mean'],s['mean'].flip(-1),atol=1e-6,rtol=1e-5)

@pytest.mark.parametrize('workers',[0,2])
def test_prefetch_cursor_resume(cfg,workers):
    b=build_data(cfg)
    a=BaseDataLoader(b.train,2,100,workers); saved=None
    for _ in range(3): a.next_batch(); a.advance()
    saved=a.state_dict(); expected=a.next_batch()
    second=BaseDataLoader(b.train,2,100,workers); second.load_state_dict(saved)
    torch.testing.assert_close(second.next_batch(),expected,rtol=0,atol=0)


def test_inference_rejects_loss_change(cfg):
    t=Trainer(cfg); t.train()
    with pytest.raises(ValueError,match='Inference cannot alter'): load_inference(t.out/'last.pt',['loss.type=score'])
