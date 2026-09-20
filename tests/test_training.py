import copy
import json
import pytest
import torch
from trainer.trainer import Trainer
from data_loader.data_loaders import build_data,prepare_stats
from base.base_data_loader import BaseDataLoader
from model.metric import evaluate_dsm
from utils.util import load_checkpoint,capture_rng
from utils.inference import load_inference
from parse_config import experiment_name
from model.objectives import GAUSSIAN_OBJECTIVES


@pytest.mark.parametrize('objective',GAUSSIAN_OBJECTIVES)
def test_resume_matches_uninterrupted(cfg,objective):
    cfg['loss']['type']=objective
    a=copy.deepcopy(cfg); a['name']='full'; a['trainer']['iterations']=4
    first=Trainer(a); first.train()
    b=copy.deepcopy(cfg); b['name']='split'; b['trainer']['iterations']=2
    second=Trainer(b); second.train(); ckpt=load_checkpoint(second.out/'last.pt')
    b['trainer']['iterations']=4
    b['trainer']['console']='quiet'; b['trainer']['progress_every_seconds']=1.
    third=Trainer(b,ckpt); third.train()
    for name,p in first.model.state_dict().items():
        torch.testing.assert_close(p,third.model.state_dict()[name],rtol=0,atol=0)
    assert first.stream.state_dict()==third.stream.state_dict()
    assert torch.equal(first.generator.get_state(),third.generator.get_state())
    for n,v in first.ema.shadow.items(): torch.testing.assert_close(v,third.ema.shadow[n],atol=0,rtol=0)


@pytest.mark.parametrize('objective',GAUSSIAN_OBJECTIVES)
def test_ema_snapshot_matches_last(cfg,objective):
    cfg['loss']['type']=objective
    t=Trainer(cfg); t.train()
    a,_,_,_=load_inference(t.out/'last.pt'); b,_,_,_=load_inference(t.out/f'ema_{t.step:09d}.pt')
    for n,p in a.state_dict().items(): torch.testing.assert_close(p,b.state_dict()[n],atol=0,rtol=0)


def test_eval_isolated_and_repeatable(cfg):
    t=Trainer(cfg); before=capture_rng(t.device)
    a=evaluate_dsm(t.model,cfg,t.bundle.validation,t.device); after=capture_rng(t.device)
    progress=[]
    b=evaluate_dsm(t.model,cfg,t.bundle.validation,t.device,progress=lambda done,total:progress.append((done,total)))
    assert a==b and torch.equal(before['torch'],after['torch'])
    assert progress==[(2,4),(4,4)]
    assert before['python']==after['python']; t.log.close()


@pytest.mark.parametrize('reduction',['mean','half_sum'])
def test_readable_training_preserves_diagnostics_and_reports_first_step(cfg,capsys,reduction):
    cfg['loss']['reduction']=reduction
    cfg['trainer']['log_every']=50
    cfg['evaluation']['frequency_bins']=6
    trainer=Trainer(cfg); trainer.train()
    output=capsys.readouterr()
    assert output.out=='' and '\r' not in output.err and '\x1b' not in output.err
    assert '[train] 1/3' in output.err and '[train] 3/3 (100.0%)' in output.err
    assert 'ETA(train)=' in output.err and 'pixel(avg)=' in output.err
    assert '[eval] step=0' in output.err and '[saved] step=3' in output.err and '[done]' in output.err
    assert 'dsm_by_noise_frequency' not in output.err
    records=[json.loads(line) for line in (trainer.out/'metrics.jsonl').read_text().splitlines()]
    train=[r for r in records if r['split']=='train']
    assert len(train)==1  # Console refreshes must not change the JSONL logging cadence.
    record=train[0]
    assert record['window_steps']==3 and record['window_images']==6
    scale=2/(1*8*8) if reduction=='half_sum' else 1.
    assert record['loss_pixel_mean']==pytest.approx(record['loss']*scale)
    assert record['loss_pixel_mean_avg']==pytest.approx(record['loss_avg']*scale)
    assert record['steps_per_second']>0 and record['images_per_second']>0
    assert record['eta_train_seconds']==0
    assert 'dsm_by_noise_frequency' in records[0] and 'dsm_by_frequency' in records[0]


@pytest.mark.parametrize('mode',['json','quiet'])
def test_machine_and_quiet_console_modes(cfg,capsys,mode):
    cfg['trainer']['console']=mode
    trainer=Trainer(cfg); trainer.train()
    output=capsys.readouterr()
    assert output.err==''
    metrics=(trainer.out/'metrics.jsonl').read_text()
    assert output.out==(metrics if mode=='json' else '')


def test_report_averages_weight_partial_batches_and_exclude_evaluation_time(cfg,monkeypatch):
    cfg['trainer'].update(iterations=5,log_every=3,eval_every=1,save_every=100,snapshot_every=100,console='quiet')
    cfg['data_loader']['args'].update(synthetic_size=9,validation_size=4)  # Batches of 2, 2, 1.
    trainer=Trainer(cfg)
    clock={'now':0.}
    monkeypatch.setattr('trainer.trainer.time.perf_counter',lambda:clock['now'])
    def train_step(clean):
        clock['now']+=2.
        trainer.step+=1; trainer.stream.advance()
        return float(trainer.step),1.
    def evaluate(): clock['now']+=100.
    monkeypatch.setattr(trainer,'train_step',train_step)
    monkeypatch.setattr(trainer,'evaluate',evaluate)
    monkeypatch.setattr(trainer,'save',lambda snapshot:None)
    trainer.train()
    records=[json.loads(line) for line in (trainer.out/'metrics.jsonl').read_text().splitlines()]
    first,last=records
    assert first['window_steps']==3 and first['window_images']==5
    assert first['loss_avg']==pytest.approx((1*2+2*2+3*1)/5)
    assert first['steps_per_second']==pytest.approx(0.5)
    assert first['images_per_second']==pytest.approx(5/6)
    assert first['eta_train_seconds']==pytest.approx(4.)
    assert last['window_steps']==2 and last['loss_avg']==pytest.approx(4.5)


def test_interruption_closes_logging_without_success_message(cfg,monkeypatch,capsys):
    trainer=Trainer(cfg)
    def interrupt(clean): raise KeyboardInterrupt
    monkeypatch.setattr(trainer,'train_step',interrupt)
    with pytest.raises(KeyboardInterrupt): trainer.train()
    assert trainer.log.file.closed
    output=capsys.readouterr().err
    assert '[stopped] step=0/3' in output and '[done]' not in output

@pytest.mark.parametrize('change',[('loss','type','score'),('loss','type','scalar_gaussian'),
                                  ('loss','type','fourier_gaussian_unscaled'),
                                  ('arch','args',None),('process','sigma_max',3.0)])
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


def test_cumulative_time_excludes_resume_downtime(cfg,monkeypatch):
    clock={'now':10.}
    monkeypatch.setattr('base.base_trainer.time.perf_counter',lambda:clock['now'])
    first=Trainer(cfg)
    clock['now']=17.
    first.save(snapshot=True); first.log.close()
    checkpoint=load_checkpoint(first.out/'last.pt')
    snapshot=load_checkpoint(first.out/'ema_000000000.pt')
    assert checkpoint['training_wall_seconds']==snapshot['training_wall_seconds']==7.
    clock['now']=1000.
    resumed=Trainer(cfg,checkpoint)
    assert resumed.training_wall_seconds()==7.
    clock['now']=1004.
    resumed.save(); resumed.log.close()
    assert load_checkpoint(resumed.out/'last.pt')['training_wall_seconds']==11.


def test_old_snapshot_accepts_new_diagnostic_override(cfg,tmp_path):
    t=Trainer(cfg); t.train()
    state=load_checkpoint(t.out/'last.pt')
    del state['config']['evaluation']['frequency_bins']
    legacy=tmp_path/'old.pt'; torch.save(state,legacy)
    _,resolved,_,_=load_inference(legacy,['evaluation.frequency_bins=6'])
    assert resolved['evaluation']['frequency_bins']==6
