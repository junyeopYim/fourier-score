import copy
import json
import pytest
import torch
from fourier_score.training import Trainer
from fourier_score.data import build_data,prepare_stats
from fourier_score.data import BatchStream
from fourier_score.utils import load_checkpoint
from fourier_score.checkpoints import load_inference
from fourier_score.config import validate


@pytest.mark.parametrize('parameterization,objective,gated', [
    ('fourier_gaussian','dsm',False),
    ('scalar_gaussian','normalized_residual',False),
    ('fourier_gaussian','normalized_residual',False),
    ('fourier_gaussian','normalized_residual','log_sigma'),
    ('fourier_gaussian','normalized_residual','log_sigma_plateau'),
    ('fourier_gaussian','normalized_residual','spectral_cap'),
    ('fourier_gaussian','normalized_residual','linear_sigma'),
    ('fourier_gaussian','normalized_residual','tanh_sigma'),
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
    b['trainer']['console']='quiet'; b['trainer']['progress_every_seconds']=1.
    third=Trainer(b,ckpt); third.train()
    for name,p in first.model.state_dict().items():
        torch.testing.assert_close(p,third.model.state_dict()[name],rtol=0,atol=0)
    assert first.stream.state_dict()==third.stream.state_dict()
    assert torch.equal(first.generator.get_state(),third.generator.get_state())
    for n,v in first.ema.shadow.items(): torch.testing.assert_close(v,third.ema.shadow[n],atol=0,rtol=0)
    last,loaded_cfg,_,_=load_inference(first.out/'last.pt')
    assert loaded_cfg['loss']['objective']==objective
    assert loaded_cfg['fourier']['gate']==cfg['fourier']['gate']
    if gated:
        from fourier_score.diffusion import sample_batch
        generated=sample_batch(last,loaded_cfg,2,torch.device('cpu'),torch.Generator().manual_seed(50))
        assert torch.isfinite(generated[0]).all()
        changed=copy.deepcopy(cfg)
        changed['fourier']['gate']['sigma_switch']=1.
        with pytest.raises(ValueError,match='Resume config mismatch'):
            Trainer(changed,load_checkpoint(first.out/'last.pt'))
        with pytest.raises(ValueError,match='Inference cannot alter'):
            load_inference(first.out/'last.pt',['fourier.gate.sigma_switch=1.0'])
        if gated in ('log_sigma_plateau', 'spectral_cap'):
            changed=copy.deepcopy(cfg)
            changed['fourier']['gate']['sigma_hi']=1.1
            with pytest.raises(ValueError,match='Resume config mismatch'):
                Trainer(changed,load_checkpoint(first.out/'last.pt'))
        if gated == 'spectral_cap':
            changed=copy.deepcopy(cfg)
            changed['fourier']['gate']['delta']=.25
            with pytest.raises(ValueError,match='Resume config mismatch'):
                Trainer(changed,load_checkpoint(first.out/'last.pt'))
    snapshot,_,_,_=load_inference(first.out/f'ema_{first.step:09d}.pt')
    for name,value in last.state_dict().items():
        torch.testing.assert_close(value,snapshot.state_dict()[name],atol=0,rtol=0)
    records=[json.loads(line) for line in (first.out/'metrics.jsonl').read_text().splitlines()]
    assert all(r['objective']==objective for r in records if r['split']=='train')
    assert any('dsm_pixel_mean' in r for r in records)


@pytest.mark.parametrize('source_change', ['edited', 'missing'])
def test_checkpoint_keeps_startup_source_when_checkout_changes(cfg, monkeypatch, source_change):
    trainer = Trainer(cfg)
    startup_hash = trainer.env['source_sha256']
    trainer.train_step(trainer.stream.next_batch())

    def changed_source_hash():
        if source_change == 'missing':
            raise FileNotFoundError('Source file moved while training was running')
        return '0' * 64

    monkeypatch.setattr('fourier_score.training.source_hash', changed_source_hash)
    try:
        trainer.save(snapshot=True)
    finally:
        trainer.log.close()
    checkpoint = load_checkpoint(trainer.out / 'last.pt')
    snapshot = load_checkpoint(trainer.out / f'ema_{trainer.step:09d}.pt')
    for state in (checkpoint, snapshot):
        assert state['step'] == 1
        assert state['source_sha256'] == startup_hash
        assert state['source_sha256'] == state['environment']['source_sha256']

    # Freezing save-time provenance must not bypass the resume source guard.
    if source_change == 'edited':
        with pytest.raises(ValueError, match='Source differs from checkpoint'):
            Trainer(cfg, checkpoint)


@pytest.mark.parametrize('objective,patch', [
    ('dsm',{'type':'scalar_gaussian'}),
    ('dsm',{'objective':'normalized_residual'}),
    ('normalized_residual',{'objective':'dsm'}),
])
def test_resume_rejects_changes(cfg,objective,patch):
    cfg['loss']['objective']=objective
    t=Trainer(cfg); t.train(); state=load_checkpoint(t.out/'last.pt')
    c=copy.deepcopy(cfg); c['trainer']['iterations']=4
    c['loss'].update(patch)
    with pytest.raises(ValueError,match='Resume config mismatch'): Trainer(c,state)


def test_legacy_dsm_checkpoint_without_objective(cfg,tmp_path):
    trainer=Trainer(cfg)
    try:
        trainer.save()
    finally:
        trainer.log.close()
    state=load_checkpoint(trainer.out/'last.pt')
    del state['config']['loss']['objective']
    assert 'objective' not in state['signature']['loss']
    path=tmp_path/'legacy.pt'
    torch.save(state,path)
    _,loaded_cfg,_,_=load_inference(path)
    assert loaded_cfg['loss']['objective']=='dsm'
    resumed=Trainer(validate(state['config']),state)
    resumed.log.close()


def test_reference_statistics_no_validation_leak(cfg):
    b=build_data(cfg); stats=prepare_stats(cfg,b)
    train=set(stats['train_indices'].tolist()); val=set(stats['val_indices'].tolist())
    assert not train&val and len(train)+len(val)==len(b.full)
    direct=torch.stack([b.full[i] for i in stats['train_indices'].tolist()])
    torch.testing.assert_close(stats['mean'],direct.mean(0),atol=1e-6,rtol=1e-5)


def test_prefetch_cursor_resume(cfg):
    workers = 2
    b=build_data(cfg)
    a=BatchStream(b.train,2,100,workers); saved=None
    for _ in range(3): a.next_batch(); a.advance()
    saved=a.state_dict(); expected=a.next_batch()
    second=BatchStream(b.train,2,100,workers); second.load_state_dict(saved)
    torch.testing.assert_close(second.next_batch(),expected,rtol=0,atol=0)


def test_cumulative_time_excludes_resume_downtime(cfg,monkeypatch):
    clock={'now':10.}
    monkeypatch.setattr('fourier_score.training.time.perf_counter',lambda:clock['now'])
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
