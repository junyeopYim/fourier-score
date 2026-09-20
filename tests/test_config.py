import copy
from pathlib import Path
import pytest
from parse_config import load_config,validate,apply_overrides
from model.objectives import OBJECTIVES

@pytest.mark.parametrize('path',sorted(Path('configs').glob('*.json')))
def test_all_presets(path):
    load_config(path)

@pytest.mark.parametrize('change',[
 'loss.type=spectral_mmse','arch.args.attention_heads=8','data_loader.args.batch_size=0',
 'backend.precision=amp','arch.args.nf=7','process.sigma_min=0','optimizer.args.lr=-1',
 'trainer.microbatch_size=0','sampling.steps=0','arch.args.attn_resolutions=[123]',
 'arch.args.dropout=1.0','loss.reduction=bogus','schema_version=2',
 'process.type=ddpm','fourier.power_floor=0','name=../escape',
 'data_loader.args.channels=2','data_loader.args.batch_size=true',
 'evaluation.frequency_bins=-1','evaluation.frequency_bins=true',
])
def test_reject_bad_settings(change):
    with pytest.raises((ValueError,TypeError)): load_config('configs/smoke.json',[change])

def test_only_loss_changes(cfg):
    for name in OBJECTIVES:
        c=validate(apply_overrides(cfg,['loss.type='+name]))
        assert c['arch']==cfg['arch'] and c['process']==cfg['process']
        c['loss']=cfg['loss']; assert c==cfg

def test_inheritance_cycle(tmp_path):
    (tmp_path/'a.json').write_text('{"extends":"b.json"}')
    (tmp_path/'b.json').write_text('{"extends":"a.json"}')
    with pytest.raises(ValueError,match='cycle'): load_config(tmp_path/'a.json')

def test_ddpm_is_explicit():
    ve=load_config('configs/cifar10.json'); ddpm=load_config('configs/cifar10_ddpm.json')
    assert ve['arch']==ddpm['arch']
    assert ddpm['process']['type']=='ddpm' and ddpm['sampling']['method']=='ddpm'
