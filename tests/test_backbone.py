import copy
import hashlib
from pathlib import Path
import pytest
import torch
from model.model import build_model,architecture_report,weight_hash,backbone_config
from model.backbones.ncsnpp import NCSNpp
from model.loss import training_loss

@pytest.mark.parametrize('filename,sha',[
 ('ncsnpp.py','ea16eb35b5e6f2fa92db10be2552b3acdfe8fa7b'),
 ('layers.py','eb772b2e6606ed92295dd031cb43be8a82a992c7'),
 ('layerspp.py','2eb4e5de372e799f0608272408718664c35719c0')])
def test_source_files_exact(filename,sha):
    b=(Path('model/backbones')/filename).read_bytes()
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==sha


def test_identical_architecture_and_initialization(cfg,stats):
    reports=[]; hashes=[]
    for name in ('score','diffusion','fourier_gaussian'):
        c=copy.deepcopy(cfg); c['loss']['type']=name; torch.manual_seed(42)
        model=build_model(c,stats)
        reports.append(architecture_report(model.backbone)); hashes.append(weight_hash(model.backbone))
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==reports[-1]['trainable_parameters']
    assert reports[0]==reports[1]==reports[2]
    assert len(set(hashes))==1
    assert 'AttnBlockpp' in reports[0]['modules'].values()
    assert 'ResnetBlockBigGANpp' in reports[0]['modules'].values()


def test_baseline_matches_original_sigma_output(cfg,stats):
    c=copy.deepcopy(cfg); c['loss']['type']='score'
    model=build_model(c,stats); model.eval()
    sourcecfg=backbone_config(c); sourcecfg.model.scale_by_sigma=True
    source=NCSNpp(sourcecfg).float().eval(); source.load_state_dict(model.backbone.state_dict())
    x=torch.randn(2,1,8,8); lev=model.process.level(torch.tensor([0.2,0.8]))
    torch.testing.assert_close(model(x,lev),source(x,lev.sigma),atol=0,rtol=0)

@pytest.mark.parametrize('resblock',['biggan','ddpm'])
@pytest.mark.parametrize('fir',[True,False])
@pytest.mark.parametrize('progressive',['none','output_skip','residual'])
@pytest.mark.parametrize('progressive_input',['none','input_skip','residual'])
def test_all_backbone_paths(cfg,stats,resblock,fir,progressive,progressive_input):
    cfg=copy.deepcopy(cfg); a=cfg['arch']['args']
    a.update(resblock_type=resblock,fir=fir,progressive=progressive,progressive_input=progressive_input)
    model=build_model(cfg,stats)
    gen=torch.Generator().manual_seed(3); lev=model.process.sample(2,'cpu',gen)
    x=torch.randn(2,1,8,8); loss=training_loss(model,x,lev,torch.randn_like(x)); loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and bool(p.grad.abs().max()>0) for p in model.parameters())

@pytest.mark.parametrize('embedding',['fourier','positional'])
def test_embeddings(cfg,stats,embedding):
    cfg['arch']['args']['embedding_type']=embedding
    model=build_model(cfg,stats); lev=model.process.level(torch.tensor([0.1,0.9]))
    out=model(torch.randn(2,1,8,8),lev); assert out.shape==(2,1,8,8)
