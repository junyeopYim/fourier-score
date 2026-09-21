import copy
import torch
from fourier_score.model import build_model,architecture_report,weight_hash,backbone_config
from fourier_score.backbones.ncsnpp import NCSNpp
from fourier_score.method import OBJECTIVES


def test_identical_architecture_and_initialization(cfg,stats):
    reports=[]; hashes=[]
    for name in OBJECTIVES:
        c=copy.deepcopy(cfg); c['loss']['type']=name; torch.manual_seed(42)
        model=build_model(c,stats)
        reports.append(architecture_report(model.backbone)); hashes.append(weight_hash(model.backbone))
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==reports[-1]['trainable_parameters']
    assert all(report==reports[0] for report in reports)
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
