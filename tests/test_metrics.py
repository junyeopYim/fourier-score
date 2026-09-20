import copy
import math
import pytest
import torch
from model.metric import evaluate_dsm,frequency_band_sums,radial_frequency_bands
from trainer.trainer import Trainer
from utils.util import capture_rng


def test_radial_energy_locates_dc_and_known_sinusoid():
    ids,counts,_=radial_frequency_bands(8,8,4)
    dc=frequency_band_sums(torch.full((2,3,8,8),3.),ids,4)
    torch.testing.assert_close(dc[:,0],torch.full((2,),9.*64,dtype=torch.float64))
    assert torch.count_nonzero(dc[:,1:])==0 and counts.sum()==64
    x=torch.arange(8,dtype=torch.float64)
    wave=torch.cos(2*math.pi*2*x/8)[None,None,None,:].expand(1,2,8,8)
    energy=frequency_band_sums(wave,ids,4)[0]
    assert energy[1].item()==pytest.approx(32.)  # radius 0.25 cycles/pixel, both partners
    assert energy[[0,2,3]].sum().item()<1e-20


@pytest.mark.parametrize('bands',[6,100])
def test_spectral_dsm_reconstructs_pixel_and_noise_means(cfg,bands):
    trainer=Trainer(cfg)
    try:
        baseline=evaluate_dsm(trainer.model,cfg,trainer.bundle.validation,trainer.device)
        diagnostic=copy.deepcopy(cfg); diagnostic['evaluation']['frequency_bins']=bands
        before=capture_rng(trainer.device)
        result=evaluate_dsm(trainer.model,diagnostic,trainer.bundle.validation,trainer.device)
        after=capture_rng(trainer.device)
        assert torch.equal(before['torch'],after['torch']) and before['python']==after['python']
        for key,value in baseline.items(): assert result[key]==value
        counts=result['frequency_bands']['modes_per_channel']
        weighted=lambda values: sum(c*v for c,v in zip(counts,values) if c)/sum(counts)
        assert weighted(result['dsm_by_frequency'])==pytest.approx(result['dsm_pixel_mean'],rel=2e-6)
        for row,original in zip(result['dsm_by_noise_frequency'],result['dsm_by_noise']):
            assert row['count']==original['count']
            if row['count']: assert weighted(row['mean'])==pytest.approx(original['mean'],rel=2e-6)
            else: assert all(x is None for x in row['mean'])
        for count,value in zip(counts,result['dsm_by_frequency']):
            if count==0: assert value is None
    finally:
        trainer.log.close()
