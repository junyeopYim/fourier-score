"""Small real NCSN++ forward/backward/Adam/sampling and RNG device check."""
import argparse
import copy
import torch
from parse_config import load_config
from model.model import build_model
from model.objectives import OBJECTIVES
from model.loss import training_loss
from model.spectral import SpectralFilter,conjugate_symmetrize
from sde.sampling import sample_batch
from utils.util import configure_runtime,seed_all,environment,capture_rng,restore_rng,json_write

def run(device='auto',spectral='auto'):
    cfg=load_config('configs/smoke.json',[f'device={device}',f'backend.spectral_transform={spectral}'])
    dev=configure_runtime(cfg); result=environment(dev,cfg); result['tests']=[]
    for lossname in OBJECTIVES:
        c=copy.deepcopy(cfg); c['loss']['type']=lossname; seed_all(123,dev)
        model=build_model(c,{'mean':torch.zeros(1,8,8),'power':torch.ones(1,8,8)},dev)
        gen=torch.Generator().manual_seed(123); x=torch.randn(2,1,8,8,generator=gen).to(dev)
        lev=model.process.sample(2,dev,gen); eps=torch.randn(2,1,8,8,generator=gen).to(dev)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        value=training_loss(model,x,lev,eps); value.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
        if norm<=0: raise AssertionError('Gradient did not reach backbone')
        optimizer.step(); model.eval()
        samples,nfe=sample_batch(model,c,2,dev,gen)
        result['tests'].append({'loss':lossname,'loss_value':float(value.detach()),'gradient_norm':float(norm),'sample_shape':list(samples.shape),'nfe':nfe})
    x=torch.randn(2,1,8,8,device=dev,requires_grad=True)
    w=conjugate_symmetrize(torch.rand(1,1,8,8)).to(dev)
    got=SpectralFilter(8,8,'matmul').to(dev)(x,w)
    expected=SpectralFilter(8,8,'fft')(x.detach().cpu(),w.cpu())
    error=float((got.detach().cpu()-expected).abs().max()); got.square().sum().backward()
    if error>2e-4 or not torch.isfinite(x.grad).all(): raise AssertionError('Spectral device parity failed')
    state=capture_rng(dev); a=torch.randn(4,device=dev); restore_rng(state,dev); b=torch.randn(4,device=dev)
    if not torch.equal(a,b): raise AssertionError('Device RNG restore failed')
    result.update(passed=True,matmul_fft_max_abs_error=error,rng_restore=True)
    return result

def main():
    import json
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',default='auto'); p.add_argument('--spectral',default='auto',choices=['auto','fft','matmul','cpu']); p.add_argument('-o','--output')
    a=p.parse_args(); result=run(a.device,a.spectral)
    if a.output: json_write(result,a.output)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
