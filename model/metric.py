"""Common pixel-mean DSM, fixed random draws, time/noise bins, no FID claims."""
import torch
from torch.utils.data import DataLoader,Subset
from model.loss import per_image_dsm
from utils.util import isolated_rng

@torch.no_grad()
def evaluate_dsm(model,cfg,dataset,device):
    opt=cfg['evaluation']; n=min(len(dataset),opt['max_images'])
    if n<1: raise ValueError('Empty evaluation set')
    bins=opt['noise_bins']; sums=torch.zeros(bins,dtype=torch.float64); counts=torch.zeros(bins,dtype=torch.long)
    total=0.; sq=0.; seen=0; old=model.training
    try:
        model.eval()
        with isolated_rng(device):
            gen=torch.Generator().manual_seed(opt['seed'])
            loader=DataLoader(Subset(dataset,range(n)),batch_size=opt['batch_size'],shuffle=False,num_workers=0,
                              generator=torch.Generator().manual_seed(opt['seed']+1))
            for clean in loader:
                lev=model.process.sample(len(clean),device,gen)
                noise=torch.randn(clean.shape,generator=gen).to(device)
                values=per_image_dsm(model,clean.to(device),lev,noise).cpu().double()
                if not torch.isfinite(values).all(): raise FloatingPointError('Nonfinite evaluation loss')
                which=(model.process.bin_coordinate(lev).cpu()*bins).long().clamp(0,bins-1)
                sums.scatter_add_(0,which,values); counts.scatter_add_(0,which,torch.ones_like(which))
                total+=values.sum().item(); sq+=values.square().sum().item(); seen+=len(values)
    finally: model.train(old)
    mean=total/seen; var=max(0.,(sq-seen*mean*mean)/max(1,seen-1))
    return {'dsm_pixel_mean':mean,'standard_error':(var/seen)**0.5,'n_images':seen,
            'eval_seed':opt['seed'],'eval_batch_size':opt['batch_size'],
            'bin_axis':'linear diffusion step' if model.process.kind=='ddpm' else 'linear t = logarithmic sigma',
            'dsm_by_noise':[{'bin':i,'count':int(counts[i]),'mean':float(sums[i]/counts[i]) if counts[i] else None} for i in range(bins)]}
