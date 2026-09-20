"""Common DSM and optional noise-by-frequency diagnostics, no FID claims."""
import math
import torch
from torch.utils.data import DataLoader,Subset
from model.loss import noise_residual
from utils.util import isolated_rng


def radial_frequency_bands(height,width,bins):
    """Full FFT coordinates, equal radial bands in cycles/pixel, DC included."""
    if bins<1: raise ValueError('At least one frequency band is required')
    fy=torch.fft.fftfreq(height,dtype=torch.float64)
    fx=torch.fft.fftfreq(width,dtype=torch.float64)
    radius=(fy[:,None].square()+fx[None,:].square()).sqrt()
    maximum=math.sqrt(0.5)
    ids=(radius/maximum*bins).long().clamp_max(bins-1).flatten()
    return ids,torch.bincount(ids,minlength=bins),torch.linspace(0,maximum,bins+1,dtype=torch.float64).tolist()


def frequency_band_sums(residual,band_ids,bins):
    """Per-image spectral energy, summed over modes and averaged over channels.

    Diagnostic-only CPU FFT also works when training/evaluating on MPS.
    Divide by each band's mode count to obtain its mean squared residual.
    """
    spectrum=torch.fft.fft2(residual.detach().to(device='cpu',dtype=torch.float64),norm='ortho')
    energy=spectrum.abs().square().mean(1).flatten(1)
    result=torch.zeros(len(residual),bins,dtype=torch.float64)
    return result.scatter_add_(1,band_ids[None].expand(len(residual),-1),energy)


@torch.no_grad()
def evaluate_dsm(model,cfg,dataset,device,progress=None):
    opt=cfg['evaluation']; n=min(len(dataset),opt['max_images'])
    if n<1: raise ValueError('Empty evaluation set')
    bins=opt['noise_bins']; sums=torch.zeros(bins,dtype=torch.float64); counts=torch.zeros(bins,dtype=torch.long)
    frequency_bins=opt.get('frequency_bins',0)
    spectral_sums=torch.zeros(bins,frequency_bins,dtype=torch.float64)
    band_ids=mode_counts=band_edges=None
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
                residual=noise_residual(model,clean.to(device),lev,noise)
                values=residual.square().flatten(1).mean(1).cpu().double()
                if not torch.isfinite(values).all(): raise FloatingPointError('Nonfinite evaluation loss')
                which=(model.process.bin_coordinate(lev).cpu()*bins).long().clamp(0,bins-1)
                sums.scatter_add_(0,which,values); counts.scatter_add_(0,which,torch.ones_like(which))
                if frequency_bins:
                    if band_ids is None:
                        band_ids,mode_counts,band_edges=radial_frequency_bands(*residual.shape[-2:],frequency_bins)
                    spectral_sums.index_add_(0,which,frequency_band_sums(residual,band_ids,frequency_bins))
                total+=values.sum().item(); sq+=values.square().sum().item(); seen+=len(values)
                if progress is not None: progress(seen,n)
    finally: model.train(old)
    mean=total/seen; var=max(0.,(sq-seen*mean*mean)/max(1,seen-1))
    result={'dsm_pixel_mean':mean,'standard_error':(var/seen)**0.5,'n_images':seen,
            'eval_seed':opt['seed'],'eval_batch_size':opt['batch_size'],
            'bin_axis':'linear diffusion step' if model.process.kind=='ddpm' else 'linear t = logarithmic sigma',
            'dsm_by_noise':[{'bin':i,'count':int(counts[i]),'mean':float(sums[i]/counts[i]) if counts[i] else None} for i in range(bins)]}
    if frequency_bins:
        result['frequency_bands']={'edges':band_edges,'units':'radial cycles/pixel',
                                  'modes_per_channel':mode_counts.tolist(),
                                  'transform':'full orthonormal FFT; includes DC and both conjugate partners'}
        result['dsm_by_frequency']=[float(spectral_sums[:,j].sum()/(seen*mode_counts[j]))
                                    if mode_counts[j] else None for j in range(frequency_bins)]
        result['dsm_by_noise_frequency']=[
            {'noise_bin':i,'count':int(counts[i]),
             'mean':[float(spectral_sums[i,j]/(counts[i]*mode_counts[j]))
                     if counts[i] and mode_counts[j] else None for j in range(frequency_bins)]}
            for i in range(bins)]
    return result
