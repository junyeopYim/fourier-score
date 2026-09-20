"""Fourier-diagonal Gaussian score and frequencywise residual scaling.

Full orthonormal FFT: P_k = E|F(X-mu)_k|^2. No extra factor 1/2.
No scalar gate, no learned gate, no independent-frequency neural networks.
"""
from __future__ import annotations
import math
import torch
from torch import nn


def conjugate_symmetrize(power: torch.Tensor) -> torch.Tensor:
    h,w=power.shape[-2:]
    iy=(-torch.arange(h,device=power.device))%h
    ix=(-torch.arange(w,device=power.device))%w
    return 0.5*(power+power.index_select(-2,iy).index_select(-1,ix))


@torch.no_grad()
def estimate_stats(batches, floor: float = 1e-4) -> dict:
    """Float64 CPU accumulation, float32 saved tensors, population moments."""
    if floor<=0: raise ValueError('floor must be positive')
    n=0; mean=None; m2=None
    # Batch-combined Welford in image space / spectral second moments.
    for batch in batches:
        x=batch.detach().to(device='cpu',dtype=torch.float64)
        if x.ndim!=4 or not len(x) or not torch.isfinite(x).all():
            raise ValueError('Statistics require nonempty finite BCHW data')
        b=len(x); bm=x.mean(0)
        z=torch.fft.fft2(x-bm, norm='ortho')
        bm2=z.abs().square().sum(0)
        if mean is None:
            mean=bm; m2=bm2; n=b
        else:
            if bm.shape!=mean.shape: raise ValueError('Inconsistent image shapes')
            delta=bm-mean
            m2 += bm2 + torch.fft.fft2(delta,norm='ortho').abs().square()*(n*b/(n+b))
            mean += delta*(b/(n+b)); n+=b
    if n<2: raise ValueError('At least two training images required')
    raw=conjugate_symmetrize(m2/n).clamp_min(0)
    return {'mean':mean.float(),'power':raw.clamp_min(floor).float(),
            'n_effective':n,'floor':float(floor),
            'floored_fraction':float((raw<floor).double().mean())}


class SpectralFilter(nn.Module):
    """Apply a real, conjugate-symmetric multiplier without severing autograd.

    auto: FFT on CPU/CUDA, separable real DFT matmul on MPS.
    matmul: no complex tensors and no hidden CPU transfers in forward/backward.
    cpu: explicitly transfer the FFT branch to CPU, preserving gradients.
    """
    def __init__(self, height: int, width: int, backend: str = 'auto'):
        super().__init__()
        if backend not in ('auto','fft','matmul','cpu'): raise ValueError(backend)
        self.backend=backend; self.height=height; self.width=width
        # Compute trigonometry in float64 once on CPU, then store FP32.
        for axis,size in (('h',height),('w',width)):
            t=torch.arange(size,dtype=torch.float64)
            angle=(2*math.pi/size)*t[:,None]*t[None,:]
            self.register_buffer('cos_'+axis,(angle.cos()/math.sqrt(size)).float(),persistent=False)
            self.register_buffer('sin_'+axis,(angle.sin()/math.sqrt(size)).float(),persistent=False)

    def resolved_backend(self,device):
        return ('matmul' if device.type=='mps' else 'fft') if self.backend=='auto' else self.backend

    def forward(self,x:torch.Tensor,multiplier:torch.Tensor)->torch.Tensor:
        if tuple(x.shape[-2:])!=(self.height,self.width): raise ValueError('Spectral shape mismatch')
        method=self.resolved_backend(x.device)
        if method in ('fft','cpu'):
            work=x.to('cpu') if method=='cpu' else x
            # .to() is differentiable; intentionally never detach here.
            weight=multiplier.to(work.device,dtype=work.dtype)
            out=torch.fft.ifft2(torch.fft.fft2(work,norm='ortho')*weight,norm='ortho').real
            return out.to(x.device)
        ch,sh=self.cos_h.to(x),self.sin_h.to(x)
        cw,sw=self.cos_w.to(x),self.sin_w.to(x)
        # (Ch-iSh) x (Cw-iSw)^T, then the real part of its inverse.
        hr=ch@x; hi=-(sh@x)
        real=(hr@cw.T+hi@sw.T)*multiplier
        imag=(hi@cw.T-hr@sw.T)*multiplier
        ar=ch.T@real-sh.T@imag
        ai=ch.T@imag+sh.T@real
        return ar@cw-ai@sw


class FourierGaussian(nn.Module):
    def __init__(self,stats:dict,backend='auto',*,covariance='fourier',scale_residual=True):
        super().__init__()
        if covariance not in ('fourier','scalar'): raise ValueError('Invalid Gaussian covariance')
        if type(scale_residual) is not bool: raise ValueError('scale_residual must be boolean')
        mean=stats['mean'].detach().cpu().float().clone()
        power=stats['power'].detach().cpu().float().clone()
        if mean.ndim!=3 or power.shape!=mean.shape: raise ValueError('Statistics must be [C,H,W]')
        if not torch.isfinite(mean).all() or not torch.isfinite(power).all() or (power<=0).any(): raise ValueError('Invalid Fourier statistics')
        if not torch.allclose(power,conjugate_symmetrize(power),atol=1e-6,rtol=1e-5): raise ValueError('Power is not conjugate symmetric')
        self.covariance=covariance
        self.scale_residual=scale_residual
        # Average the SAME floored training spectrum, retaining the full mean.
        # The shared statistics cache must never be flattened in place.
        if covariance=='scalar': power=power.mean(dim=(-2,-1),keepdim=True)
        self.register_buffer('mean',mean)
        self.register_buffer('power',power)
        self.filter=SpectralFilter(*mean.shape[-2:],backend)

    def resolved_backend(self,device):
        return 'elementwise' if self.covariance=='scalar' else self.filter.resolved_backend(device)

    def scaled_score(self,raw,y,alpha,sigma):
        """Return sigma * score, not score. raw is the unscaled NCSN++ output.

        VE alpha=1 is the proposed method. For explicit DDPM experiments,
        V_k = alpha^2 P_k, mean_t = alpha mu (documented generalization).
        """
        a=alpha[:,None,None,None]; s=sigma[:,None,None,None]
        prior=a.square()*self.power[None]
        denom=prior+s.square()
        if self.covariance=='scalar':
            # A spatially constant spectral multiplier is pointwise in pixels.
            gaussian=-s*(y-a*self.mean[None])/denom
            residual=(prior/denom).sqrt()*raw if self.scale_residual else raw
            return gaussian+residual
        gaussian=-s*self.filter(y-a*self.mean[None],denom.reciprocal())
        residual=self.filter(raw,(prior/denom).sqrt()) if self.scale_residual else raw
        return gaussian+residual
