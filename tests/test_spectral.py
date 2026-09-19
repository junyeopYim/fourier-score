import pytest
import torch
from model.spectral import SpectralFilter,FourierGaussian,estimate_stats,conjugate_symmetrize

@pytest.mark.parametrize('shape',[(8,8),(7,9),(16,12)])
@pytest.mark.parametrize('backend',['matmul','cpu'])
def test_filter_value_and_gradient(shape,backend):
    h,w=shape; x=torch.randn(2,3,h,w,requires_grad=True)
    weight=conjugate_symmetrize(torch.rand(2,3,h,w))
    expected=SpectralFilter(h,w,'fft')(x,weight)
    got=SpectralFilter(h,w,backend)(x,weight)
    torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-5)
    probe=torch.randn_like(got)
    a=torch.autograd.grad((got*probe).sum(),x,retain_graph=True)[0]
    b=torch.autograd.grad((expected*probe).sum(),x)[0]
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


def test_statistics_population_moments():
    x=torch.randn(23,3,8,8,dtype=torch.float64)+10
    got=estimate_stats(x.split(7),1e-8)
    mean=x.mean(0); power=torch.fft.fft2(x-mean,norm='ortho').abs().square().mean(0)
    torch.testing.assert_close(got['mean'],mean.float())
    torch.testing.assert_close(got['power'],power.float(),atol=1e-6,rtol=1e-5)
    assert got['n_effective']==23


def test_floor_and_conjugate_symmetry():
    result=estimate_stats([torch.ones(4,1,8,8)],1e-4)
    assert torch.all(result['power']==1e-4) and result['floored_fraction']==1.0
    torch.testing.assert_close(result['power'],conjugate_symmetrize(result['power']))

@pytest.mark.parametrize('alpha',[1.0,0.3])
def test_gaussian_and_b_formula(alpha):
    power=conjugate_symmetrize(torch.rand(1,8,8)+0.1)
    mean=torch.randn(1,8,8); ref=FourierGaussian({'mean':mean,'power':power})
    y=torch.randn(3,1,8,8); raw=torch.randn_like(y); sigma=torch.tensor([0.1,1.,10.]); a=torch.full_like(sigma,alpha)
    s=sigma[:,None,None,None]; prior=alpha**2*power[None]; denom=prior+s*s
    expected=-s*torch.fft.ifft2(torch.fft.fft2(y-alpha*mean,norm='ortho')/denom,norm='ortho').real
    expected+=torch.fft.ifft2(torch.fft.fft2(raw,norm='ortho')*(prior/denom).sqrt(),norm='ortho').real
    torch.testing.assert_close(ref.scaled_score(raw,y,a,sigma),expected)


def test_residual_target_second_moment_non_gaussian():
    gen=torch.Generator().manual_seed(42); n=100000
    # Rademacher, explicitly non-Gaussian, E[X^2]=P.
    P=2.; sigma=1.3
    x=(torch.randint(2,(n,),generator=gen)*2-1).float()*P**0.5
    eps=torch.randn(n,generator=gen); y=x+sigma*eps
    target=-eps+sigma*y/(P+sigma**2)
    torch.testing.assert_close(target,(sigma*x-P*eps)/(P+sigma**2))
    assert abs(float(target.square().mean())-P/(P+sigma**2))<0.015


def test_invalid_statistics():
    with pytest.raises(ValueError): FourierGaussian({'mean':torch.zeros(1,8,8),'power':torch.zeros(1,8,8)})
