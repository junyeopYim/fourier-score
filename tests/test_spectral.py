import pytest
import torch
from fourier_score.spectral import SpectralFilter,conjugate_symmetrize
from fourier_score.method import FourierGaussian
from fourier_score.statistics import estimate_stats

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


@pytest.mark.parametrize('alpha',[1.0,0.3])
def test_scalar_matches_flat_spectrum_without_fft(alpha,monkeypatch):
    # Nonuniform spectra and nonconstant means catch accidental mean removal,
    # channel pooling, or modification of the shared cached statistics.
    mean=torch.randn(3,7,9)
    power=conjugate_symmetrize(torch.rand_like(mean)+0.1)*torch.tensor([1.,2.,4.])[:,None,None]
    stored={'mean':mean.clone(),'power':power.clone()}
    scalar=FourierGaussian(stored,covariance='scalar')
    flat=power.mean((-2,-1),keepdim=True).expand_as(power).clone()
    reference=FourierGaussian({'mean':mean,'power':flat})
    y=torch.randn(3,3,7,9); raw=torch.randn_like(y,requires_grad=True)
    sigma=torch.tensor([0.1,1.,10.]); a=torch.full_like(sigma,alpha)
    expected=reference.scaled_score(raw,y,a,sigma)
    def fail_fft(*args,**kwargs): raise AssertionError('Scalar Gaussian should use pointwise operations')
    monkeypatch.setattr(scalar.filter,'forward',fail_fft)
    got=scalar.scaled_score(raw,y,a,sigma)
    torch.testing.assert_close(got,expected)
    probe=torch.randn_like(got)
    grad=torch.autograd.grad((got*probe).sum(),raw,retain_graph=True)[0]
    expected_grad=torch.autograd.grad((expected*probe).sum(),raw)[0]
    torch.testing.assert_close(grad,expected_grad)
    torch.testing.assert_close(scalar.mean,mean,atol=0,rtol=0)
    torch.testing.assert_close(stored['power'],power,atol=0,rtol=0)
    torch.testing.assert_close(stored['mean'],mean,atol=0,rtol=0)


@pytest.mark.parametrize('backend',['fft','matmul'])
def test_unscaled_keeps_reference_and_identity_residual(backend):
    stats={'mean':torch.randn(2,8,8),'power':conjugate_symmetrize(torch.rand(2,8,8)+0.1)}
    scaled=FourierGaussian(stats,backend)
    unscaled=FourierGaussian(stats,backend,scale_residual=False)
    y=torch.randn(3,2,8,8); raw=torch.randn_like(y,requires_grad=True)
    sigma=torch.tensor([0.1,1.,10.]); alpha=torch.tensor([1.,0.8,0.3])
    baseline=scaled.scaled_score(torch.zeros_like(raw),y,alpha,sigma)
    got=unscaled.scaled_score(raw,y,alpha,sigma)
    torch.testing.assert_close(got,baseline+raw,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(got.sum(),raw)[0],torch.ones_like(raw),atol=0,rtol=0)
