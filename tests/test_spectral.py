import torch
from fourier_score.model.spectral import conjugate_symmetrize
from fourier_score.model.reference import FourierGaussian
from fourier_score.data_loader.statistics import estimate_stats


def test_statistics_population_moments():
    x=torch.randn(23,3,8,8,dtype=torch.float64)+10
    got=estimate_stats(x.split(7),1e-8)
    mean=x.mean(0); power=torch.fft.fft2(x-mean,norm='ortho').abs().square().mean(0)
    torch.testing.assert_close(got['mean'],mean.float())
    torch.testing.assert_close(got['power'],power.float(),atol=1e-6,rtol=1e-5)
    assert got['n_effective']==23


def test_gaussian_and_b_formula():
    alpha = 0.3
    power=conjugate_symmetrize(torch.rand(1,8,8)+0.1)
    mean=torch.randn(1,8,8); ref=FourierGaussian({'mean':mean,'power':power})
    y=torch.randn(3,1,8,8); raw=torch.randn_like(y); sigma=torch.tensor([0.1,1.,10.]); a=torch.full_like(sigma,alpha)
    s=sigma[:,None,None,None]; prior=alpha**2*power[None]; denom=prior+s*s
    expected=-s*torch.fft.ifft2(torch.fft.fft2(y-alpha*mean,norm='ortho')/denom,norm='ortho').real
    expected+=torch.fft.ifft2(torch.fft.fft2(raw,norm='ortho')*(prior/denom).sqrt(),norm='ortho').real
    torch.testing.assert_close(ref.scaled_score(raw,y,a,sigma),expected)


def test_scalar_matches_flat_spectrum_without_fft(monkeypatch):
    alpha = 0.3
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
