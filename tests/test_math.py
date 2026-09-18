import pytest
import torch
from model.gaussian import estimate_stats, GaussianReference, MMSEGaussianReference, make_reference, REFERENCE_MODES, to_model_space
from model.loss import get_sde_loss_fn, get_smld_loss_fn
from sde.sde_lib import VESDE, VPSDE, subVPSDE
from utils.util import isolated_rng


def stats(shape=(1, 4, 4)):
    return estimate_stats([torch.randn(32, *shape)], floor=1e-4)


def test_preprocessing():
    x = to_model_space(torch.zeros(2, 28, 28, dtype=torch.uint8))
    assert x.shape == (2, 1, 32, 32) and torch.equal(x, torch.full_like(x, -1))
    y = to_model_space(torch.full((2, 3, 32, 32), 255, dtype=torch.uint8), 'cifar10')
    assert torch.equal(y, torch.ones_like(y))


def test_stats_population_and_parseval():
    x = torch.randn(17, 2, 4, 4)
    s = estimate_stats(x.split(3))
    mean = x.double().mean(0)
    p = torch.fft.fft2(x.double() - mean, norm='ortho').abs().square().mean(0)
    torch.testing.assert_close(s['mean'].double(), mean, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(s['power'].double(), p, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(s['isotropic_variance'].double(), p.mean(), atol=1e-7, rtol=1e-6)
    assert s['n_train'] == len(x)


@pytest.mark.parametrize('mode', REFERENCE_MODES)
def test_reference_modes_shapes_and_gradients(mode):
    s = stats()
    ref = make_reference(s, mode, sigma_min=.01, sigma_max=50)
    if mode == 'baseline':
        assert ref is None
        return
    assert sum(p.numel() for p in ref.parameters()) == 0
    x = torch.randn(3, 1, 4, 4, requires_grad=True)
    sigma = torch.tensor([.1, 1., 10.], requires_grad=True)
    y = ref(x, sigma)
    assert y.shape == x.shape and torch.isfinite(y).all()
    dx, ds = torch.autograd.grad(y.square().sum(), (x, sigma))
    assert torch.isfinite(dx).all() and torch.isfinite(ds).all()


def test_spectral_reference_against_dense_inverse():
    ref = GaussianReference(stats(), 'spectral').double()
    basis = torch.eye(16, dtype=torch.float64).reshape(16, 1, 4, 4)
    covariance = torch.fft.ifft2(torch.fft.fft2(basis, norm='ortho') * ref.power, norm='ortho').real.reshape(16, 16)
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    sigma = torch.tensor([.7], dtype=torch.float64)
    expected = torch.linalg.solve(covariance + sigma.square().item() * torch.eye(16, dtype=torch.float64), -(x - ref.mean).flatten())
    torch.testing.assert_close(ref(x, sigma).flatten(), expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('mode', ['spectral', 'isotropic'])
def test_mmse_closed_form(mode):
    ref = MMSEGaussianReference(stats(), mode).double()
    sigmas = torch.tensor([0., .01, .2, 1., 50., 1e5], dtype=torch.float64)
    p = ref.power.flatten() if mode == 'spectral' else ref.isotropic_variance.reshape(1)
    v = sigmas[:, None].square()
    expected = (p * v / (p + v)).sum(1) / p.sum()
    torch.testing.assert_close(ref.noise_weight(sigmas), expected)
    assert torch.all(torch.diff(expected) >= 0) and expected[0] == 0 and expected[-1] <= 1


def test_mmse_chunking_matches_full_sum():
    s = {'mean':torch.zeros(1,64,64), 'power':torch.ones(1,64,64)*.3, 'isotropic_variance':torch.tensor(.3)}
    ref = MMSEGaussianReference(s, 'spectral')
    sigma = torch.linspace(.01, 50., 513)
    torch.testing.assert_close(ref.noise_weight(sigma), sigma.square() / (.3 + sigma.square()))


def test_linear_weights():
    ref = make_reference(stats(), 'spectral_linear', sigma_min=.01, sigma_max=100)
    torch.testing.assert_close(ref.noise_weight(torch.tensor([.001, .01, 1., 100., 1000.])),
                               torch.tensor([0., 0., .5, 1., 1.]))


@pytest.mark.parametrize('reduce_mean', [False, True])
@pytest.mark.parametrize('likelihood', [False, True])
def test_continuous_dsm_against_explicit_formula(reduce_mean, likelihood):
    class Score(torch.nn.Module):
        def forward(self, x, sigma):
            return -.25 * x
    model = Score()
    sde = VESDE()
    x = torch.randn(3, 1, 4, 4)
    with isolated_rng(2026):
        actual = get_sde_loss_fn(sde, False, reduce_mean, True, likelihood)(model, x)
    with isolated_rng(2026):
        t = torch.rand(3) * (1 - 1e-5) + 1e-5
        z = torch.randn_like(x)
        mean, sigma = sde.marginal_prob(x, t)
        score = model(mean + sigma[:, None, None, None] * z, sigma)
        if likelihood:
            error = (score + z / sigma[:,None,None,None]).square()
            weight = sde.sde(x, t)[1].square()
        else:
            error = (score * sigma[:,None,None,None] + z).square()
            weight = 1.
        values = error.flatten(1).mean(1) if reduce_mean else .5 * error.flatten(1).sum(1)
        expected = (values * weight).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_smld_against_explicit_formula():
    class Zero(torch.nn.Module):
        def forward(self, x, labels): return torch.zeros_like(x)
    x = torch.randn(3, 1, 4, 4)
    sde = VESDE(N=7)
    with isolated_rng(72):
        actual = get_smld_loss_fn(sde, False, False)(Zero(), x)
    with isolated_rng(72):
        labels = torch.randint(0, 7, (3,))
        sigma = sde.discrete_sigmas.flip(0)[labels]
        noise = torch.randn_like(x) * sigma[:,None,None,None]
        target = -noise / sigma.square()[:,None,None,None]
        expected = (.5 * target.square().flatten(1).sum(1) * sigma.square()).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('sde', [VESDE(), VPSDE(), subVPSDE()])
def test_reverse_sde_formula(sde):
    x, t = torch.randn(2, 1, 4, 4), torch.tensor([.2, .8])
    score = lambda x, t: -x
    drift, diffusion = sde.sde(x, t)
    actual, _ = sde.reverse(score).sde(x, t)
    torch.testing.assert_close(actual, drift + diffusion[:,None,None,None].square() * x)
    pf, _ = sde.reverse(score, True).sde(x, t)
    torch.testing.assert_close(pf, drift + .5 * diffusion[:,None,None,None].square() * x)
