"""PC and probability-flow ODE sampling, adapted from source sampling.py.

Native FP32 operations. Langevin norms remain BATCH means as in the source;
changing sampling batch size can therefore change samples and reported metrics.
"""
import numpy as np
import torch
from model.model import get_score_fn
from .sde_lib import VESDE, VPSDE, make_sde


def predictor_update(sde, score, x, t, name, probability_flow=False):
    if name == 'none':
        return x, x
    if name == 'ancestral_sampling':
        if probability_flow:
            raise ValueError('Ancestral sampling does not support probability_flow')
        index = (t * (sde.N - 1) / sde.T).long()
        if isinstance(sde, VESDE):
            grid = sde.discrete_sigmas.to(t.device)
            sigma = grid[index]
            adjacent = torch.where(index == 0, torch.zeros_like(t), grid[index - 1])
            mean = x + score(x, t) * (sigma.square() - adjacent.square())[:, None, None, None]
            std = torch.sqrt(adjacent.square() * (sigma.square() - adjacent.square()) / sigma.square())
        elif isinstance(sde, VPSDE):
            beta = sde.discrete_betas.to(t.device)[index]
            mean = (x + beta[:, None, None, None] * score(x, t)) / torch.sqrt(1 - beta)[:, None, None, None]
            std = torch.sqrt(beta)
        else:
            raise ValueError('Ancestral sampling requires VE or VP')
        return mean + std[:, None, None, None] * torch.randn_like(x), mean
    reverse = sde.reverse(score, probability_flow)
    if name == 'reverse_diffusion':
        f, g = reverse.discretize(x, t)
        noise = torch.randn_like(x)
        mean = x - f
        return mean + g[:, None, None, None] * noise, mean
    if name == 'euler_maruyama':
        dt = -1. / sde.N
        noise = torch.randn_like(x)
        drift, diffusion = reverse.sde(x, t)
        mean = x + drift * dt
        # Source returns scalar 0 for the probability-flow diffusion.
        if isinstance(diffusion, torch.Tensor):
            diffusion = diffusion[:, None, None, None]
        return mean + diffusion * np.sqrt(-dt) * noise, mean
    raise ValueError(f'Unknown predictor: {name}')


def corrector_update(sde, score, x, t, name, snr, n_steps):
    if name == 'none' or n_steps == 0:
        return x, x
    if isinstance(sde, VESDE):
        alpha = torch.ones_like(t)
    elif isinstance(sde, VPSDE):
        index = (t * (sde.N - 1) / sde.T).long()
        alpha = sde.alphas.to(t.device)[index]
    else:
        raise ValueError('Use corrector=none for sub-VP; source sub-VP has no alpha grid')
    mean = x
    for _ in range(n_steps):
        grad, noise = score(x, t), torch.randn_like(x)
        if name == 'langevin':
            grad_norm = torch.norm(grad.reshape(len(x), -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(len(x), -1), dim=-1).mean()
            # No clipping/regularization of the source adaptive step size.
            step = (snr * noise_norm / grad_norm).square() * 2 * alpha
        elif name == 'ald':
            std = sde.marginal_prob(x, t)[1]
            step = (snr * std).square() * 2 * alpha
        else:
            raise ValueError(f'Unknown corrector: {name}')
        mean = x + step[:, None, None, None] * grad
        x = mean + torch.sqrt(step * 2)[:, None, None, None] * noise
    return x, mean


@torch.no_grad()
def sample_batch(model, cfg, shape, device, steps=None):
    """Return unclipped [0,1]-coordinate images and actual score call count."""
    sde = make_sde(cfg, steps)
    eps = cfg.sampling.eps
    count = 0
    base_score = get_score_fn(sde, model, train=False, continuous=cfg.training.continuous)
    def score(x, t):
        nonlocal count
        count += 1
        return base_score(x, t)
    # Retain the source's CPU prior draw followed by device transfer.
    x = sde.prior_sampling(shape).to(device=device, dtype=torch.float32)
    if cfg.sampling.method == 'pc':
        for t in torch.linspace(sde.T, eps, sde.N, device=device):
            vec_t = torch.ones(shape[0], device=device) * t
            x, mean = corrector_update(sde, score, x, vec_t, cfg.sampling.corrector,
                                      cfg.sampling.snr, cfg.sampling.n_steps_each)
            x, mean = predictor_update(sde, score, x, vec_t, cfg.sampling.predictor,
                                      cfg.sampling.probability_flow)
        x = mean if cfg.sampling.noise_removal else x
    elif cfg.sampling.method == 'ode':
        if not cfg.training.continuous:
            raise ValueError('ODE sampling requires a continuously trained model')
        from scipy.integrate import solve_ivp
        reverse = sde.reverse(score, probability_flow=True)
        def rhs(t, flat):
            state = torch.from_numpy(flat.reshape(shape)).to(device=device, dtype=torch.float32)
            vec_t = torch.ones(shape[0], device=device) * t
            return reverse.sde(state, vec_t)[0].cpu().numpy().reshape(-1)
        solution = solve_ivp(rhs, (sde.T, eps), x.cpu().numpy().reshape(-1),
                             rtol=cfg.sampling.rtol, atol=cfg.sampling.atol,
                             method=cfg.sampling.ode_method)
        if not solution.success:
            raise RuntimeError(f'ODE integration failed: {solution.message}')
        x = torch.from_numpy(solution.y[:, -1].reshape(shape)).to(device=device, dtype=torch.float32)
        if cfg.sampling.noise_removal:
            _, x = predictor_update(sde, score, x, torch.full((shape[0],), eps, device=device),
                                    'reverse_diffusion')
    else:
        raise ValueError(f'Unknown sampler: {cfg.sampling.method}')
    if not torch.isfinite(x).all():
        raise FloatingPointError('Non-finite samples; inspect model, sampler and adaptive Langevin step')
    return ((x + 1.) / 2. if cfg.data.centered else x), count
