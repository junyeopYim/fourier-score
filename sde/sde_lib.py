"""SDEs from score_sde_pytorch; mathematical definitions retained."""
import abc
import numpy as np
import torch


class SDE(abc.ABC):
    def __init__(self, N):
        if not isinstance(N, int) or N < 1:
            raise ValueError('N must be a positive integer')
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sde(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def prior_sampling(self, shape):
        raise NotImplementedError

    @abc.abstractmethod
    def prior_logp(self, z):
        raise NotImplementedError

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        return drift * dt, diffusion * torch.sqrt(torch.tensor(dt, device=t.device))

    def reverse(self, score_fn, probability_flow=False):
        N, T, sde_fn, discretize_fn = self.N, self.T, self.sde, self.discretize
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if probability_flow else 1.)
                return drift, 0. if probability_flow else diffusion

            def discretize(self, x, t):
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if probability_flow else 1.)
                return rev_f, torch.zeros_like(G) if probability_flow else G
        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0, self.beta_1 = beta_min, beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return -0.5 * beta_t[:, None, None, None] * x, torch.sqrt(beta_t)

    def marginal_prob(self, x, t):
        log_mean = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        return torch.exp(log_mean[:, None, None, None]) * x, torch.sqrt(1. - torch.exp(2. * log_mean))

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        return -np.prod(z.shape[1:]) / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        return torch.sqrt(alpha)[:, None, None, None] * x - x, torch.sqrt(beta)


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0, self.beta_1 = beta_min, beta_max

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        return -0.5 * beta_t[:, None, None, None] * x, torch.sqrt(beta_t * discount)

    def marginal_prob(self, x, t):
        log_mean = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        # This IS std, not variance: do not insert an extra square root.
        return torch.exp(log_mean)[:, None, None, None] * x, 1 - torch.exp(2. * log_mean)

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        return -np.prod(z.shape[1:]) / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        if not 0 < sigma_min < sigma_max:
            raise ValueError('Expected 0 < sigma_min < sigma_max')
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), N))

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return torch.zeros_like(x), diffusion

    def marginal_prob(self, x, t):
        return x, self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        return (-np.prod(z.shape[1:]) / 2. * np.log(2 * np.pi * self.sigma_max ** 2)
                - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2))

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent = torch.where(timestep == 0, torch.zeros_like(t),
                               self.discrete_sigmas.to(t.device)[timestep - 1])
        return torch.zeros_like(x), torch.sqrt(sigma ** 2 - adjacent ** 2)


def make_sde(cfg, sampling_steps=None):
    name = cfg.training.sde.lower()
    n = cfg.model.num_scales if sampling_steps is None else sampling_steps
    if not cfg.training.continuous and n != cfg.model.num_scales:
        raise ValueError('Discrete sampling must retain the trained sigma/label grid')
    if name == 'vesde':
        return VESDE(cfg.model.sigma_min, cfg.model.sigma_max, n)
    if name in ('vpsde', 'subvpsde'):
        return (VPSDE if name == 'vpsde' else subVPSDE)(cfg.model.beta_min, cfg.model.beta_max, n)
    raise ValueError(f'Unknown SDE: {name}')
