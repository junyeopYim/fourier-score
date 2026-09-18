"""Model factory and the original score/output convention adapters."""
import torch
from .backbones.ncsnpp import NCSNpp
from .gaussian import ResidualScore, make_reference
from sde.sde_lib import VESDE, VPSDE, subVPSDE, make_sde


class DiscreteResidualScore(torch.nn.Module):
    """Labels go to NCSN++; their actual SMLD sigmas go to the reference."""
    def __init__(self, backbone, reference, sigmas):
        super().__init__()
        self.backbone, self.reference = backbone, reference
        self.register_buffer('reference_sigmas', sigmas.clone())

    def forward(self, x, labels):
        residual = self.backbone(x, labels)
        if self.reference is None:
            return residual
        return residual + self.reference(x, self.reference_sigmas[labels.long()])


def build_model(cfg, stats=None, device='cpu'):
    if cfg.model.name != 'ncsnpp':
        raise ValueError('This lean template retains NCSN++ and its DDPM++ configuration only')
    mode = cfg.reference.mode
    if cfg.training.sde.lower() != 'vesde' and mode != 'baseline':
        raise ValueError('Gaussian residual modes are defined for VE/SMLD, not VP/subVP')
    base = NCSNpp(cfg)
    reference = make_reference(stats or {}, mode, sigma_min=cfg.model.sigma_min, sigma_max=cfg.model.sigma_max)
    if cfg.training.sde.lower() == 'vesde' and not cfg.training.continuous:
        model = DiscreteResidualScore(base, reference, make_sde(cfg).discrete_sigmas.flip(0))
    else:
        model = ResidualScore(base, reference)
    # Avoid the upstream positional sigma buffer's float64 promotion.
    return model.to(device=device, dtype=torch.float32)


def get_model_fn(model, train=False):
    def model_fn(x, labels):
        model.train(train)
        return model(x, labels)
    return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    model_fn = get_model_fn(model, train)
    if isinstance(sde, (VPSDE, subVPSDE)):
        def score_fn(x, t):
            if continuous or isinstance(sde, subVPSDE):
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
            return -score / std[:, None, None, None]
        return score_fn
    if isinstance(sde, VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                labels = torch.round((sde.T - t) * (sde.N - 1)).long()
            return model_fn(x, labels)
        return score_fn
    raise ValueError(f'Unsupported SDE: {type(sde).__name__}')
