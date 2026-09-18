# Copyright 2020 The Google Research Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.
# Adapted from losses.py; loss equations and reductions are unchanged.
import torch
from model.model import get_model_fn, get_score_fn
from sde.sde_lib import VESDE, VPSDE


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    def loss_fn(model, batch):
        score_fn = get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        score = score_fn(mean + std[:, None, None, None] * z, t)
        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        return torch.mean(losses)
    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    assert isinstance(vesde, VESDE)
    sigmas_descending = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    def loss_fn(model, batch):
        model_fn = get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = sigmas_descending.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        score = model_fn(noise + batch, labels)
        target = -noise / (sigmas ** 2)[:, None, None, None]
        losses = torch.square(score - target)
        return (reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2).mean()
    return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    assert isinstance(vpsde, VPSDE)
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    def loss_fn(model, batch):
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        a = vpsde.sqrt_alphas_cumprod.to(batch.device)
        b = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed = a[labels, None, None, None] * batch + b[labels, None, None, None] * noise
        errors = (get_model_fn(model, train)(perturbed, labels) - noise).square()
        return reduce_op(errors.reshape(errors.shape[0], -1), dim=-1).mean()
    return loss_fn


def build_loss(cfg, sde, train=True):
    if cfg.training.continuous:
        return get_sde_loss_fn(sde, train, cfg.training.reduce_mean, True, cfg.training.likelihood_weighting)
    if cfg.training.likelihood_weighting:
        raise ValueError('Discrete SMLD/DDPM does not support likelihood weighting')
    if isinstance(sde, VESDE):
        return get_smld_loss_fn(sde, train, cfg.training.reduce_mean)
    if isinstance(sde, VPSDE):
        return get_ddpm_loss_fn(sde, train, cfg.training.reduce_mean)
    raise ValueError('Discrete subVP is not supported')
