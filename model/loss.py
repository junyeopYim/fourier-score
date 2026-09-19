"""Shared sigma^2-weighted DSM, with explicit parameterization conventions.

score: raw = sigma*s; diffusion: raw = epsilon_hat; Fourier Gaussian:
sigma*s = sigma*s_G + F^-1[b*F(raw)]. Loss changes NO backbone parameters.
"""
import torch

def noise_residual(model,clean,level,noise):
    y=model.process.perturb(clean,level,noise)
    return model.scaled_score(y,level)+noise


def per_image_dsm(model,clean,level,noise):
    return noise_residual(model,clean,level,noise).square().flatten(1).mean(1)


def training_loss(model,clean,level,noise,reduction='mean'):
    errors=noise_residual(model,clean,level,noise).square().flatten(1)
    values=errors.mean(1) if reduction=='mean' else 0.5*errors.sum(1)
    return values.mean()
