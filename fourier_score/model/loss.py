"""DSM and direct normalized-residual training objectives.

score: raw = sigma*s; diffusion: raw = epsilon_hat; Fourier Gaussian:
sigma*s = sigma*s_G + F^-1[b*F(raw)]. Loss changes NO backbone parameters.
scalar_gaussian replaces P_k by its per-channel frequency mean.
normalized_residual regresses raw against the normalized VE target, while
noise_residual retains the common DSM evaluation metric.
"""


def noise_residual(model, clean, level, noise):
    y = model.process.perturb(clean, level, noise)
    return model.scaled_score(y, level) + noise


def training_loss(model, clean, level, noise, reduction="mean", *, objective="dsm"):
    if objective == "dsm":
        residual = noise_residual(model, clean, level, noise)
    elif objective == "normalized_residual":
        if model.process.kind != "ve" or model.reference is None:
            raise ValueError("normalized_residual requires a VE Gaussian parameterization")
        y = model.process.perturb(clean, level, noise)
        raw = model.backbone(y, model.process.condition(level, model.embedding))
        target = model.reference.normalized_target(clean, noise, level.sigma)
        residual = raw - target
    else:
        raise ValueError(f"Unknown loss objective: {objective}")
    errors = residual.square().flatten(1)
    values = errors.mean(1) if reduction == "mean" else 0.5 * errors.sum(1)
    return values.mean()
