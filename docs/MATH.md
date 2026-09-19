# Objective definitions and comparison boundaries

## Backbone convention

The vendored NCSN++ code is unchanged. Its final deterministic `scale_by_sigma`
operation is disabled **for all arms**, then performed exactly once in the
shared score adapter. All learned layers, shapes, module order, initialization,
time conditioning and input preprocessing are independent of `loss.type`.
For the score baseline under VE, a regression test verifies exact agreement
with the original NCSN++ `scale_by_sigma=True` output using the same state dict.

Let h_theta(y,t) be the raw output and y=alpha_t x + sigma_t epsilon.

- score: s_theta = h_theta / sigma_t.
- diffusion: epsilon_theta = h_theta, s_theta = -epsilon_theta/sigma_t.
- Fourier Gaussian:
  sigma_t s_theta = sigma_t s_G + F^{-1}[b_t F h_theta],
  s_G = -F^{-1}[F(y-alpha_t mu)/(alpha_t^2 P + sigma_t^2)],
  b_t = sqrt(alpha_t^2 P/(alpha_t^2 P+sigma_t^2)).

The primary proposal is VE (alpha=1). DDPM support is an explicitly documented
extension of the same Gaussian conditional algebra, not a claim that the
original proposal was evaluated on DDPM.

The empirical mean mu is a full [C,H,W] image. P is a full per-channel,
conjugate-symmetric orthonormal-FFT spectrum. Cross-channel and cross-frequency
covariance are NOT learned or estimated here. P is floored for stability.
Its buffers contain no learned parameters. The full FFT automatically accounts
for real-valued conjugate pairs; there is no factor 1/2 in P+sigma^2.

## Why the frequencywise residual scale appears

For a centered Fourier coordinate U=alpha X with variance V=alpha^2 P and
Y=U+sigma epsilon, the scaled residual target after subtracting the Gaussian
score is

    T = -epsilon + sigma Y/(V+sigma^2)
      = (sigma U - V epsilon)/(V+sigma^2).

With independent unit-variance epsilon and matching second moments,
E|T|^2 = V/(V+sigma^2) = b^2. Gaussianity of X is unnecessary for this
second-moment identity. Thus T/b has unit second moment (where V>0).
Estimated/floored P makes this an approximate normalization for real data.

Conditional expectation yields
sigma(s_* - s_G) = E[T | entire noisy image].
By Jensen, E|sigma(s_* - s_G)_k|^2 <= b_k^2.
The neural network has access to the ENTIRE noisy image; this does not assume
that Fourier coordinates are independent or that their marginal scores equal
the joint score. Fourier CLT is a motivation, not a proof of whole-image
Gaussianity, score convergence, FID improvement, or integrability of the
unconstrained neural vector field.

## Loss and evaluation

All arms use

    L = E || sigma s_theta(y,t) + epsilon ||^2,

with either `mean` over pixels or `half_sum` = (1/2) sum over pixels, then
average over images. Time is uniform in [t_min,1] for VE, so log sigma is
uniform; DDPM draws a uniform discrete step. The reduction is fixed by config.
All validation/test results report canonical PIXEL-MEAN DSM, regardless of
training reduction. Fourier output normalization does not remove b^2 from
this objective. Training plain unweighted MSE on T/b would be a DIFFERENT
frequency-weighted objective and is not implemented.

The Gaussian reference is always coefficient 1; there is no scalar MMSE gate,
linear gate, learned gate, reference-only bypass, or per-frequency independent
network. Input whitening and analytic Gaussian splitting of the sampler are
intentionally not included, to isolate the output-parameterization experiment.

## Score versus diffusion

For the same forward process, h_epsilon=-h_score gives exactly the same score
and per-image loss. These are two output conventions, not independent methods.
Equal initial raw h values generally produce opposite signs of the initial
score; the initial *backbone weights* are identical, not the interpreted score.
A DDPM-versus-VE comparison changes the forward distribution and sampler and
must be labeled separately from a loss-only ablation. The DDPM presets retain
the NCSN++ architecture; they are not a reproduction of the original DDPM U-Net.

## Common sampler

PC uses a reverse VE Euler predictor with delta sigma^2 and a batch-mean
Langevin corrector. Heun integrates the VE probability-flow ODE in sigma:
dx/dsigma = -sigma*s_theta, with optional terminal Tweedie denoising.
DDPM uses the standard fixed-small posterior variance and the model's x0
estimate, with optional x0 clipping. It requires all trained discrete steps.
Every sampler obtains the SAME converted total score used in training.
The terminal prior remains the same across objectives; no learned or
Fourier-specific terminal distribution is silently substituted.

Gaussian statistics are estimated from training images only. If random_flip is
enabled, both each image and its mirror contribute, which exactly matches the
50/50 flip mixture in its first and second moments. Evaluation receives no
random augmentation. Dedicated CPU generators isolate t/noise draws from
backend implementations; dropout / floating point can still differ across
hardware or microbatch settings.

## Primary-source context

- Score-SDE: https://arxiv.org/abs/2011.13456
- DDPM: https://arxiv.org/abs/2006.11239
- Peligrad–Wu Fourier CLT: https://arxiv.org/abs/0910.3451
- EDM and related preconditioning: https://arxiv.org/abs/2206.00364

The normalization is closely related to Gaussian/EDM preconditioning. This
project does not assert novelty of that algebra or a theorem about generative
quality. New statistical claims require their own assumptions and experiments.
