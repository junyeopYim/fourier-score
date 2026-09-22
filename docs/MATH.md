# Gaussian references and residual score parameterizations

Implementation: [method.py](../fourier_score/method.py),
[statistics.py](../fourier_score/statistics.py), and
[loss.py](../fourier_score/loss.py). The
[GMM notebook](../notebooks/gmm_fourier_residual.ipynb) checks the population-moment
identities and evaluates against an exact joint score.

## Fixed reference and shared backbone

For the forward marginal, write

$$
y_t=\alpha_t x+\sigma_t\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
$$

with noise independent of the clean data. The primary pixel-space experiment
uses VE noise, so $\alpha_t=1$. The same conditional Gaussian algebra also
supports the explicitly configured DDPM and latent experiments.

The training-data mean $\mu$ is a full image of shape `[C,H,W]`. The spectrum
$P_{c,k}$ is the per-channel centered second moment under an orthonormal spatial
FFT $F$:

$$
P_{c,k}=\mathbb E\left[\left|F(x_c-\mu_c)_k\right|^2\right].
$$

For each channel, the Gaussian reference uses the covariance
$C_c=F^{-1}\operatorname{diag}(P_c)F$. It retains frequency-dependent power, but
not cross-channel or off-diagonal cross-frequency covariance. Matching these
moments does not assume that the data distribution is jointly Gaussian. In
practice the moments are estimated from training data, conjugate-symmetrized,
and power-floored for numerical stability.

Set

$$
V_{t,c,k}=\alpha_t^2P_{c,k},\qquad
D_{t,c,k}=V_{t,c,k}+\sigma_t^2,\qquad
b_{t,c,k}=\sqrt{\frac{V_{t,c,k}}{D_{t,c,k}}}.
$$

The analytic reference score and learned total score are

$$
s_G(y_t,t)=-F^{-1}\left[
\frac{F(y_t-\alpha_t\mu)}{D_t}\right],
$$

$$
\sigma_t s_\theta(y_t,t)
=\sigma_t s_G(y_t,t)+F^{-1}\left[b_t Fh_\theta(y_t,t)\right].
$$

All spectral operations are per channel, with elementwise division and
multiplication. The full orthonormal FFT includes both conjugate partners of
real data; the denominator has $\sigma_t^2$, with no extra factor of one half.
The reference buffers have no learned parameters.

The vendored NCSN++ architecture is shared across parameterizations. Its final
`scale_by_sigma` operation is disabled for every method and applied exactly
once by the common score adapter. Learned layers, shapes, module order,
initialization, time conditioning, and input preprocessing do not depend on
`loss.type`. A regression test compares the VE score baseline with the upstream
`scale_by_sigma=True` output using the same state dictionary.

## Compared parameterizations

| Configuration | Interpreted output |
|:--|:--|
| `score` | $\sigma_t s_\theta=h_\theta$ |
| `scalar_gaussian` | Gaussian reference and residual scaling with constant per-channel power $v_c$ |
| `fourier_gaussian` | Gaussian reference and residual scaling with frequency-dependent power $P_{c,k}$ |
| `diffusion` | Optional noise-prediction convention: $\epsilon_\theta=h_\theta$, $\sigma_t s_\theta=-h_\theta$ |

Scalar Gaussian retains the **same full mean image** and replaces power in
both the reference and residual multiplier by

$$
v_c=\frac{1}{HW}\sum_k P_{c,k}.
$$

The corresponding spectral multipliers are spatially constant, so pixelwise
arithmetic implements the same operator without an FFT. Both Gaussian methods
use the same training-statistics cache; Scalar transforms a private copy.
This control is not the complete EDM input/noise/loss/sampling recipe.

Score and diffusion are sign conventions for the same forward process:
$h_{\mathrm{diffusion}}=-h_{\mathrm{score}}$ gives identical total scores and
losses. Equal raw outputs instead give opposite initial scores. A DDPM-versus-VE
comparison also changes the forward distribution and sampler and must be
reported separately. Pixel-space DDPM presets retain NCSN++; they do not
reproduce the original DDPM U-Net.

## Why the residual scale appears

Consider a centered Fourier coordinate
$U_k=\alpha_t F(x-\mu)_k$, with second moment $V_k=\alpha_t^2P_k$, and
$Y_k=U_k+\sigma_t\varepsilon_k$. Orthonormal transformation gives
$\mathbb E|\varepsilon_k|^2=1$. Suppress the time and channel indices below.
The scaled DSM target after subtracting the reference score is

$$
T_k=-\varepsilon_k+\frac{\sigma Y_k}{V_k+\sigma^2}
=\frac{\sigma U_k-V_k\varepsilon_k}{V_k+\sigma^2}.
$$

Independence of clean data and noise gives

$$
\mathbb E|T_k|^2
=\frac{\sigma^2V_k+V_k^2}{(V_k+\sigma^2)^2}
=\frac{V_k}{V_k+\sigma^2}=b_k^2.
$$

This second-moment identity does not require Gaussian data, independence
between Fourier modes, or a Gaussian joint distribution. Where $V_k>0$,
$T_k/b_k$ has unit second moment. Empirical estimation and power flooring make
this an approximate population normalization in image experiments.

Let $s_* = \nabla_y\log p_t(y)$ be the true **joint** noisy score. The DSM
conditional-expectation identity yields

$$
\sigma_t(s_*(y_t,t)-s_G(y_t,t))
=\mathbb E[-\epsilon-\sigma_t s_G(y_t,t)\mid y_t].
$$

Under the matched-moment assumptions above, conditional Jensen gives

$$
\mathbb E\left[\left|F\{\sigma_t(s_*-s_G)\}_k\right|^2\right]
\leq b_{t,k}^2.
$$

The learned residual's second moment therefore need not be one after scaling.
For exact Gaussian data the optimal residual is zero, while the individual
noisy regression target still has nonzero variance. The neural network observes
the entire noisy image; no marginal Fourier score is substituted for the joint
score.

For an assumed coordinate variance $Q>0$ and actual variance $V>0$, the
reference-subtracted noisy target instead has second moment

$$
R(Q)=\frac{\sigma^2V+Q^2}{(Q+\sigma^2)^2},\qquad
R(Q)-R(V)=\frac{\sigma^2(Q-V)^2}{(Q+\sigma^2)^2(V+\sigma^2)}\geq0.
$$

This is a target-moment identity, not an optimization or sample-quality bound.
In particular, Scalar's per-channel average generally does not match every
individual frequency. Its scaling is exact per coordinate only for a flat
spectrum; averaged over frequencies in a channel, the target second moment
matches the scalar scale when that channel's average variance is exact.

## Preserve the DSM objective

The pixel-space training objective is

$$
\mathcal L(\theta)=\mathbb E_{t,x,\epsilon}\left[
\operatorname{reduce}_{\mathrm{pixels}}
\left|\sigma_t s_\theta(y_t,t)+\epsilon\right|^2\right].
$$

The configured reduction is `mean`, or `half_sum` (one half of the pixel sum),
followed by averaging over images. VE draws time uniformly in `[t_min,1]`,
which is uniform in log noise scale; DDPM draws a uniform discrete step.
Pixel-model validation always reports pixel-mean DSM for comparison.

With the Fourier parameterization, Parseval gives a per-frequency error
$|b_k(Fh_\theta)_k-T_k|^2$. Dividing the regression target by $b_k$ does not
remove its $b_k^2$ weighting from the original loss. An unweighted MSE on
$T_k/b_k$ would be a different objective.

The Gaussian reference enters with coefficient one. This experiment keeps
input preprocessing and the sampling interface shared. It does not introduce
a learned reference gate, per-frequency independent networks, input whitening,
or a special Gaussian-residual sampling split.

For the latent backend, [model.py](../fourier_score/ldm/model.py) converts the
same total scaled score into an epsilon prediction,
$\epsilon_\theta=-\sigma_t s_\theta$. The scalar/Fourier Churches comparison
uses the configured L2 epsilon loss. The upstream L1 recipe and the L2
comparison are distinct protocols; preserving the backbone and iteration
budget does not make them identical. See [LDM.md](LDM.md).

## Matched-moment GMM mechanism test

For $d$ spatial coordinates, choose a fixed real orthogonal basis $q_j$,
$C=F^{-1}\operatorname{diag}(P)F$, and $L=C^{1/2}$. For $0\leq\rho<1$, let

$$
m_{j,\pm}=\pm\rho\sqrt d\,Lq_j,\qquad
\Sigma_{\mathrm{within}}=(1-\rho^2)C.
$$

The equal-weight mixture over $2d$ components has mean zero and covariance
$C$, exactly matching $\mathcal N(0,C)$. Both use the same population Gaussian
reference. The mixture's noisy density and score are available in closed form:

$$
p_t(y)=\sum_j\pi_j\mathcal N(y;\alpha_t m_j,C_{j,t}),\qquad
C_{j,t}=\alpha_t^2\Sigma_j+\sigma_t^2I,
$$

$$
\gamma_j(y,t)=\frac{\pi_j\mathcal N(y;\alpha_t m_j,C_{j,t})}{p_t(y)},\qquad
s_*(y,t)=-\sum_j\gamma_j(y,t)C_{j,t}^{-1}(y-\alpha_t m_j).
$$

The [notebook](../notebooks/gmm_fourier_residual.ipynb) uses this oracle only
for validation/evaluation and trains with fresh samples under the shared DSM
loss. Its primary held-out metric is

$$
E_\theta=\mathbb E_{t,y}\left[
\frac{\sigma_t^2}{d}\|s_\theta(y,t)-s_*(y,t)\|^2\right].
$$

A learned total score improving on the Gaussian reference would demonstrate
structure beyond its first two moments. It would not, by itself, establish a
Fourier advantage over Scalar or improved FID. These are separate questions.

On the default 8×8 pilot/full grid, spectral heterogeneity varies as
$P_k(\lambda)=(1-\lambda)\bar P+\lambda P_k^{\mathrm{structured}}$, with
matched average power. At $\lambda=0$, Scalar and Fourier are numerically
equivalent; differences for larger $\lambda$ are measured without assuming
their direction. The shorter smoke run uses a 4×4 grid.

## Sampling and diagnostics

Every sampler receives the same converted total score used in training. The
terminal prior is shared across methods. PC uses a reverse VE Euler predictor
with increments in $\sigma^2$ and a batch-mean Langevin corrector. Heun
integrates the VE probability-flow ODE

$$
\frac{dx}{d\sigma}=-\sigma s_\theta(x,\sigma),
$$

with optional terminal Tweedie denoising. DDPM uses fixed-small posterior
variance and the model's clean-image estimate, with optional clipping and the
full trained step grid.

Optional `evaluation.frequency_bins` partitions the full orthonormal FFT of
the common noisy DSM residual $r=\sigma s_\theta+\epsilon$ into equal radial
bands from zero to $\sqrt{1/2}$ cycles/pixel, including DC and conjugate
partners. Mode-count-weighted band means reproduce pixel-mean DSM by Parseval,
both globally and within each noise bin. Empty bands/bins are null. CPU
float64 FFT is used for these diagnostics, including on MPS hosts.

Those image-model diagnostics measure the **noisy DSM residual**, not exact
true-score error or image quality. The GMM notebook separately has access to
the exact score. Observation-level standard errors also differ from variation
across independent training seeds.

Image statistics use training data only. When horizontal flipping is enabled,
both each image and its mirror contribute, matching the empirical 50/50 flip
mixture in its first and second moments. Evaluation has no random augmentation.
Dedicated CPU generators isolate time/noise draws from spectral kernels;
dropout and floating-point arithmetic can still differ across hardware or
microbatch settings.

## Scope and primary sources

Fourier asymptotic Gaussianity motivates the choice of reference. It is not
used as a proof of whole-image Gaussianity, score convergence, FID improvement,
or integrability of the unconstrained neural score field. Gaussian/EDM
preconditioning provides related algebra; this repository does not assert
novelty of that algebra or of using a GMM as a diagnostic.

- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- [Peligrad and Wu: Central limit theorem for Fourier transforms of stationary processes](https://arxiv.org/abs/0910.3451).
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).

New statistical or performance claims require their own assumptions and
experiments. CIFAR-10 and Churches results cannot be inferred from the toy
mechanism experiment.
