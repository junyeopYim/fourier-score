# Noise-gated Gaussian residuals: equations and gate designs

[Back to README](../../README.md#noise-gated-gaussian-residuals-ve) · [All reports](../README.md) · Provenance: [epoch 0](../provenance.md)

## Noise-gated Gaussian residuals (VE)

An optional fixed gate smoothly connects direct score prediction at low noise
to the Gaussian residual parameterization at high noise:

$$
g(\sigma)=\operatorname{sigmoid}\!\left[p(\log\sigma-\log\sigma_c)\right],
\qquad c_{\sigma,k}^2=(1-g)^2+g(2-g)\frac{P_k}{P_k+\sigma^2},
$$

$$
\sigma s_\theta=g\sigma s_G+\mathcal F^{-1}[c\mathcal Fh_\theta],
\qquad
\tau_{g,k}=\frac{g\sigma\mathcal F(x-\mu)_k-
[P_k+(1-g)\sigma^2]\mathcal F\epsilon_k}{(P_k+\sigma^2)c_{\sigma,k}}.
$$

With `loss.objective=normalized_residual`, training minimizes
$\mathbb E\|h_\theta-\mathcal F^{-1}\tau_g\|^2$.
At $g=0$ this is Score/DSM; at $g=1$ it recovers the original normalized residual
objective. The second-moment identity requires matched population mean and
power, but not Gaussian data. Scalar covariance normalizes the **channel-average**
target second moment; Fourier covariance normalizes each frequency. With estimated
or floored statistics, these are plug-in approximations.

```bash
uv run --locked python train.py -c configs/smoke.json \
  --set loss.objective=normalized_residual \
  --set fourier.gate.mode=log_sigma \
  --set fourier.gate.sigma_switch=1.0 \
  --set fourier.gate.sharpness=4.0 --dry-run
```

`fourier.gate.mode=none` is the default and preserves the original $g=1$ path.
`constant` with `fourier.gate.value=0` or `1` provides endpoint controls.
The gate is part of the model adapter, so training and sampling use the same
settings. Active gates require a VE Gaussian parameterization. They can also
be trained with DSM as a separate ablation. Run names include the gate settings;
resume and inference reject changes to the trained gate. Older configs without
gate fields retain their original behavior and configuration signatures.

The implementation computes $g$ and $1-g$ from opposite sigmoid arguments,
and constructs the target directly from clean data and noise. This avoids
subtracting nearly equal values when the reference dominates at high noise.

An optional `log_sigma_plateau` gate retains this sigmoid below a specified
noise level and reaches **exactly one** at a finite upper boundary:

$$
z=\operatorname{clip}\!\left(\frac{\log\sigma-\log\sigma_L}
{\log\sigma_H-\log\sigma_L},0,1\right),\qquad
w=z^2(3-2z),\qquad g_{\rm plateau}=g+w(1-g).
$$

Both the residual scale and normalized target use this same modified gate.
For $\sigma\leq\sigma_L$, the adapter and target retain the sigmoid calculation;
for $\sigma\geq\sigma_H$, they recover the original $g=1$ parameterization.
This preserves the **equations**, not the trained high-noise predictions of a
different checkpoint: all noise levels still share the newly trained backbone.
There is no teacher, ensemble or added model capacity. Retrain when changing
gate modes; do not substitute the plateau into a trained sigmoid checkpoint.

```bash
uv run --locked python train.py -c configs/smoke.json \
  --set loss.objective=normalized_residual \
  --set fourier.gate.mode=log_sigma_plateau \
  --set fourier.gate.sigma_switch=1.5 \
  --set fourier.gate.sharpness=4.0 \
  --set fourier.gate.sigma_lo=0.8 --set fourier.gate.sigma_hi=1.0 --dry-run
```

The implementation computes $1-g_{\rm plateau}=(1-z)^2(1+2z)(1-g)$ directly
and enforces the endpoints using comparisons of $\sigma$, avoiding subtraction
cancellation near the plateau. Bounds must satisfy $0<\sigma_L<\sigma_H$;
they are included in experiment names and resume signatures for this mode.
The added bounds do not change existing sigmoid-gate configuration signatures.

The `spectral_cap` mode makes the gate depend on both noise and training power.
Starting with the sigmoid complement $q_S=1-g_S$, let $H(\sigma)$ be the same
log-noise smoothstep between `sigma_lo` and `sigma_hi`, and define

$$
q_k=\frac{q_S}{\sqrt{1+H(\sigma)q_S^2\sigma^2/(\delta^2P_k)}},\qquad
g_k=1-q_k,\qquad
c_k^2=q_k^2+(1-q_k^2)\frac{P_k}{P_k+\sigma^2}.
$$

Use $g_k$ in the **Fourier domain** for both the reference and normalized target:

$$
\widehat{u_\theta}_k=g_k\widehat{\sigma s_G}_k+c_k\widehat{h_\theta}_k,
\qquad
\mathcal L=\mathbb E\frac1D\sum_k
\frac{|\widehat{u_\theta+\epsilon}_k|^2}{c_k^2}.
$$

This is one normalized residual MSE with one backbone. No teacher, auxiliary
loss, pretrained initialization or additional learned parameters are used.
The implementation regresses the raw output against the equivalent stable
clean/noise target, as in the other normalized arms. A frequency-dependent
$g_k$ must be applied inside the spectral transform, not as a pixel mask.
The scalar control uses channel-average power and pointwise operations.

When $H=0$, this reduces to the original sigmoid gate. When $H=1$,
$c_k/b_k\leq\sqrt{1+\delta^2}$ for every frequency, where
$b_k^2=P_k/(P_k+\sigma^2)$. With $\delta=0.5$, the residual scale is at most
11.8% above the original Fourier-normalized scale. This bounds the adapter's
scale difference; it does not guarantee a bound on learned score error.

```bash
uv run --locked python train.py -c configs/smoke.json \
  --set loss.objective=normalized_residual \
  --set fourier.gate.mode=spectral_cap \
  --set fourier.gate.sigma_switch=1.5 --set fourier.gate.sharpness=4.0 \
  --set fourier.gate.sigma_lo=1.0 --set fourier.gate.sigma_hi=2.0 \
  --set fourier.gate.delta=0.5 --dry-run
```

`delta` must be finite and positive. It participates in names and resume
signatures for `spectral_cap`; adding its default preserves older gate
configuration signatures. See the [GMM spectral gate experiment](../gmm/spectral-gate.md)
for the fixed-protocol comparison with sigmoid and plateau gates.

Two further modes replace the gate itself with a function of **linear noise
$\sigma$**, using the same normalized target, residual scale and single MSE:

$$
g_{\rm linear}(\sigma)=\operatorname{clip}\!\left[
\frac12+\frac p4\left(\frac\sigma{\sigma_c}-1\right),0,1\right],
\qquad
g_{\rm tanh}(\sigma)=\frac12\left[1+\tanh\!\left(
\frac p2\left(\frac\sigma{\sigma_c}-1\right)\right)\right].
$$

With $\sigma_c=1.5$ and $p=4$, both gates match the original log-sigma sigmoid's
center $g(1.5)=0.5$ and local slope $g'(1.5)=2/3$. The linear gate is exactly
zero below $\sigma=0.75$ and exactly one above $2.25$; tanh approaches one
smoothly. Their low-noise shapes are different. Neither changes the backbone
or introduces an extra loss.

A tanh of **log noise** would give
$[1+\tanh((p/2)\log(\sigma/\sigma_c))]/2
=\operatorname{sigmoid}(p\log(\sigma/\sigma_c))$, exactly the existing gate.
The new `tanh_sigma` mode therefore uses a sigma-linear argument. Internally,
opposite sigmoid arguments evaluate the same tanh formula and its complement
without cancellation near one. `linear_sigma` computes both clipped ramps
directly. Gate modes, centers and sharpness values enter checkpoint signatures.

```bash
uv run --locked python train.py -c configs/smoke.json \
  --set loss.objective=normalized_residual \
  --set fourier.gate.mode=linear_sigma \
  --set fourier.gate.sigma_switch=1.5 --set fourier.gate.sharpness=4.0 --dry-run
```

Use `fourier.gate.mode=tanh_sigma` for the tanh variant. See the
[paired GMM experiment](../gmm/linear-tanh-gates.md) for their comparison.
Those archived results concern sigma-coordinate schedules. A line that appears
straight on a **log-sigma plot** requires the distinct design below.

## Log-axis gate design

The `linear_log_sigma` gate spans the whole noise interval and is straight on
the logarithmic horizontal axis:

$$
g_L(\sigma)=\operatorname{clip}\!\left[
\frac{\log(\sigma/0.1)}{\log(3/0.1)},0,1\right].
$$

For an S-shaped gate that stays near zero at low noise and joins an exact
upper plateau smoothly, use `bounded_log_sigmoid`. Define

$$
z=\operatorname{clip}\!\left[
\frac{\log(\sigma/\sigma_L)}{\log(\sigma_H/\sigma_L)},0,1\right],
\qquad z_c=\frac{\log(\sigma_c/\sigma_L)}{\log(\sigma_H/\sigma_L)},
$$

$$
g_S(\sigma)=\frac{[z(1-z_c)]^\kappa}
{[z(1-z_c)]^\kappa+[(1-z)z_c]^\kappa},
\qquad (\sigma_L,\sigma_c,\sigma_H,\kappa)=(0.1,0.75,1.5,2).
$$

This gives **exactly zero at $\sigma\leq0.1$, one-half at $\sigma=0.75$,
and exactly one at $\sigma\geq1.5$**. Both plateau joins have zero slope for
$\kappa>1$. Inside the interval, the same gate can be written as

$$
g_S=\operatorname{sigmoid}\!\left[\kappa(\operatorname{logit}z-\operatorname{logit}z_c)\right]
=\frac{1+\tanh[\tfrac\kappa2(\operatorname{logit}z-\operatorname{logit}z_c)]}{2}.
$$

Thus sigmoid and tanh are equivalent expressions for this one S curve.
It uses the logit of normalized log noise so that finite endpoints can be
reached smoothly; it is distinct from the earlier plain log-sigma sigmoid.
The implementation uses positive powers scaled by their largest base, avoiding
endpoint logarithms and simultaneous underflow for steep curves.

![A black log-linear ramp and a gold bounded sigmoid with smooth plateaus.](../../assets/log_gate_design/log_gate_design.svg)

`sigma_lo` and `sigma_hi` control endpoints. For the S curve, `sigma_switch`
sets the exact half-height noise and `sharpness` is $\kappa>1$; increasing it
keeps the gate closer to zero/one for longer. The log-linear mode depends only
on its bounds. These settings enter the appropriate checkpoint signatures.
Both designs use the existing $c_k^2=(1-g)^2+g(2-g)P_k/(P_k+\sigma^2)$,
the same Gaussian reference and normalized target, **one backbone and one
normalized MSE, with no teacher**. Exact endpoints recover the Score/DSM and
Fourier-normalized equations, respectively; this does not guarantee the
predictions or score error of separately trained checkpoints.

```bash
uv run --locked python train.py -c configs/smoke.json \
  --set loss.objective=normalized_residual \
  --set fourier.gate.mode=linear_log_sigma \
  --set fourier.gate.sigma_lo=0.1 --set fourier.gate.sigma_hi=3.0 --dry-run

uv run --locked python train.py -c configs/smoke.json \
  --set loss.objective=normalized_residual \
  --set fourier.gate.mode=bounded_log_sigmoid \
  --set fourier.gate.sigma_lo=0.1 --set fourier.gate.sigma_hi=1.5 \
  --set fourier.gate.sigma_switch=0.75 --set fourier.gate.sharpness=2.0 --dry-run
```

The [GMM log-axis experiment](../gmm/log-axis-gates.md) evaluates these exact
settings. The earlier sigma-coordinate Linear/Tanh results concern different
shapes. Regenerate the original design figure with
`uv run --locked --extra figures python scripts/plot_log_gate_design.py`.
The script also accepts endpoint, center and sharpness overrides and exports
[sample values and settings](../../assets/log_gate_design/design.json),
[curve data](../../assets/log_gate_design/curves.csv),
[PNG](../../assets/log_gate_design/log_gate_design.png) and
[PDF](../../assets/log_gate_design/log_gate_design.pdf).
The [verification record](../../assets/log_gate_design/verification.json) covers loss,
gradient, endpoint, checkpoint and sampling checks.

Run the controlled GMM comparison with:

```bash
uv run --locked --extra figures python scripts/run_gmm_comparison.py \
  --output saved/gmm_gated_example --workers 3
```

This trains five existing baselines and six gate candidates: Scalar/Fourier
with $\sigma_c\in\{0.5,1,1.5\}$ and $p=4$, at three spectra and three paired
seeds, for **99 runs of 5,000 updates**. One shared switch is chosen by the mean
final EMA **validation** error across both covariances, all spectra and seeds.
Only then are the five baselines and two selected gated arms evaluated on a
new, shared test bank (63 evaluations). The runner records the selection before
constructing the test bank. It uses the notebook's shared implementation in
[`fourier_score/gmm.py`](../../fourier_score/gmm.py), without rewriting notebook cells.

Use `--preset smoke --seeds 42` for a short pipeline check. `--bank-version`
controls independent evaluation banks; changing only the output directory does
not change them. Matching interrupted runs resume from saved optimizer, EMA and
data RNG states. Tables and PNG/SVG/PDF figures are saved in `<output>/report`,
or a directory supplied with `--report`.
`--workers` schedules individual runs, each using one CPU thread; use, for
example, `--workers 12` to occupy more cores without changing numerical training
settings. The worker count can change on resume. Launcher invocations are
recorded in `execution_history.json`, while numerical source and protocol
checks continue to guard checkpoint compatibility.
