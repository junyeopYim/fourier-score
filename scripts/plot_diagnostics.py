"""Original, deterministic oracle illustrations; no training or dataset download.

Run ``python scripts/plot_diagnostics.py`` from any directory. See README.md#figures for the mathematical scope and the distinction
between these illustrations and empirical results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"data": "#2667A1", "reference": "#CF7B24", "residual": "#087F73"}


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#718096", "text.color": "#172636",
        "axes.labelcolor": "#172636", "svg.hashsalt": "fourier-score-oracles-v1",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig, output: Path, stem: str):
    for extension in ("png", "svg", "pdf"):
        metadata = {"Creator": "Fourier Score scientific figure scripts"}
        if extension == "png":
            metadata = {"Software": "Fourier Score scientific figure scripts"}
        elif extension == "svg":
            metadata["Date"] = None
        else:
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(output / f"{stem}.{extension}", dpi=180, metadata=metadata,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def mixture_density_score(y, noise_variance, rho=0.92):
    """Density and score of 0.5 N(-rho,v) + 0.5 N(rho,v), v=1-rho²+σ²."""
    y = np.asarray(y)
    variance = 1 - rho**2 + noise_variance
    log_components = -0.5 * (y[..., None] - np.array([-rho, rho]))**2 / np.asarray(variance)[..., None]
    offset = log_components.max(axis=-1, keepdims=True)
    exp_components = np.exp(log_components - offset)
    weights = exp_components / exp_components.sum(axis=-1, keepdims=True)
    score = -(y - np.sum(weights * np.array([-rho, rho]), axis=-1)) / variance
    log_density = (offset[..., 0] + np.log(exp_components.sum(axis=-1))
                   - np.log(2) - 0.5 * np.log(2 * np.pi * variance))
    return np.exp(log_density), score


def make_process_figure(output: Path, rng, data: dict, rho: float):
    # VE diffusion with sigma²(t)=4t; dX=2 dW. Reverse paths start at the exact p_T.
    grid = np.linspace(-6, 6, 501)
    times = np.linspace(0, 1, 1001)
    dt, g2 = times[1] - times[0], 4.0
    density_times = np.linspace(0, 1, 301)
    density, _ = mixture_density_score(grid[:, None], g2 * density_times, rho)
    n_paths = 7
    component_signs = np.array([-1, 1, -1, 1, -1, 1, 1])
    initial = rho * component_signs + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
    forward = np.empty((len(times), n_paths))
    forward[0] = initial
    forward[1:] = initial + np.cumsum(np.sqrt(g2 * dt) * rng.standard_normal((len(times)-1, n_paths)), axis=0)
    reverse = np.empty_like(forward)
    reverse[-1] = rho * rng.choice([-1, 1], n_paths) + np.sqrt(1 - rho**2 + g2) * rng.standard_normal(n_paths)
    for j in range(len(times)-1, 0, -1):
        _, score = mixture_density_score(reverse[j], g2 * times[j], rho)
        reverse[j-1] = reverse[j] + g2 * score * dt + np.sqrt(g2 * dt) * rng.standard_normal(n_paths)

    # Probability-flow ODE, integrated with Heun in physical time.
    ode = np.empty((len(times), 5))
    ode[0] = [-1.45, -0.85, 0.0, 0.85, 1.45]
    for j in range(len(times)-1):
        drift = -0.5 * g2 * mixture_density_score(ode[j], g2 * times[j], rho)[1]
        proposal = ode[j] + dt * drift
        next_drift = -0.5 * g2 * mixture_density_score(proposal, g2 * times[j+1], rho)[1]
        ode[j+1] = ode[j] + 0.5 * dt * (drift + next_drift)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True)
    path_colors = ["#fdae61", "#f46d43", "#f9ce62", "#e86ea5", "#9fd1b5", "#80b1d3", "#d4a6dd"]
    for ax, paths, title, direction in zip(
        axes, [forward, reverse],
        ["Forward diffusion: data to noise", "Reverse diffusion: noise to data"], [1, -1]
    ):
        ax.pcolormesh(density_times, grid, density, shading="auto", cmap="cividis", rasterized=True)
        for i in range(n_paths):
            ax.plot(times, paths[:, i], color=path_colors[i], lw=0.95, alpha=0.85)
        ax.plot(times, ode, color="white", ls="--", lw=1.25, alpha=0.95)
        ax.set(xlabel="Diffusion time $t$", xlim=(0, 1) if direction == 1 else (1, 0),
               ylim=(-5.4, 5.4), title=title)
        ax.text(0.03, 0.95, r"$\sigma^2(t)=4t$", color="white", transform=ax.transAxes, va="top")
    axes[0].set_ylabel("State $x$")
    axes[1].legend(handles=[Line2D([0], [0], color="#fdae61", lw=1.6, label="SDE sample path"),
                           Line2D([0], [0], color="white", ls="--", label="Probability-flow ODE")],
                   loc="lower right", facecolor="#273b50", labelcolor="white", framealpha=0.95, fontsize=9)
    fig.suptitle("Exact GMM marginals and oracle-score dynamics", fontsize=15, y=1.02)
    fig.text(0.5, -0.03, "Background: analytic density. Reverse SDE starts at the exact finite-time marginal; paths use numerical integration.",
             ha="center", fontsize=9, color="#526476")
    fig.tight_layout()
    save_figure(fig, output, "stochastic_process")
    data.update(process_grid=grid, process_density_times=density_times, process_density=density,
                process_times=times, process_forward_paths=forward, process_reverse_paths=reverse,
                process_ode_paths=ode)


def make_residual_figure(output: Path, data: dict, rho: float):
    sigma = 0.35
    grid = np.linspace(-3.7, 3.7, 1001)
    density, score = mixture_density_score(grid, sigma**2, rho)
    reference_density = np.exp(-grid**2 / (2 * (1 + sigma**2))) / np.sqrt(2 * np.pi * (1 + sigma**2))
    reference_score = -grid / (1 + sigma**2)
    residual = sigma * (score - reference_score)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    axes[0].plot(grid, density, color=COLORS["data"], lw=2.3, label="Noisy GMM $p_t$")
    axes[0].plot(grid, reference_density, color=COLORS["reference"], lw=2, ls="--", label="Gaussian reference $q_t$")
    axes[0].fill_between(grid, density, color=COLORS["data"], alpha=0.08)
    axes[0].set(ylabel="Probability density", title="Same mean and variance", ylim=(0, 0.49))
    axes[0].legend(fontsize=9, loc="upper right")
    axes[1].plot(grid, sigma * score, color=COLORS["data"], lw=2.3, label=r"$\sigma_t s_*(y,t)$")
    axes[1].plot(grid, sigma * reference_score, color=COLORS["reference"], lw=2, ls="--", label=r"$\sigma_t s_G(y,t)$")
    axes[1].set(ylabel="Scaled score", title="Different exact scores")
    axes[1].legend(fontsize=10)
    axes[2].plot(grid, residual, color=COLORS["residual"], lw=2.3)
    axes[2].axhline(0, color="#718096", ls=":", lw=1)
    axes[2].fill_between(grid, residual, color=COLORS["residual"], alpha=0.10)
    axes[2].set(ylabel=r"$\sigma_t(s_*-s_G)$", title="Oracle residual remains")
    for ax in axes:
        ax.set(xlabel="Noisy observation $y$", xlim=(-3.7, 3.7))
        ax.grid(alpha=0.15)
    fig.suptitle(r"Matched population moments do not determine the score", fontsize=15, y=1.03)
    fig.text(0.5, -0.025, r"Clean GMM: $\frac{1}{2}\mathcal{N}(-\rho,1-\rho^2)+\frac{1}{2}\mathcal{N}(\rho,1-\rho^2)$; reference: $\mathcal{N}(0,1)$; "
             + rf"$\rho={rho}$, $\sigma_t={sigma}$. All curves are analytic; no network is trained.",
             ha="center", fontsize=9, color="#526476")
    fig.tight_layout()
    save_figure(fig, output, "gaussian_residual")
    data.update(residual_grid=grid, gmm_density=density, reference_density=reference_density,
                true_scaled_score=sigma*score, reference_scaled_score=sigma*reference_score,
                oracle_scaled_residual=residual)


def fft_filter(x, multiplier):
    return np.fft.ifft2(np.fft.fft2(x, norm="ortho") * multiplier, norm="ortho").real


def make_frequency_figure(output: Path, rng, data: dict, checks: dict, n_samples: int):
    size, rho, sigma = 8, 0.85, 0.7
    dimension = size**2
    frequency = np.fft.fftfreq(size)
    radius2 = frequency[:, None]**2 + frequency[None, :]**2
    power = (1 + radius2 / 0.15**2)**-1.5
    power /= power.mean()
    flat = np.full_like(power, power.mean())
    orthogonal, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    means = np.concatenate([orthogonal, -orthogonal]).reshape(2*dimension, size, size) * rho * np.sqrt(dimension)
    means = fft_filter(means, np.sqrt(power))
    basis = np.eye(dimension).reshape(dimension, size, size)
    covariance = fft_filter(basis, power).reshape(dimension, dimension)
    mixture_covariance = means.reshape(2*dimension, dimension).T @ means.reshape(2*dimension, dimension) / (2*dimension) + (1-rho**2)*covariance
    checks["gmm_mean_max_abs"] = float(np.abs(means.mean(0)).max())
    checks["matched_covariance_max_abs"] = float(np.abs(mixture_covariance-covariance).max())
    assert checks["gmm_mean_max_abs"] < 1e-12
    assert checks["matched_covariance_max_abs"] < 1e-12
    checks["flat_scalar_fourier_scale_max_abs"] = float(np.max(np.abs(np.sqrt(flat/(flat+sigma**2)) - np.sqrt(power.mean()/(power.mean()+sigma**2)))))
    assert checks["flat_scalar_fourier_scale_max_abs"] == 0
    probe = np.arange(dimension).reshape(size, size)/dimension
    checks["flat_scalar_fourier_score_max_abs"] = float(np.max(np.abs(
        -sigma*fft_filter(probe, 1/(flat+sigma**2)) + sigma*probe/(power.mean()+sigma**2))))
    assert checks["flat_scalar_fourier_score_max_abs"] < 1e-12
    radius_levels, group = np.unique(radius2, return_inverse=True)
    group = group.reshape(size, size)
    mode_power = np.array([power[group == i].mean() for i in range(len(radius_levels))])
    analytic = mode_power / (mode_power + sigma**2)
    estimates = {}
    for distribution in ("gaussian", "gmm"):
        total = np.zeros(len(radius_levels))
        total_squared = np.zeros_like(total)
        for start in range(0, n_samples, 1024):
            batch = min(1024, n_samples-start)
            white = rng.standard_normal((batch, size, size))
            if distribution == "gaussian":
                clean = fft_filter(white, np.sqrt(power))
            else:
                clean = means[rng.integers(2*dimension, size=batch)] + np.sqrt(1-rho**2)*fft_filter(white, np.sqrt(power))
            epsilon = rng.standard_normal(clean.shape)
            noisy = clean + sigma*epsilon
            # T = -epsilon - sigma*s_G; the regression target, not the conditional residual mean.
            target = -epsilon + sigma*fft_filter(noisy, 1/(power+sigma**2))
            target_power = np.abs(np.fft.fft2(target, norm="ortho"))**2
            grouped = np.stack([target_power[:, group == i].mean(1) for i in range(len(radius_levels))], axis=1)
            total += grouped.sum(0)
            total_squared += np.square(grouped).sum(0)
        estimate = total/n_samples
        standard_error = np.sqrt(np.maximum((total_squared-n_samples*estimate**2)/(n_samples-1), 0)/n_samples)
        estimates[distribution] = (estimate, standard_error)
        # Finite-sample Monte Carlo check, not a guarantee at every seed/budget.
        checks[f"{distribution}_target_moment_max_abs_error"] = float(np.abs(estimate-analytic).max())
        assert checks[f"{distribution}_target_moment_max_abs_error"] < 0.045
        data[f"{distribution}_target_moment"] = estimate
        data[f"{distribution}_target_moment_se"] = standard_error

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), gridspec_kw={"width_ratios": [0.9, 1, 1.1]})
    shifted_frequency = np.fft.fftshift(frequency)
    edge_lo, edge_hi = shifted_frequency[0]-0.5/size, shifted_frequency[-1]+0.5/size
    image = axes[0].imshow(np.fft.fftshift(power), origin="lower", cmap="cividis", extent=(edge_lo, edge_hi, edge_lo, edge_hi))
    fig.colorbar(image, ax=axes[0], fraction=0.047, pad=0.04, label="Power $P_k$")
    axes[0].set(xlabel="Horizontal frequency", ylabel="Vertical frequency", title="Structured population spectrum")
    noise = np.logspace(-2, 2, 201)
    for value, label, color in [(power.max(), "Highest-power mode", COLORS["data"]),
                                (np.median(power), "Median-power mode", COLORS["residual"]),
                                (power.min(), "Lowest-power mode", "#8D5DAA")]:
        axes[1].semilogx(noise, np.sqrt(value/(value+noise**2)), lw=2, color=color, label=label)
    axes[1].semilogx(noise, np.sqrt(1/(1+noise**2)), color=COLORS["reference"], ls="--", lw=2, label="Scalar control ($P=1$)")
    axes[1].set(xlabel=r"Noise scale $\sigma_t$", ylabel=r"Residual output scale $b_{t,k}$",
                title="Scales depend on signal power", ylim=(0, 1.04))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.15)
    radius = np.sqrt(radius_levels)
    axes[2].plot(radius, analytic, color="#172636", lw=1.8, label=r"Exact $P_k/(P_k+\sigma_t^2)$")
    for distribution, marker, color in [("gaussian", "o", COLORS["reference"]), ("gmm", "s", COLORS["data"])]:
        estimate, standard_error = estimates[distribution]
        axes[2].errorbar(radius, estimate, yerr=2*standard_error, fmt=marker, ms=3.8,
                         color=color, mfc="white", capsize=2, lw=1,
                         label=f"{distribution.upper() if distribution == 'gmm' else 'Gaussian'} Monte Carlo")
    axes[2].set(xlabel="Radial frequency", ylabel=r"$\mathbb{E}|\widehat T_k|^2$",
                title="Same target second moments", ylim=(0, 1.06))
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.15)
    fig.suptitle("Frequency scaling uses population second moments, not Gaussian data", fontsize=15, y=1.035)
    fig.text(0.5, -0.035, rf"$8\times8$ grid, mean power 1. Right: $\sigma_t={sigma}$, {n_samples:,} samples per distribution, bars = 2 Monte Carlo SE; modes grouped by radius.",
             ha="center", fontsize=9, color="#526476")
    fig.tight_layout()
    save_figure(fig, output, "frequency_scaling")
    data.update(frequency_power=power, frequency_radius=radius, analytic_target_moment=analytic,
                frequency_noise=noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", type=Path, default=ROOT / "assets")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--samples", type=int, default=32768,
                        help="Monte Carlo observations per synthetic distribution (minimum 8192)")
    args = parser.parse_args()
    if args.samples < 8192:
        parser.error("--samples must be at least 8192 for the moment diagnostic")
    args.out.mkdir(parents=True, exist_ok=True)
    configure_style()
    seeds = np.random.SeedSequence(args.seed).spawn(2)
    rho = 0.92
    checks, data = {}, {}
    grid = np.linspace(-3, 3, 301)
    h = 1e-5
    _, score = mixture_density_score(grid, 0.35**2, rho)
    finite_difference = (np.log(mixture_density_score(grid+h, 0.35**2, rho)[0]) - np.log(mixture_density_score(grid-h, 0.35**2, rho)[0])) / (2*h)
    checks["score_log_density_gradient_max_abs_error"] = float(np.max(np.abs(score-finite_difference)))
    assert checks["score_log_density_gradient_max_abs_error"] < 1e-7
    make_process_figure(args.out, np.random.default_rng(seeds[0]), data, rho)
    make_residual_figure(args.out, data, rho)
    make_frequency_figure(args.out, np.random.default_rng(seeds[1]), data, checks, args.samples)
    np.savez_compressed(args.out / "diagnostics_data.npz", **data)
    metadata = {
        "kind": "synthetic oracle illustrations; no learned-model or image-benchmark results",
        "seed": args.seed, "monte_carlo_samples_per_distribution": args.samples,
        "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {"process": "VE, alpha=1, sigma_squared=4*t, t in [0,1]", "integration_steps": 1000,
                     "process_mixture_rho": rho, "residual_figure_sigma": 0.35,
                     "spectrum_grid": [8, 8], "spectrum_knee": 0.15, "spectrum_exponent": 1.5,
                     "spectrum_mean_power": 1.0, "spectrum_mixture_rho": 0.85, "spectrum_moment_sigma": 0.7},
        "checks": checks,
        "figures": ["stochastic_process", "gaussian_residual", "frequency_scaling"],
    }
    (args.out / "diagnostics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
