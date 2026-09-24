"""Report tables and figures of the GMM gate experiments.

File names, CSV columns and figure stems are those of the epoch-0 runners.
Figures pick arms by gate mode and covariance (never by position) and take
labels and colours from ``registry.STYLES``.  matplotlib is imported lazily.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from fourier_score.gmm import MatchedMomentFamily, make_model
from fourier_score.utils import json_write
from experiments.common import mean_sd, pyplot, save_figure, write_csv
from experiments.gmm.registry import style

AUDIT_KEYS = ("spectrum_lambda", "seed", "method", "origin", "checkpoint", "checkpoint_sha256",
              "initial_backbone_sha256", "training_stream_first_batch_sha256", "final_data_rng_sha256",
              "final_ema_backbone_sha256", "training_provenance")
ORIGIN_KEYS = ("origin_output_root", "origin_protocol_sha256")  # reused checkpoints only


def sigma_curve(rows):
    """Sorted sigmas and the mean and sd (0 for one seed) of ``score_error`` at each."""
    xs = sorted({r["sigma"] for r in rows})
    stats = [mean_sd([r["score_error"] for r in rows if r["sigma"] == x]) for x in xs]
    return xs, np.array([s[0] for s in stats]), np.array([s[1] or 0 for s in stats])


def export_report(report, cfg, provenance, results, tested, selection, pairing, arms, *, figure_stem, paired_arms):
    """Tables, summary.json and the comparison figure; ``paired_arms`` are compared with every arm."""
    report.mkdir(parents=True, exist_ok=True)
    per_seed, noise, frequency, curves = [], [], [], []
    for result in tested:
        metric = result["test"]
        common = {key: result[key] for key in ("spectrum_lambda", "seed", "method")}
        per_seed.append({**common, "steps": result["step"],
                         "score_error": metric["score_error"],
                         "unweighted_score_error": sum(r["unweighted_score_error"] * r["n"] for r in metric["per_noise"]) / metric["n_observations"],
                         "dsm_pixel_mean": metric["dsm_pixel_mean"],
                         "gradient_norm_mean": result["gradient_norm_sum"] / result["step"],
                         "gradient_clip_fraction": result["gradient_clip_count"] / result["step"],
                         "test_bank_sha256": metric["bank_sha256"]})
        noise.extend({**common, **row} for row in metric["per_noise"])
        frequency.extend({**common, "band": band, "score_error": value}
                         for band, value in enumerate(metric["score_error_by_frequency"]))
    for result in results:
        curves.extend({"spectrum_lambda": result["spectrum_lambda"], "seed": result["seed"],
                       "method": result["method"], "step": row["step"],
                       "validation_score_error": row["score_error"]}
                      for row in result["validation"])
    summary = []
    for lam in cfg.spectrum_lambdas:
        for arm in arms:
            rows = [r for r in per_seed if r["spectrum_lambda"] == lam and r["method"] == arm.name]
            row = dict(spectrum_lambda=lam, method=arm.name, n_seeds=len(rows))
            for metric in ("score_error", "unweighted_score_error"):
                row[metric + "_mean"], row[metric + "_sd"] = mean_sd([r[metric] for r in rows])
            summary.append(row)
        hashes = {r["test_bank_sha256"] for r in per_seed if r["spectrum_lambda"] == lam}
        if len(hashes) != 1:
            raise ValueError("Final comparisons must share a test bank")
    indexed = {(r["spectrum_lambda"], r["seed"], r["method"]): r for r in per_seed}
    paired = []
    for lam in cfg.spectrum_lambdas:
        for arm in paired_arms:
            for baseline in arms:
                if baseline.name == arm.name:
                    continue
                a = [indexed[(lam, seed, arm.name)]["score_error"] for seed in cfg.seeds]
                b = [indexed[(lam, seed, baseline.name)]["score_error"] for seed in cfg.seeds]
                delta = np.asarray(a) - np.asarray(b)
                avg, sd = mean_sd(delta)
                paired.append(dict(spectrum_lambda=lam, method=arm.name, baseline=baseline.name,
                                   paired_delta_mean=avg, paired_delta_sd=sd,
                                   relative_change_percent=float(100 * (np.mean(a) / np.mean(b) - 1)),
                                   wins=int((delta < 0).sum()), n_seeds=len(a)))
    for name, rows in (("summary", summary), ("per_seed", per_seed), ("paired_comparisons", paired),
                       ("noise_resolved", noise), ("frequency_resolved", frequency),
                       ("validation_curves", curves), ("validation_selection", selection["candidates"])):
        write_csv(report / f"{name}.csv", rows)
    payload = dict(config=asdict(cfg), provenance=provenance, selection=selection,
                   n_training_runs=len(results), n_test_runs=len(tested),
                   test_observations_per_spectrum=cfg.n_noise_levels * cfg.test_per_noise,
                   arms=[asdict(a) for a in arms], summary=summary, paired=paired, pairing_checks=pairing)
    json_write(payload, report / "summary.json")
    plot_report(report, cfg, arms, summary, noise, figure_stem=figure_stem)
    return payload


def export_diagnostics(report, tested, cfg, arms):
    regions, transition = [], []
    for result in tested:
        common = {key: result[key] for key in ("spectrum_lambda", "seed", "method")}
        for region, lo, hi in (("low", 0., .3), ("middle", .3, 1.), ("high", 1., float("inf"))):
            bins = [r for r in result["test"]["per_noise"] if lo <= r["sigma"] < hi]
            if bins:
                n = sum(r["n"] for r in bins)
                regions.append({**common, "region": region, "n": n,
                                "score_error": sum(r["n"] * r["score_error"] for r in bins) / n})
        if "transition_test" in result:
            transition.extend({**common, **row} for row in result["transition_test"]["per_noise"])
    summary = []
    for lam in cfg.spectrum_lambdas:
        for arm in arms:
            for region in ("low", "middle", "high"):
                values = [r["score_error"] for r in regions if r["spectrum_lambda"] == lam
                          and r["method"] == arm.name and r["region"] == region]
                if values:
                    mean, sd = mean_sd(values)
                    summary.append(dict(spectrum_lambda=lam, method=arm.name, region=region,
                                        score_error_mean=mean, score_error_sd=sd, n_seeds=len(values)))
    write_csv(report / "noise_regions_per_seed.csv", regions)
    write_csv(report / "noise_regions.csv", summary)
    write_csv(report / "transition_resolved.csv", transition)
    audit = [{key: r[key] for key in (*AUDIT_KEYS, *(k for k in ORIGIN_KEYS if k in r))} for r in tested]
    json_write(audit, report / "checkpoint_audit.json")
    return summary, transition


def plot_report(report, cfg, arms, summary, noise, *, figure_stem):
    plt = pyplot()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8 if len(arms) > 15 else 5.2 if len(arms) > 10 else 4.7))
    focus = max(cfg.spectrum_lambdas)
    for arm in arms:
        label, color, linestyle = style(arm, arms)
        rows = [r for r in summary if r["method"] == arm.name]
        axes[0].errorbar([r["spectrum_lambda"] for r in rows], [r["score_error_mean"] for r in rows],
                         yerr=[r["score_error_sd"] or 0 for r in rows], label=label,
                         color=color, ls=linestyle, marker="o", capsize=3)
        xs, means, sds = sigma_curve([r for r in noise if r["method"] == arm.name and r["spectrum_lambda"] == focus])
        axes[1].plot(xs, means, color=color, ls=linestyle, label=label, lw=1.8)
        axes[1].fill_between(xs, np.maximum(0, means-sds), means+sds, color=color, alpha=.10)
    axes[0].set(xlabel="Spectrum strength lambda (0 = flat)", ylabel="Test scaled true-score MSE",
                title=f"Final EMA after {cfg.steps:,} updates", xticks=cfg.spectrum_lambdas)
    axes[1].set(xlabel="Noise sigma", ylabel="Test scaled true-score MSE", xscale="log",
                title=f"Noise-resolved error: lambda = {focus:g}")
    axes[1].set_xticks([.12, .25, .5, 1., 2.5], labels=["0.12", "0.25", "0.5", "1", "2.5"])
    axes[1].minorticks_off()
    for ax in axes:
        ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=5 if len(arms) > 7 else 4, frameon=False)
    fig.tight_layout(rect=(0, .25 if len(arms) > 15 else .19 if len(arms) > 10 else .13, 1, 1))
    save_figure(fig, report, figure_stem)


def fourier_normalized(arms, gate_modes=None):
    """Fourier-covariance arms trained with the normalized residual loss, in arm order."""
    return [a for a in arms if a.parameterization == "fourier_gaussian" and a.objective == "normalized_residual"
            and (gate_modes is None or a.gate_mode in gate_modes)]


def gate_window(selected, gate_mode):
    """The (last) selected arm of ``gate_mode``; its sigma_lo/sigma_hi mark the transition."""
    return next(a for a in reversed(selected) if a.gate_mode == gate_mode)


def plot_transition(report, cfg, arms, rows, *, figure_stem):
    """Plateau experiment: fixed gates and the transition-resolved test error."""
    plt = pyplot()
    family = MatchedMomentFamily(cfg, "gmm", max(cfg.spectrum_lambdas))
    selected = fourier_normalized(arms)
    window = gate_window(selected, "log_sigma_plateau")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sigma = torch.logspace(np.log10(.1), np.log10(3.), 400)
    for arm in selected:
        label, color, _ = style(arm, selected)
        model = make_model(family, arm, cfg.seeds[0])
        gate = model.reference.gate_value(sigma).flatten().numpy()
        axes[0].plot(sigma.numpy(), gate, label=label, color=color)
        xs, means, sd = sigma_curve([r for r in rows if r["method"] == arm.name])
        axes[1].plot(xs, means, label=label, color=color, marker=".")
        axes[1].fill_between(xs, means-sd, means+sd, color=color, alpha=.12)
    for ax in axes:
        ax.axvspan(window.sigma_lo, window.sigma_hi, color="#777777", alpha=.1)
        ax.grid(alpha=.18)
        ax.set_xlabel("Noise sigma")
    axes[0].set(xscale="log", ylabel="Gaussian reference gate g", title="Fixed gate before training")
    axes[1].set(ylabel="Test scaled true-score MSE", title=f"Transition diagnostic: lambda = {family.lam:g}")
    axes[0].set_xticks([.1, .3, .8, 1., 3.], labels=["0.1", "0.3", "0.8", "1", "3"])
    axes[0].minorticks_off()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    save_figure(fig, report, figure_stem)


# Two historical layouts of the gate diagnostic figure: the spectral experiment's
# (with the spectral transition window and cap) and the gate-shape experiments'.
LAYOUTS = {
    "spectral": dict(figsize=(15, 4.4), n_sigma=300, title="Gate range across frequencies",
                     xticks=[.1, .3, 1., 2., 3.], labels=["0.1", "0.3", "1", "2", "3"],
                     band_alpha=.12, ncol=4, bottom=.08, spectral_window=True),
    "shapes": dict(figsize=(15, 4.7), n_sigma=400, title="Fixed gate shapes",
                   xticks=[.1, .3, 1., 1.5, 3.], labels=["0.1", "0.3", "1", "1.5", "3"],
                   band_alpha=.1, ncol=3, bottom=.13, spectral_window=False),
}


def plot_gate_diagnostics(report, cfg, arms, transition, *, figure_stem, layout, gate_modes=None, styles=None):
    """Gate range and residual scale over sigma, and the transition-resolved test error.

    A gate that varies across frequencies is drawn as its min-max band; ``styles``
    (gate_mode -> (label, colour)) overrides ``registry.style`` for legacy figures.
    """
    plt = pyplot()
    look = LAYOUTS[layout]
    family = MatchedMomentFamily(cfg, "gmm", max(cfg.spectrum_lambdas))
    selected = fourier_normalized(arms, gate_modes)
    fig, axes = plt.subplots(1, 3, figsize=look["figsize"])
    sigma = torch.logspace(np.log10(cfg.sigma_min), np.log10(cfg.sigma_max), look["n_sigma"])
    profile = []
    for arm in selected:
        label, color, _ = style(arm, selected)
        override_label, override_color = (styles or {}).get(arm.gate_mode, (None, None))
        label, color = override_label or label, override_color or color
        ref = make_model(family, arm, cfg.seeds[0]).reference
        gate, _, scale, total = ref._coefficients(sigma, ref.power[None])
        b = (ref.power[None] / total).sqrt()
        lower = gate.flatten(1).min(1).values.numpy()
        upper = gate.flatten(1).max(1).values.numpy()
        ratio = (scale / b).flatten(1).max(1).values.numpy()
        axes[0].plot(sigma.numpy(), upper, label=label, color=color)
        if (upper > lower).any():
            if look["spectral_window"]:
                axes[0].plot(sigma.numpy(), lower, color=color, ls="--", lw=1)
            axes[0].fill_between(sigma.numpy(), lower, upper, color=color, alpha=.1)
        axes[1].plot(sigma.numpy(), ratio, color=color)
        profile.extend(dict(method=arm.name, sigma=float(s), gate_min=float(lo), gate_max=float(hi),
                            max_scale_ratio=float(r)) for s, lo, hi, r in zip(sigma, lower, upper, ratio))
        xs, means, sd = sigma_curve([r for r in transition if r["method"] == arm.name])
        axes[2].plot(xs, means, color=color, marker=".")
        axes[2].fill_between(xs, means-sd, means+sd, color=color, alpha=look["band_alpha"])
    window = gate_window(selected, "spectral_cap") if look["spectral_window"] else None
    for ax in axes:
        if window is not None:
            ax.axvspan(window.sigma_lo, window.sigma_hi, color="#777777", alpha=.08)
        ax.grid(alpha=.18)
        ax.set_xlabel("Noise sigma")
    axes[0].set(xscale="log", ylabel="Gate g", title=look["title"])
    axes[1].set(xscale="log", ylabel="Maximum c / b", title="Residual scale relative to Fourier")
    if window is not None:
        axes[1].hlines((1+window.delta**2)**.5, window.sigma_hi, cfg.sigma_max,
                       colors="black", linestyles=":", lw=1.5)
    axes[2].set(ylabel="Test scaled true-score MSE", title=f"Transition diagnostic: lambda = {family.lam:g}")
    for ax in axes[:2]:
        ax.set_xticks(look["xticks"], labels=look["labels"])
        ax.minorticks_off()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=look["ncol"], frameon=False)
    fig.tight_layout(rect=(0, look["bottom"], 1, 1))
    save_figure(fig, report, figure_stem)
    write_csv(report / "gate_profile.csv", profile)


def plot_diagnostics(diagnostics, report, cfg, arms, transition):
    """The experiment's diagnostic figure (a ``registry.Diagnostics``)."""
    if diagnostics.layout == "transition":
        plot_transition(report, cfg, arms, transition, figure_stem=diagnostics.figure_stem)
    else:
        plot_gate_diagnostics(report, cfg, arms, transition, figure_stem=diagnostics.figure_stem,
                              layout=diagnostics.layout, gate_modes=diagnostics.gate_modes,
                              styles=diagnostics.styles)
