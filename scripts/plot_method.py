"""Render the VE method schematic as reproducible SVG, PNG, and vector PDF."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
import numpy as np


INK = "#222222"
MUTED = "#626262"
LEARNED = "#242424"
FIXED = "#555555"
RULE = "#D7D7D7"


def make_figure():
    # A two-column paper width; STIX is bundled with Matplotlib, so the figure
    # needs neither a system font installation nor an external TeX renderer.
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "fourier-score-method-v2",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.4, 5.1), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def label(x, y, text, size=9, color=INK, ha="center", weight="normal"):
        return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                       ha=ha, va="center", transform=ax.transAxes)

    def line(points, color=RULE, width=0.65):
        x, y = zip(*points)
        ax.plot(x, y, color=color, lw=width, solid_capstyle="butt")

    def arrow(start, end, color=MUTED):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=7,
            linewidth=0.75, color=color, shrinkA=0, shrinkB=0,
        ))

    def block(x, y, w, h, title, equation, fixed=False):
        color = FIXED if fixed else LEARNED
        ax.add_patch(Rectangle(
            (x, y), w, h, linewidth=0.8, edgecolor=color,
            facecolor="white" if fixed else "#FAFAFA",
            linestyle=(0, (3, 2)) if fixed else "solid",
        ))
        label(x + w / 2, y + 0.70 * h, title, 8.1, color)
        label(x + w / 2, y + 0.29 * h, equation, 11, color)

    def panel(x, y, letter, title):
        label(x, y, f"({letter})", 10, ha="left", weight="bold")
        label(x + 0.039, y, title, 10, ha="left")

    # The sampling distribution and objective are common to both branches.
    label(.035, .962, "VE perturbation", 8.5, MUTED, ha="left")
    label(.606, .962,
          r"$x\sim p_{\mathrm{train}},\quad t\sim\mathcal{U}[t_{\min},1],"
          r"\quad\epsilon\sim\mathcal{N}(0,I),\quad y=x+\sigma_t\epsilon$", 10)
    line([(.035, .918), (.965, .918)])

    panel(.035, .875, "a", "Direct score (Score-SDE)")
    panel(.345, .875, "b", "Fourier Gaussian")

    # (a) Direct prediction of the complete scaled score.
    label(.168, .806, r"$(y,t)$", 11)
    arrow((.168, .782), (.168, .746))
    block(.071, .640, .194, .106, "Learned network", r"$h_\theta(y,t)$")
    arrow((.168, .640), (.168, .514))
    label(.168, .485, r"$\sigma_t s_\theta=h_\theta$", 12)

    # (b) A fixed analytic branch and a learned residual branch. The latter is
    # scaled in the Fourier basis before addition, exactly as in model/reference.py.
    label(.645, .806, r"$(y,t)$", 11)
    line([(.451, .772), (.828, .772)], MUTED, .75)
    line([(.645, .786), (.645, .772)], MUTED, .75)
    arrow((.451, .772), (.451, .746))
    arrow((.828, .772), (.828, .746))
    block(.345, .640, .212, .106,
          "Fixed Gaussian score", r"$\sigma_t s_G(y,t)$", fixed=True)
    block(.722, .640, .212, .106,
          "Learned network", r"$h_\theta(y,t)$")

    arrow((.828, .640), (.828, .603))
    block(.722, .497, .212, .106,
          "Fixed spectral scaling", r"$\mathcal{F}^{-1}[b_t\,\mathcal{F}h_\theta]$",
          fixed=True)

    # Keep the addition node circular in physical (not normalized) units.
    sum_x, sum_y, radius = .645, .550, .011
    ax.add_patch(Ellipse(
        (sum_x, sum_y), 2 * radius, 2 * radius * 7.4 / 5.1,
        facecolor="white", edgecolor=INK, linewidth=.75,
    ))
    label(sum_x, sum_y, "+", 10)
    line([(.451, .640), (.451, sum_y)], FIXED, .75)
    arrow((.451, sum_y), (sum_x - radius, sum_y), FIXED)
    arrow((.722, sum_y), (sum_x + radius, sum_y), FIXED)
    arrow((sum_x, .534), (sum_x, .510))
    label(sum_x, .485, r"$\sigma_t s_\theta$", 12)

    # Merge the two parameterizations into a single, unchanged DSM objective.
    line([(.168, .461), (.168, .434), (.645, .434), (.645, .461)], MUTED, .75)
    arrow((.500, .434), (.500, .412))
    label(.500, .382,
          r"$\mathcal{L}_{\mathrm{DSM}}="
          r"\mathbb{E}_{x,t,\epsilon}\,\|\sigma_t s_\theta(y,t)+\epsilon\|^2$", 12)
    label(.500, .339, "Shared DSM objective · matched backbone architecture", 8.2, MUTED)
    line([(.035, .307), (.965, .307)])

    # (c) The adapter is analytic; no learned covariance or residual multiplier.
    panel(.035, .271, "c", "Fixed spectral adapter")
    label(.047, .199,
          r"$\sigma_t\widehat{s}_{G,k}="
          r"-\dfrac{\sigma_t(\widehat{y}_k-\widehat{\mu}_k)}{P_k+\sigma_t^2}$",
          11, ha="left")
    label(.047, .112,
          r"$b_{t,k}=\sqrt{\dfrac{P_k}{P_k+\sigma_t^2}}$", 11, ha="left")
    label(.047, .054, r"Training-only moments: $\mu,\,P_k$ (held fixed).",
          8, MUTED, ha="left")
    label(.047, .026, r"Residual-coordinate DSM weights: $b_{t,k}^{\,2}$.",
          8, MUTED, ha="left")

    plot = fig.add_axes((.672, .086, .287, .193))
    ratio = np.logspace(-2, 2, 401)
    scale = 1 / np.sqrt(1 + ratio**2)
    plot.semilogx(ratio, scale, color=FIXED, lw=1.35)
    plot.plot([1, 1], [0, 1 / np.sqrt(2)], color=RULE, lw=.75, ls="--")
    plot.plot(1, 1 / np.sqrt(2), "o", color=FIXED, markersize=3)
    plot.annotate(r"$1/\sqrt{2}$", (1, 1 / np.sqrt(2)), xytext=(7, 5),
                  textcoords="offset points", fontsize=8, color=MUTED)
    plot.set(xlim=(.01, 100), ylim=(0, 1.05), yticks=[0, .5, 1],
             xticks=[.01, 1, 100],
             xlabel=r"Noise-to-signal ratio $\sigma_t/\sqrt{P_k}$",
             ylabel=r"$b_{t,k}$")
    plot.minorticks_off()
    plot.tick_params(labelsize=7.5, length=2.5, width=.6, pad=2)
    plot.xaxis.label.set_size(8)
    plot.xaxis.labelpad = 3
    plot.yaxis.label.set_size(10)
    plot.yaxis.labelpad = 3
    for spine in plot.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "assets")
    args = parser.parse_args()
    fig = make_figure()
    args.out.mkdir(parents=True, exist_ok=True)
    svg = args.out / "loss_comparison.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(args.out / "loss_comparison.png", dpi=300,
                metadata={"Software": "Fourier Score / matplotlib"})
    fig.savefig(args.out / "loss_comparison.pdf",
                metadata={"Creator": "Fourier Score / matplotlib",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(svg)


if __name__ == "__main__":
    main()
