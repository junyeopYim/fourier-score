"""Render the README method diagram as reproducible SVG and PNG artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "stix",
        "font.size": 13, "svg.hashsalt": "fourier-score-method-v1",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink, muted, blue, teal = "#172636", "#526476", "#305da8", "#087c78"

    def label(x, y, text, size=14, color=ink, weight="normal", ha="center"):
        ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                ha=ha, va="center", transform=ax.transAxes)

    def box(x, y, w, h, face, edge="none"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                  facecolor=face, edgecolor=edge, linewidth=1.2))

    def arrow(start, end, color=muted):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                    linewidth=1.4, color=color, shrinkA=3, shrinkB=3))

    label(.05, .958, "Same DSM objective, different score parameterizations", 23, weight="bold", ha="left")
    label(.05, .913, "Continuous VE: matched backbone initialization, noise draws, loss reduction and sampler", 13, muted, ha="left")
    box(.18, .804, .64, .073, "#f0f3f7")
    label(.50, .840, r"$x\sim p_{\mathrm{train}},\quad t\sim\mathcal{U}[t_{\min},1],\quad \epsilon\sim\mathcal{N}(0,I),\quad y=x+\sigma_t\epsilon$", 20)
    arrow((.32, .80), (.26, .75))
    arrow((.68, .80), (.74, .75))

    box(.045, .373, .43, .362, "#f3f6fc", "#cdd9ed")
    box(.525, .373, .43, .362, "#eef8f6", "#b9ddd6")
    label(.075, .705, "Score-SDE baseline", 19, blue, "bold", "left")
    label(.555, .705, "Fourier Gaussian (ours)", 19, teal, "bold", "left")
    label(.26, .664, "Network predicts the full scaled score", 12, muted)
    label(.74, .664, "Fixed Gaussian score + learned residual", 12, muted)

    box(.093, .546, .334, .076, "white", "#cdd9ed")
    label(.26, .585, r"$\sigma_t s_\theta(y,t)=h_\theta(y,t)$", 23, blue)
    box(.56, .546, .36, .076, "white", "#b9ddd6")
    label(.74, .585, r"$\sigma_t s_\theta=\sigma_t s_G+\mathcal{F}^{-1}[b_t\,\mathcal{F}h_\theta]$", 20, teal)
    label(.74, .510, r"$\mu,\,P_k$: fixed training-only mean and spectrum", 12, muted)
    arrow((.26, .54), (.26, .466), blue)
    arrow((.74, .49), (.74, .466), teal)
    label(.26, .430, r"$\mathcal{L}_{\mathrm{base}}=\mathbb{E}\,\|h_\theta+\epsilon\|^2$", 24, blue)
    label(.74, .430, r"$\mathcal{L}_{\mathrm{FG}}=\mathbb{E}\,\|\sigma_t s_G+\mathcal{F}^{-1}[b_t\mathcal{F}h_\theta]+\epsilon\|^2$", 18, teal)

    label(.50, .325, r"Both optimize  $\mathcal{L}_{\mathrm{DSM}}=\mathbb{E}\,\|\sigma_t s_\theta(y,t)+\epsilon\|^2$", 22, weight="bold")
    label(.05, .260, "What is fixed in the proposed adapter?", 15, weight="bold", ha="left")
    label(.05, .205, r"$\sigma_t\widehat{s}_{G,k}=-\sigma_t(\widehat y_k-\widehat\mu_k)/(P_k+\sigma_t^2)$", 22, teal, ha="left")
    label(.05, .150, r"$b_{t,k}=\sqrt{P_k/(P_k+\sigma_t^2)}$", 22, teal, ha="left")
    label(.05, .092, "Gaussian reference coefficient = 1. No added learned weights.", 12, muted, ha="left")
    label(.05, .052, r"In residual coordinates, the loss retains $b_{t,k}^{\,2}$ weighting.", 12, muted, ha="left")

    plot = fig.add_axes((.67, .073, .27, .180))
    ratio = np.logspace(-2, 2, 250)
    plot.semilogx(ratio, 1 / np.sqrt(1 + ratio**2), color=teal, lw=2.5)
    plot.set(xlim=(.01, 100), ylim=(0, 1.08), yticks=[0, .5, 1],
             xlabel=r"Noise / signal scale  $\sigma_t/\sqrt{P_k}$", ylabel=r"$b_{t,k}$")
    plot.set_title("Analytic residual scale", fontsize=12, loc="left", pad=5)
    plot.tick_params(labelsize=10)
    plot.grid(alpha=.18)
    plot.set_axisbelow(True)
    plot.xaxis.label.set_size(11)
    for spine in plot.spines.values():
        spine.set_color("#b0bac4")

    out = Path(__file__).resolve().parents[1] / "docs/assets"
    out.mkdir(parents=True, exist_ok=True)
    svg = out / "loss_comparison.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(out / "loss_comparison.png", dpi=160, metadata={"Software": "Fourier Score / matplotlib"})
    plt.close(fig)
    print(out / "loss_comparison.svg")


if __name__ == "__main__":
    main()
