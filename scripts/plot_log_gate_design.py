"""Plot the requested log-linear and bounded S-shaped gates, without training.

The committed proposal assets/log_gate_design/design.json is an input of the
log_gates experiment (hashed into its protocol); write there only on purpose,
with --output assets/log_gate_design.
"""

from dataclasses import asdict
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fourier_score.gmm import log_gate_arm
from fourier_score.model.reference import FourierGaussian
from experiments.common import pyplot, save_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "saved/log_gate_design")
    parser.add_argument("--sigma-lo", type=float, default=.1)
    parser.add_argument("--linear-hi", type=float, default=3.)
    parser.add_argument("--sigmoid-hi", type=float, default=1.5)
    parser.add_argument("--sigmoid-center", type=float, default=.75)
    parser.add_argument("--sharpness", type=float, default=2.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    arms = [log_gate_arm("fourier", "linear_log_sigma", sigma_lo=args.sigma_lo, sigma_hi=args.linear_hi),
            log_gate_arm("fourier", "bounded_log_sigmoid", sigma_lo=args.sigma_lo, sigma_hi=args.sigmoid_hi,
                         sigma_switch=args.sigmoid_center, sharpness=args.sharpness)]
    stats = dict(mean=torch.zeros(1, 2, 2), power=torch.ones(1, 2, 2))
    refs = [FourierGaussian(stats, gate=arm.gate).double() for arm in arms]
    lo, hi = args.sigma_lo*.85, max(args.linear_hi, args.sigmoid_hi)*1.1
    sigma = torch.logspace(math.log10(lo), math.log10(hi), 801, dtype=torch.float64)
    samples = torch.tensor([.1, .2, .3, .5, .75, 1., 1.5, 3.], dtype=torch.float64)
    curves = [ref.gate_value(sigma).flatten().tolist() for ref in refs]
    values = [ref.gate_value(samples).flatten().tolist() for ref in refs]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "curves.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["sigma", "linear_log_sigma", "bounded_log_sigmoid"])
        writer.writerows(zip(sigma.tolist(), *curves))
    design = dict(status="Gate design and implementation; no new performance experiment",
                  axis="log sigma", arms=[asdict(a) for a in arms],
                  constraints=dict(backbones_per_model=1, loss="normalized_residual", teachers=False),
                  sample_values=[dict(sigma=s, linear=g, sigmoid=h) for s, g, h in zip(samples.tolist(), *values)],
                  sigmoid_tanh_equivalent=True,
                  previous_sigma_linear_experiment_is_this_design=False)
    (output / "design.json").write_text(json.dumps(design, indent=2) + "\n")

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    ax.plot(sigma.tolist(), curves[0], color="#111111", lw=3,
            label=f"Linear in log sigma: {args.sigma_lo:g} to {args.linear_hi:g}")
    ax.plot(sigma.tolist(), curves[1], color="#d8a000", lw=3.5,
            label=f"Sigmoid / tanh: center {args.sigmoid_center:g}, plateau at {args.sigmoid_hi:g}")
    ax.scatter([args.sigmoid_center, args.sigmoid_hi], [.5, 1.], color="#d8a000", s=45, zorder=5)
    ax.axvline(args.sigmoid_center, color="#888888", lw=.8, ls=":")
    ax.axvline(args.sigmoid_hi, color="#888888", lw=.8, ls=":")
    ax.axhline(.5, color="#aaaaaa", lw=.7, ls=":")
    ax.set(xscale="log", xlim=(lo, hi), ylim=(-.025, 1.045), xlabel="Noise sigma (log scale)", ylabel="Gate g",
           title="Gate design on the requested log-noise axis")
    ax.set_xticks([.1, .3, .5, .75, 1., 1.5, 3.], labels=["0.1", "0.3", "0.5", "0.75", "1", "1.5", "3"])
    ax.minorticks_off()
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    save_figure(fig, output, "log_gate_design")
    print(json.dumps(design["sample_values"], indent=2))


if __name__ == "__main__":
    main()
