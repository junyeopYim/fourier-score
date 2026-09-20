"""Download supported datasets, create the split and train-only Fourier cache."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fourier_score.config import add_config_args, from_args
from fourier_score.data import build_data, prepare_stats


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.download:
        args.set.append("data_loader.args.download=true")
    cfg = from_args(args)
    bundle = build_data(cfg)
    stats = prepare_stats(cfg, bundle, force=args.force)
    print(
        json.dumps(
            {
                "identity": stats["identity"],
                "n_train": stats["n_train"],
                "n_effective": stats["n_effective"],
                "n_validation": len(bundle.validation),
                "eval_split": stats["eval_split"],
                "floored_fraction": stats["floored_fraction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
