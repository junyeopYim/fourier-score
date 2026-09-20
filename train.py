"""Train or resume. All options are resolved into one saved JSON config."""

import argparse
import json
from fourier_score.config import add_config_args, load_config, apply_overrides, validate
from fourier_score.training import Trainer
from fourier_score.utils import load_checkpoint


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(config=None)
    parser.add_argument("-r", "--resume")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config without loading data or training",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download MNIST/CIFAR-10 if missing"
    )
    args = parser.parse_args()
    changes = list(args.set)
    if args.device is not None:
        changes.append("device=" + args.device)
    if args.parameterization is not None:
        changes.append("parameterization=" + args.parameterization)
    if args.download:
        changes.append("data_loader.args.download=true")
    checkpoint = None
    if args.resume:
        if args.config is not None:
            parser.error("Resume uses checkpoint config; use --set for allowed changes")
        checkpoint = load_checkpoint(args.resume)
        cfg = validate(apply_overrides(validate(checkpoint["config"]), changes))
    else:
        cfg = load_config(args.config or "config.json", changes)
    if args.dry_run:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return
    Trainer(cfg, checkpoint).train()


if __name__ == "__main__":
    main()
