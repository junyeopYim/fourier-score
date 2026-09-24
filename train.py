"""Train or resume. All options are resolved into one saved JSON config."""

import argparse
import json
from fourier_score.config import (
    CLI_OPTIONS,
    DOWNLOAD,
    add_config_args,
    apply_overrides,
    from_args,
    validate,
)
from fourier_score.parse_config import add_options
from fourier_score.trainer.trainer import Trainer
from fourier_score.utils import load_checkpoint


def resume(path, changes):
    """The checkpoint's config with defaults added since it was saved, then changes."""
    checkpoint = load_checkpoint(path)
    return validate(apply_overrides(validate(checkpoint["config"]), changes)), checkpoint


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(config=None)
    parser.add_argument("-r", "--resume")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config without loading data or training",
    )
    add_options(parser, [DOWNLOAD])
    args = parser.parse_args()
    cfg, checkpoint = from_args(parser, args, [*CLI_OPTIONS, DOWNLOAD], resume)
    if args.dry_run:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return
    Trainer(cfg, checkpoint).train()


if __name__ == "__main__":
    main()
