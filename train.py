"""Train, or resume a template checkpoint. Run from the project root."""
import argparse
from pathlib import Path
from parse_config import ConfigParser, config_arguments
from trainer.trainer import Trainer
from utils.util import load_checkpoint


def main():
    p = config_arguments(argparse.ArgumentParser(description=__doc__), default=None)
    p.add_argument('-r', '--resume', type=Path)
    p.add_argument('--dry-run', action='store_true', help='Print resolved config without loading data or training')
    a = p.parse_args()
    ck = load_checkpoint(a.resume) if a.resume else None
    if a.config is None and ck is not None:
        if 'optimizer' not in ck:
            p.error('EMA-only snapshots cannot resume training; use last.pt')
        config_path = a.resume.parent / 'config.json'
        # Read embedded config, not a potentially unrelated neighboring file.
        parser = ConfigParser.from_dict(ck['config'], a.set)
    else:
        parser = ConfigParser.from_file(a.config or 'config.json', a.set)
    if a.dry_run:
        import json
        print(json.dumps(parser.data, indent=2))
        return
    Trainer(parser.config, ck).train()


if __name__ == '__main__':
    main()
