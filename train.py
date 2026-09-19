"""Train or resume. All options are resolved into one saved JSON config."""
import argparse
import json
from parse_config import add_config_args,load_config,apply_overrides,validate
from trainer.trainer import Trainer
from utils.util import load_checkpoint

def main():
    parser=add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(config=None)
    parser.add_argument('-r','--resume')
    parser.add_argument('--dry-run',action='store_true',help='Print resolved config without loading data or training')
    args=parser.parse_args(); changes=list(args.set)
    if args.device is not None: changes.append('device='+args.device)
    checkpoint=None
    if args.resume:
        if args.config is not None: parser.error('Resume uses checkpoint config; use --set for allowed changes')
        checkpoint=load_checkpoint(args.resume)
        cfg=validate(apply_overrides(checkpoint['config'],changes))
    else: cfg=load_config(args.config or 'config.json',changes)
    if args.dry_run:
        print(json.dumps(cfg,indent=2,ensure_ascii=False)); return
    Trainer(cfg,checkpoint).train()

if __name__=='__main__': main()
