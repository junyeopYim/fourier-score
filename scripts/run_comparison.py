"""Run three independent experiments, changing ONLY objective + run name."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parse_config import load_config,apply_overrides,validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-c','--config',default='configs/mnist.json'); p.add_argument('--set',action='append',default=[])
    p.add_argument('--device'); p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); changes=list(a.set)
    if a.device: changes.append('device='+a.device)
    cfg=load_config(a.config,changes)
    for loss in ('score','diffusion','fourier_gaussian'):
        overrides=changes+['loss.type='+loss]
        if cfg['name']!='auto': overrides+=['name='+cfg['name']+'_'+loss]
        command=[sys.executable,'train.py','-c',a.config]
        for option in overrides: command+=['--set',option]
        print(json.dumps(command),flush=True)
        if not a.dry_run: subprocess.run(command,check=True)

if __name__=='__main__': main()
