"""Audit identical architecture AND initial values across all three losses."""
import argparse
import copy
import json
import torch
from parse_config import add_config_args,from_args
from model.model import build_model,architecture_report,weight_hash
from utils.util import seed_all,json_write

def main():
    parser=add_config_args(argparse.ArgumentParser(description=__doc__)); parser.add_argument('-o','--output')
    args=parser.parse_args(); cfg=from_args(args); d=cfg['data_loader']['args']; shape=(d['channels'],d['image_size'],d['image_size'])
    stats={'mean':torch.zeros(shape),'power':torch.ones(shape)}; result={}
    # No data loading; fake positive statistics are ONLY for architecture audit.
    for objective in ('score','diffusion','fourier_gaussian'):
        c=copy.deepcopy(cfg); c['loss']['type']=objective; seed_all(c['seed'])
        model=build_model(c,stats,'cpu'); report=architecture_report(model.backbone)
        result[objective]={k:report[k] for k in ('trainable_parameters','all_parameters','architecture_sha256')}
        result[objective]['initial_backbone_sha256']=weight_hash(model.backbone)
        result[objective]['attention_blocks']=sum(v=='AttnBlockpp' for v in report['modules'].values())
        result[objective]['biggan_resblocks']=sum(v=='ResnetBlockBigGANpp' for v in report['modules'].values())
        result[objective]['ddpm_resblocks']=sum(v=='ResnetBlockDDPMpp' for v in report['modules'].values())
        del model
    result['identical']=len({json.dumps(v,sort_keys=True) for v in result.values()})==1
    if not result['identical']: raise AssertionError('Architecture or initialization differs across objectives')
    if args.output: json_write(result,args.output)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
