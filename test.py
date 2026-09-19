"""Held-out EMA DSM. This metric is not a surrogate declaration of FID quality."""
import argparse
import json
from utils.inference import load_inference
from model.metric import evaluate_dsm
from data_loader.data_loaders import build_data,prepare_stats
from utils.util import json_write,environment

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r','--resume',required=True); parser.add_argument('-o','--output',required=True)
    parser.add_argument('--set',action='append',default=[]); parser.add_argument('--device')
    args=parser.parse_args()
    model,cfg,device,ckpt=load_inference(args.resume,args.set,args.device)
    bundle=build_data(cfg); prepare_stats(cfg,bundle,ckpt['stats'])
    result=evaluate_dsm(model,cfg,bundle.validation,device)
    result.update(step=ckpt['step'],split=bundle.metadata['eval_split'],weights='EMA',environment=environment(device,cfg))
    json_write(result,args.output); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
