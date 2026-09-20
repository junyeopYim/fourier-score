"""Load EMA from either last.pt or an EMA-only snapshot."""
import copy
import warnings
import torch
from parse_config import validate,apply_overrides
from model.model import build_model
from base.base_trainer import FORMAT
from utils.util import load_checkpoint,configure_runtime,source_hash


def load_inference(path,overrides=(),device=None):
    ckpt=load_checkpoint(path)
    if ckpt.get('format')!=FORMAT: raise ValueError('Not a fourier-image-template v1 checkpoint')
    changes=list(overrides)
    if device is not None: changes.append('device='+device)
    for change in changes:
        key=change.split('=',1)[0]
        if not (key=='device' or key.startswith(('sampling.','evaluation.','backend.')) or key in ('data_loader.args.root','data_loader.args.download','data_loader.args.num_workers')):
            raise ValueError(f'Inference cannot alter the trained model/process/loss/statistics: {key}')
    # Supply new optional evaluation defaults before applying overrides to old snapshots.
    cfg=validate(apply_overrides(validate(copy.deepcopy(ckpt['config'])),changes))
    dev=configure_runtime(cfg)
    model=build_model(cfg,ckpt['stats'],dev)
    model.load_state_dict(ckpt['model'],strict=True)
    if ckpt['kind']=='training':
        params=dict(model.named_parameters())
        with torch.no_grad():
            for n,t in ckpt['ema']['shadow'].items(): params[n].copy_(t.to(params[n]))
    elif ckpt['kind']!='ema': raise ValueError('Unknown checkpoint kind')
    if ckpt['source_sha256']!=source_hash():
        warnings.warn('Inference source differs from checkpoint; results are a new evaluation protocol',stacklevel=2)
    model.eval()
    return model,cfg,dev,ckpt
