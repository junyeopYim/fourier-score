"""Optional torch-fidelity FID/IS. Not the original TensorFlow score-SDE protocol.

Install: uv sync --extra metrics. Inception weights download on first use.
MPS-trained models can be evaluated here on CPU (or a separate CUDA machine).
"""
import argparse
import hashlib
from pathlib import Path
import importlib.metadata
from utils.util import resolve_device,json_write

def folder_manifest(path):
    path=Path(path); files=sorted(p for p in path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    if len(files)<2: raise ValueError(f'At least two images required: {path}')
    h=hashlib.sha256()
    for p in files:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return {'path':str(path.resolve()),'count':len(files),'sha256':h.hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real',required=True); p.add_argument('--generated',required=True); p.add_argument('-o','--output',required=True)
    p.add_argument('--device',default='cpu',help='cpu or cuda; Inception evaluation is not on MPS')
    p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args(); device=resolve_device(a.device)
    if device.type=='mps': p.error('Use --device cpu for torch-fidelity after MPS training')
    if a.batch_size<1: p.error('batch-size must be positive')
    if device.type=='cuda':
        import torch
        torch.cuda.set_device(device.index if device.index is not None else 0)
    real=folder_manifest(a.real); generated=folder_manifest(a.generated)
    try: import torch_fidelity
    except ImportError as e: raise RuntimeError('Install optional metrics: uv sync --extra metrics') from e
    result=torch_fidelity.calculate_metrics(input1=str(Path(a.generated).resolve()),input2=str(Path(a.real).resolve()),
        cuda=device.type=='cuda',fid=True,isc=True,kid=False,prc=False,batch_size=a.batch_size,verbose=True)
    json_write({'metrics':{k:float(v) for k,v in result.items()},'implementation':'torch-fidelity',
                'version':importlib.metadata.version('torch-fidelity'),'real':real,'generated':generated,
                'device':str(device),'batch_size':a.batch_size,
                'warning':'Not directly interchangeable with original score-SDE TF-Hub/TF-GAN FID'},a.output)

if __name__=='__main__': main()
