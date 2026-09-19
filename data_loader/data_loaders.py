"""Deterministic pixel preprocessing, train-only splits/statistics, image folders.

Random horizontal flips are performed in the trainer, not worker processes.
For statistics their expectation is computed exactly by including both flips.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,Subset,DataLoader
from model.spectral import estimate_stats
from utils.util import atomic_save,load_checkpoint

class PixelDataset(Dataset):
    def __init__(self,args,train=True):
        self.args=dict(args); self.kind=args['dataset']; self.train=train
        self.raw=None; self.paths=[]
        if self.kind in ('mnist','cifar10'):
            from torchvision.datasets import MNIST,CIFAR10
            klass=MNIST if self.kind=='mnist' else CIFAR10
            self.raw=klass(args['root'],train=train,download=args['download'])
            self.size=len(self.raw)
        elif self.kind=='synthetic':
            self.size=args['synthetic_size'] if train else max(8,args['synthetic_size']//4)
        else:
            root=Path(args['root'])
            if not root.is_dir(): raise FileNotFoundError(f'Image directory not found: {root}')
            extensions={'.png','.jpg','.jpeg','.webp','.bmp'}
            self.paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
            self.size=len(self.paths)
            if not self.size: raise ValueError(f'No supported images in {root}')

    def __len__(self): return self.size

    def __getitem__(self,i):
        a=self.args
        if self.kind=='synthetic':
            g=torch.Generator().manual_seed(131071+int(i)+(0 if self.train else 10_000_000))
            x=torch.rand(a['channels'],a['image_size'],a['image_size'],generator=g)
        elif self.kind=='mnist':
            x=self.raw.data[i].unsqueeze(0).float()/255
            x=F.pad(x,(2,2,2,2))
        elif self.kind=='cifar10':
            x=torch.from_numpy(self.raw.data[i].copy()).permute(2,0,1).float()/255
        else:
            with Image.open(self.paths[i]) as src:
                img=src.convert('RGB' if a['channels']==3 else 'L')
                width,height=img.size
                side=a['crop_size'] or min(width,height)
                if side>min(width,height): raise ValueError(f'crop_size exceeds image size: {self.paths[i]}')
                left=(width-side)//2; top=(height-side)//2
                img=img.crop((left,top,left+side,top+side)).resize((a['image_size'],a['image_size']),Image.Resampling.LANCZOS)
                arr=np.asarray(img,dtype=np.uint8).copy()
            if arr.ndim==2: arr=arr[:,:,None]
            x=torch.from_numpy(arr).permute(2,0,1).float()/255
        return (2*x-1 if a['centered'] else x).contiguous()

    def fingerprint(self):
        a={k:self.args[k] for k in ('dataset','image_size','channels','centered','crop_size','synthetic_size')}
        h=hashlib.sha256(json.dumps(a,sort_keys=True).encode()); h.update(str(self.train).encode())
        if self.raw is not None:
            raw=self.raw.data
            raw=raw.numpy() if torch.is_tensor(raw) else raw
            h.update(memoryview(np.ascontiguousarray(raw)))
        elif self.paths:
            root=Path(self.args['root'])
            # A fast file identity check, not an adversarial content checksum.
            for p in self.paths:
                st=p.stat(); h.update(f'{p.relative_to(root)}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
        else: h.update(str(self.size).encode())
        return h.hexdigest()

@dataclass
class DataBundle:
    full: PixelDataset
    train: Subset
    validation: Dataset
    metadata: dict


def build_data(cfg):
    a=cfg['data_loader']['args']; full=PixelDataset(a,True)
    nval=a['validation_size']
    if nval>len(full)-2: raise ValueError('validation_size must leave at least two training images')
    ids=torch.randperm(len(full),generator=torch.Generator().manual_seed(a['split_seed']))
    valid_ids=ids[:nval]; train_ids=ids[nval:]
    if nval:
        validation=Subset(full,valid_ids.tolist()); label='validation'
    elif a['dataset']!='image_folder':
        validation=PixelDataset(a,False); label='test'
    else:
        raise ValueError('image_folder requires a nonzero validation_size; no silent training-set evaluation')
    meta={'schema_version':1,'dataset_fingerprint':full.fingerprint(),'train_indices':train_ids,
          'val_indices':valid_ids,'eval_split':label,'n_train':len(train_ids),
          'statistics_horizontal_flip_mixture':a['random_flip'],'floor':cfg['fourier']['power_floor']}
    return DataBundle(full,Subset(full,train_ids.tolist()),validation,meta)


def stats_identity(meta):
    h=hashlib.sha256()
    for k in sorted(meta):
        v=meta[k]; h.update(k.encode())
        h.update(v.numpy().tobytes() if torch.is_tensor(v) else json.dumps(v,sort_keys=True).encode())
    return h.hexdigest()


def prepare_stats(cfg,bundle,stored=None,force=False):
    identity=stats_identity(bundle.metadata)
    if stored is not None:
        if stored.get('identity')!=identity: raise ValueError('Checkpoint data/split/preprocessing/statistics mismatch')
        return stored
    path=Path(cfg['fourier']['cache_dir'])/(identity+'.pt')
    if path.exists() and not force:
        obj=load_checkpoint(path)
        if obj.get('identity')!=identity: raise ValueError('Corrupted statistics cache identity')
        return obj
    a=cfg['data_loader']['args']
    loader=DataLoader(bundle.train,batch_size=cfg['fourier']['stats_batch_size'],shuffle=False,
                      num_workers=a['num_workers'],multiprocessing_context='spawn' if a['num_workers'] else None,
                      generator=torch.Generator().manual_seed(a['split_seed']))
    def batches():
        for x in loader:
            yield x
            if a['random_flip']: yield x.flip(-1)
    stats=estimate_stats(batches(),cfg['fourier']['power_floor'])
    stats.update(bundle.metadata); stats['identity']=identity
    atomic_save(stats,path)
    return stats
