"""Checkpointed cursor counts CONSUMED batches, not prefetched batches."""
from __future__ import annotations
import math
import torch
from torch.utils.data import DataLoader, Sampler

class CursorBatchSampler(Sampler):
    def __init__(self,size,batch_size,seed):
        if min(size,batch_size)<1: raise ValueError('Empty training data')
        self.size=size; self.batch_size=batch_size; self.seed=seed
        self.epoch=0; self.batch=0

    def __len__(self): return math.ceil(self.size/self.batch_size)-self.batch

    def __iter__(self):
        # Snapshot cursor: worker prefetch never mutates checkpoint state.
        epoch,start=self.epoch,self.batch
        ids=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
        for i in range(start,math.ceil(self.size/self.batch_size)):
            yield ids[i*self.batch_size:(i+1)*self.batch_size]

    def advance(self):
        self.batch+=1
        if self.batch==math.ceil(self.size/self.batch_size): self.epoch+=1; self.batch=0

    def state_dict(self): return {'epoch':self.epoch,'batch':self.batch,'size':self.size,'batch_size':self.batch_size,'seed':self.seed}

    def load_state_dict(self,state):
        if any(state[k]!=getattr(self,k) for k in ('size','batch_size','seed')): raise ValueError('Data cursor mismatch')
        self.epoch=state['epoch']; self.batch=state['batch']
        if self.epoch<0 or not 0<=self.batch<math.ceil(self.size/self.batch_size): raise ValueError('Invalid cursor')


class BaseDataLoader:
    def __init__(self,dataset,batch_size,seed,num_workers=0,pin_memory=False):
        self.sampler=CursorBatchSampler(len(dataset),batch_size,seed)
        self.loader=DataLoader(dataset,batch_sampler=self.sampler,num_workers=num_workers,
            pin_memory=pin_memory,persistent_workers=num_workers>0,
            multiprocessing_context='spawn' if num_workers else None,
            generator=torch.Generator().manual_seed(seed+10_000_000))
        self.iterator=None

    def next_batch(self):
        if self.iterator is None: self.iterator=iter(self.loader)
        try: return next(self.iterator)
        except StopIteration:
            self.iterator=iter(self.loader)
            return next(self.iterator)

    def advance(self): self.sampler.advance()
    def state_dict(self): return self.sampler.state_dict()
    def load_state_dict(self,state): self.sampler.load_state_dict(state); self.iterator=None
