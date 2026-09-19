"""Same NCSN++ backbone and initialization for every objective.

Original deterministic sigma division is moved into the objective adapter;
no convolution, attention, normalization, resblock or learned weight changes.
"""
from __future__ import annotations
from types import SimpleNamespace
import copy
import hashlib
import json
import torch
from base.base_model import BaseModel
from model.backbones.ncsnpp import NCSNpp
from model.spectral import FourierGaussian
from sde.process import NoiseProcess


def backbone_config(cfg):
    a=copy.deepcopy(cfg['arch']['args']); p=cfg['process']; d=cfg['data_loader']['args']
    a.update(scale_by_sigma=False,sigma_min=p['sigma_min'],sigma_max=p['sigma_max'],num_scales=p['num_scales'])
    # Fourier embedding accepts real sigma even with a discrete DDPM schedule.
    return SimpleNamespace(model=SimpleNamespace(**a),data=SimpleNamespace(image_size=d['image_size'],num_channels=d['channels'],centered=d['centered']),training=SimpleNamespace(continuous=True))


def architecture_report(backbone):
    shapes={n:{'shape':list(p.shape),'trainable':p.requires_grad} for n,p in backbone.named_parameters()}
    modules={n:type(m).__name__ for n,m in backbone.named_modules()}
    serial=json.dumps({'parameters':shapes,'modules':modules},sort_keys=True)
    return {'trainable_parameters':sum(p.numel() for p in backbone.parameters() if p.requires_grad),
            'all_parameters':sum(p.numel() for p in backbone.parameters()),
            'architecture_sha256':hashlib.sha256(serial.encode()).hexdigest(),
            'parameter_shapes':shapes,'modules':modules}


def weight_hash(backbone):
    h=hashlib.sha256()
    for n,p in backbone.named_parameters():
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class GenerativeModel(BaseModel):
    def __init__(self,cfg,stats=None):
        super().__init__()
        self.cfg=copy.deepcopy(cfg)
        self.process=NoiseProcess(cfg['process'])
        self.objective=cfg['loss']['type']
        self.embedding=cfg['arch']['args']['embedding_type']
        # Source creates one unused float64 sigma buffer. Convert on CPU FIRST,
        # before transfer to MPS, where float64 is not used by this project.
        self.backbone=NCSNpp(backbone_config(cfg)).float()
        self.reference=None
        if self.objective=='fourier_gaussian':
            if stats is None: raise ValueError('fourier_gaussian requires training-only statistics')
            self.reference=FourierGaussian(stats,cfg['backend']['spectral_transform'])

    def scaled_from_raw(self,raw,y,level):
        if self.objective=='score': return raw
        if self.objective=='diffusion': return -raw
        return self.reference.scaled_score(raw,y,level.alpha,level.sigma)

    def scaled_score(self,y,level):
        raw=self.backbone(y,self.process.condition(level,self.embedding))
        return self.scaled_from_raw(raw,y,level)

    def forward(self,y,level):
        return self.scaled_score(y,level)/level.sigma[:,None,None,None]

    def denoise(self,y,level):
        return (y+level.sigma[:,None,None,None]*self.scaled_score(y,level))/level.alpha[:,None,None,None]


def build_model(cfg,stats=None,device='cpu'):
    return GenerativeModel(cfg,stats).to(device=device,dtype=torch.float32)
