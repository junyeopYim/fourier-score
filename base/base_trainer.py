"""Checkpoint / optimizer / EMA lifecycle, template-style separation."""
from __future__ import annotations
import copy
import time
from pathlib import Path
import torch
from parse_config import experiment_name
from utils.ema import EMA
from utils.util import atomic_save,capture_rng,restore_rng,environment,json_write,source_hash
from model.model import architecture_report,weight_hash

FORMAT='fourier-image-template-v1'


def resume_signature(cfg):
    c=copy.deepcopy(cfg)
    for k in ('name','device','evaluation','sampling'): c.pop(k)
    for k in ('iterations','save_dir','save_every','snapshot_every','log_every','eval_every','tensorboard'): c['trainer'].pop(k)
    for k in ('console','progress_every_seconds'): c['trainer'].pop(k,None)
    for k in ('root','download','num_workers'): c['data_loader']['args'].pop(k)
    c['fourier'].pop('cache_dir'); c['fourier'].pop('stats_batch_size')
    return c

class BaseTrainer:
    def __init__(self,cfg,model,stats,stream,device,checkpoint=None,session_started=None):
        self.cfg=cfg; self.model=model; self.stats=stats; self.stream=stream; self.device=device; self.step=0
        self._session_started=time.perf_counter() if session_started is None else session_started
        self._previous_wall_seconds=checkpoint.get('training_wall_seconds',0.) if checkpoint else 0.
        self.out=Path(cfg['trainer']['save_dir'])/experiment_name(cfg)
        if checkpoint is None and self.out.exists() and any(self.out.iterdir()): raise FileExistsError(f'Run exists: {self.out}; use --resume or a new name')
        self.out.mkdir(parents=True,exist_ok=True)
        o=dict(cfg['optimizer']['args']); o['betas']=tuple(o['betas'])
        self.optimizer=torch.optim.Adam(model.parameters(),**o)
        self.ema=EMA(model,cfg['trainer']['ema_decay'])
        self.generator=torch.Generator().manual_seed(cfg['seed']+2000)
        self.env=environment(device,cfg)
        self.initial_hash=weight_hash(model.backbone)
        if checkpoint is not None:
            if checkpoint.get('format')!=FORMAT or checkpoint.get('kind')!='training': raise ValueError('Only v1 training checkpoints can resume')
            if checkpoint['signature']!=resume_signature(cfg): raise ValueError('Resume config mismatch: model/loss/process/data/optimizer/backend must match')
            if checkpoint['source_sha256']!=source_hash(): raise ValueError('Source differs from checkpoint; no silent cross-version resume')
            old=checkpoint['environment']
            if old['device']!=str(device) or old['torch']!=str(torch.__version__): raise ValueError('Resume requires same device type/index and PyTorch version; start a new explicit fine-tuning experiment instead')
            model.load_state_dict(checkpoint['model'],strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.ema.load_state_dict(checkpoint['ema'],model)
            self.stream.load_state_dict(checkpoint['stream'])
            self.generator.set_state(checkpoint['generator'].cpu())
            self.step=checkpoint['step']; self.initial_hash=checkpoint['initial_backbone_sha256']
            restore_rng(checkpoint['rng'],device)
        report=architecture_report(model.backbone); report['initial_backbone_sha256']=self.initial_hash
        if model.reference is not None: self.env['spectral_transform_resolved']=model.reference.resolved_backend(device)
        json_write(cfg,self.out/'config.resolved.json'); json_write(self.env,self.out/'environment.json'); json_write(report,self.out/'architecture.json')

    def training_wall_seconds(self):
        """Cumulative active-session time, including setup, evaluation and prior saves.

        Excludes downtime between sessions and the write of this checkpoint.
        Synchronize only at reporting/checkpoint boundaries, not every update.
        """
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        elif self.device.type=='mps': torch.mps.synchronize()
        return self._previous_wall_seconds+time.perf_counter()-self._session_started

    def optimize(self):
        warmup=self.cfg['trainer']['warmup']
        lr=self.cfg['optimizer']['args']['lr']*(min(self.step/warmup,1.) if warmup else 1.)
        for group in self.optimizer.param_groups: group['lr']=lr
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg['trainer']['grad_clip'],error_if_nonfinite=True,foreach=False)
        self.optimizer.step(); self.step+=1; self.ema.update(self.model); self.stream.advance()
        return float(norm)

    def save(self,snapshot=False):
        elapsed=self.training_wall_seconds()
        state={'format':FORMAT,'kind':'training','config':self.cfg,'step':self.step,'model':self.model.state_dict(),
               'optimizer':self.optimizer.state_dict(),'ema':self.ema.state_dict(),'stats':self.stats,
               'stream':self.stream.state_dict(),'generator':self.generator.get_state(),'rng':capture_rng(self.device),
               'signature':resume_signature(self.cfg),'source_sha256':source_hash(),'environment':self.env,
               'initial_backbone_sha256':self.initial_hash,'training_wall_seconds':elapsed}
        atomic_save(state,self.out/'last.pt')
        if snapshot:
            with self.ema.average_parameters(self.model):
                atomic_save({'format':FORMAT,'kind':'ema','config':self.cfg,'step':self.step,'model':self.model.state_dict(),
                             'stats':self.stats,'source_sha256':source_hash(),'environment':self.env,
                             'training_wall_seconds':elapsed},self.out/f'ema_{self.step:09d}.pt')
