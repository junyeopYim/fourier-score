"""One step-based trainer for all three objectives; no objective-specific loop."""
import time
import torch
from base.base_trainer import BaseTrainer
from base.base_data_loader import BaseDataLoader
from data_loader.data_loaders import build_data,prepare_stats
from model.model import build_model
from model.loss import training_loss
from model.metric import evaluate_dsm
from utils.util import configure_runtime,seed_all
from logger.logger import ExperimentLogger

class Trainer(BaseTrainer):
    def __init__(self,cfg,checkpoint=None):
        device=configure_runtime(cfg)
        self.bundle=build_data(cfg)
        # All arms use the same split/cache identity. Computing statistics does
        # not leak validation data and happens before the model RNG is seeded.
        stats=prepare_stats(cfg,self.bundle,checkpoint['stats'] if checkpoint else None)
        seed_all(cfg['seed'],device)
        model=build_model(cfg,stats,device)
        a=cfg['data_loader']['args']
        stream=BaseDataLoader(self.bundle.train,a['batch_size'],cfg['seed']+1000,a['num_workers'],device.type=='cuda')
        super().__init__(cfg,model,stats,stream,device,checkpoint)
        self.log=ExperimentLogger(self.out,cfg['trainer']['tensorboard'])

    def train_step(self,clean):
        self.model.train()
        if self.cfg['data_loader']['args']['random_flip']:
            flip=torch.rand(len(clean),generator=self.generator)<0.5
            clean=torch.where(flip[:,None,None,None],clean.flip(-1),clean)
        lev=self.model.process.sample(len(clean),self.device,self.generator)
        noise=torch.randn(clean.shape,generator=self.generator).to(self.device)
        clean=clean.to(self.device)
        micro=self.cfg['trainer']['microbatch_size'] or len(clean)
        self.optimizer.zero_grad(set_to_none=True); total=0.
        for start in range(0,len(clean),micro):
            end=min(start+micro,len(clean))
            loss=training_loss(self.model,clean[start:end],lev.slice(start,end),noise[start:end],self.cfg['loss']['reduction'])
            loss=loss*((end-start)/len(clean))
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite training loss at step {self.step}')
            loss.backward(); total+=float(loss.detach())
        grad_norm=self.optimize()
        return total,grad_norm

    def evaluate(self):
        with self.ema.average_parameters(self.model):
            result=evaluate_dsm(self.model,self.cfg,self.bundle.validation,self.device)
        result.update(step=self.step,split=self.bundle.metadata['eval_split'],weights='EMA')
        self.log.write(result)
        return result

    def train(self):
        t=self.cfg['trainer']; begin=time.perf_counter()
        if self.step>=t['iterations']:
            self.log.close(); raise ValueError('iterations must exceed the checkpoint step')
        try:
            if self.step==0: self.evaluate()
            while self.step<t['iterations']:
                loss,norm=self.train_step(self.stream.next_batch())
                final=self.step==t['iterations']
                if final or self.step%t['log_every']==0:
                    self.log.write({'step':self.step,'split':'train','loss':loss,'reduction':self.cfg['loss']['reduction'],
                                    'grad_norm_before_clip':norm,'lr':self.optimizer.param_groups[0]['lr'],
                                    'session_wall_seconds':time.perf_counter()-begin})
                if final or self.step%t['eval_every']==0: self.evaluate()
                snapshot=final or self.step%t['snapshot_every']==0
                if snapshot or self.step%t['save_every']==0: self.save(snapshot)
        finally: self.log.close()
