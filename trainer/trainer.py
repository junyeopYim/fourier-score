"""One trainer for continuous VE/VP and discrete VE/SMLD or VP/DDPM."""
import time
import torch
from base.base_trainer import BaseTrainer
from base.base_data_loader import BatchStream
from data_loader.data_loaders import build_dataset, prepare_stats, augment
from logger.logger import ExperimentLogger
from model.model import build_model
from model.loss import build_loss
from model.metric import evaluate_dsm
from sde.sde_lib import make_sde
from utils.util import seed_all, configure_runtime


def evaluation_source(cfg, dataset, stats):
    if len(stats['val_indices']):
        return dataset, stats['val_indices'], 'validation'
    if cfg.data.dataset.lower() == 'ffhq' or cfg.data_loader.kind == 'image_folder':
        # Explicit diagnostic: NOT a held-out generalization measurement.
        return dataset, stats['train_indices'], 'train_diagnostic'
    other = build_dataset(cfg, 'validation' if cfg.data_loader.kind == 'faces' else 'test')
    return other, torch.arange(len(other)), 'validation' if cfg.data_loader.kind == 'faces' else 'test'


class Trainer(BaseTrainer):
    def __init__(self, cfg, checkpoint=None):
        device = configure_runtime(cfg)
        dataset = build_dataset(cfg)
        try:
            stats = prepare_stats(cfg, dataset, checkpoint['stats'] if checkpoint else None)
            eval_data, eval_ids, split = evaluation_source(cfg, dataset, stats)
            seed_all(cfg.seed)
            model = build_model(cfg, stats, device)
            stream = BatchStream(stats['train_indices'], cfg.training.batch_size, cfg.seed + 1000)
            super().__init__(cfg, model, stats, stream, device, checkpoint)
        except BaseException:
            dataset.close()
            if 'eval_data' in locals() and eval_data is not dataset:
                eval_data.close()
            raise
        self.dataset, self.eval_data = dataset, eval_data
        self.eval_ids, self.eval_split = eval_ids, split
        self.sde = make_sde(cfg)
        self.loss_fn = build_loss(cfg, self.sde, train=True)
        self.log = ExperimentLogger(self.out, cfg.trainer.tensorboard)

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        total = batch.new_zeros(())
        for part in batch.split(self.cfg.trainer.microbatch_size or len(batch)):
            loss = self.loss_fn(self.model, part) * (len(part) / len(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {self.step}')
            loss.backward()
            total += loss.detach()
        self.optimize()
        return float(total)

    def evaluate(self):
        n = min(len(self.eval_ids), self.cfg.eval.max_images)
        if self.cfg.data_loader.protocol == 'full_train':
            begin = (self.step // self.cfg.training.eval_freq) * n
            ids = self.eval_ids[(torch.arange(n) + begin) % len(self.eval_ids)]
        else:
            ids = self.eval_ids[:n]
        with self.ema.average_parameters(self.model):
            result = evaluate_dsm(self.model, self.cfg, self.sde, self.eval_data, ids, self.device)
        self.log.write(dict(result, step=self.step, split=self.eval_split, weights='EMA'))

    def train(self):
        cfg = self.cfg
        start = time.perf_counter()
        try:
            if cfg.training.n_iters <= self.step:
                raise ValueError('training.n_iters must exceed the checkpoint step')
            if self.step == 0:
                self.evaluate()
            while self.step < cfg.training.n_iters:
                x = augment(self.dataset.batch(self.stream.next()), cfg).to(self.device)
                loss = self.train_step(x)
                final = self.step == cfg.training.n_iters
                if self.step % cfg.training.log_freq == 0 or final:
                    self.log.write(dict(step=self.step, split='train', dsm_mean=loss,
                                        lr=self.optimizer.param_groups[0]['lr'],
                                        session_wall_seconds=time.perf_counter() - start))
                if self.step % cfg.training.eval_freq == 0 or final:
                    self.evaluate()
                snapshot = self.step % cfg.training.snapshot_freq == 0 or final
                if snapshot or self.step % cfg.training.snapshot_freq_for_preemption == 0:
                    self.save(snapshot)
        finally:
            self.log.close()
            self.dataset.close()
            if self.eval_data is not self.dataset:
                self.eval_data.close()
