"""Shared optimizer/EMA/checkpoint lifecycle for step-based trainers."""
from pathlib import Path
import copy
import platform
import torch
from parse_config import plain
from utils.ema import ExponentialMovingAverage
from utils.util import atomic_save, rng_state, restore_rng, source_hash, json_write

FORMAT_VERSION = 1


def resume_signature(config):
    data = copy.deepcopy(plain(config))
    # These alter logging/output or the final stopping point, not updates.
    for key in ('name', 'device', 'eval', 'provenance'):
        data.pop(key, None)
    for key in ('n_iters', 'log_freq', 'eval_freq', 'snapshot_freq', 'snapshot_freq_for_preemption'):
        data['training'].pop(key, None)
    data['trainer'].pop('save_dir', None)
    data['trainer'].pop('tensorboard', None)
    data['reference'].pop('stats_path', None)
    for key in ('data_dir', 'cache_dir', 'download'):
        data['data_loader'].pop(key, None)
    # Dataset identity/preprocessing are additionally validated against stats.
    return data


class BaseTrainer:
    def __init__(self, cfg, model, stats, stream, device, checkpoint=None):
        self.cfg, self.model, self.stats = cfg, model, stats
        self.stream, self.device = stream, device
        self.step = 0
        self.out = Path(cfg.trainer.save_dir) / cfg.name
        self.out.mkdir(parents=True, exist_ok=True)
        if checkpoint is None and (self.out / 'config.json').exists():
            raise FileExistsError(f'Run already exists: {self.out}; use --resume or a new name')
        if cfg.optim.optimizer != 'Adam':
            raise ValueError('Only source Adam recipe is retained')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr,
                                         betas=(cfg.optim.beta1, .999), eps=cfg.optim.eps,
                                         weight_decay=cfg.optim.weight_decay)
        self.ema = ExponentialMovingAverage(model.parameters(), cfg.model.ema_rate)
        self.code_hash = source_hash()
        if checkpoint is not None:
            if checkpoint.get('format_version') != FORMAT_VERSION:
                raise ValueError('Expected a template v1 checkpoint; legacy checkpoints need explicit conversion')
            if checkpoint['resume_signature'] != resume_signature(cfg):
                raise ValueError('Resume configuration mismatch (model/loss/data/optimizer/batch/backend)')
            if checkpoint['source_sha256'] != self.code_hash:
                raise ValueError('Source code differs from checkpoint; refuse silent cross-version resume')
            model.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.ema.load_state_dict(checkpoint['ema'])
            self.stream.load_state_dict(checkpoint['stream'])
            self.step = checkpoint['step']
            restore_rng(checkpoint['rng'])
        json_write(plain(cfg), self.out / 'config.json')
        json_write({'torch': str(torch.__version__), 'python': platform.python_version(),
                    'device': str(device), 'cuda': torch.version.cuda,
                    'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                    'source_sha256': self.code_hash, 'tf32': False, 'amp': False,
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)},
                   self.out / 'environment.json')

    def optimize(self):
        # Preserve the source's pre-increment warmup: the first LR is zero.
        if self.cfg.optim.warmup > 0:
            lr = self.cfg.optim.lr * min(self.step / self.cfg.optim.warmup, 1.)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
        if self.cfg.optim.grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip,
                                         error_if_nonfinite=True)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise FloatingPointError('Non-finite gradients')
        self.optimizer.step()
        self.step += 1
        self.ema.update(self.model.parameters())

    def save(self, snapshot=False):
        state = dict(format_version=FORMAT_VERSION, model=self.model.state_dict(),
                     optimizer=self.optimizer.state_dict(), ema=self.ema.state_dict(),
                     step=self.step, stats=self.stats, stream=self.stream.state_dict(),
                     rng=rng_state(), config=plain(self.cfg),
                     resume_signature=resume_signature(self.cfg), source_sha256=self.code_hash)
        atomic_save(state, self.out / 'last.pt')
        if snapshot:
            # Lightweight EMA-only snapshot, not resumable.
            with self.ema.average_parameters(self.model):
                atomic_save(dict(format_version=FORMAT_VERSION, weights='EMA',
                                 model=self.model.state_dict(), step=self.step,
                                 stats=self.stats, config=plain(self.cfg), source_sha256=self.code_hash),
                            self.out / f'ema_step_{self.step:07d}.pt')
