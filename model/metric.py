"""DSM evaluation with isolated random draws; no claim of FID/IS equivalence."""
import torch
from model.loss import build_loss
from utils.util import isolated_rng


@torch.no_grad()
def evaluate_dsm(model, cfg, sde, dataset, ids, device):
    if len(ids) == 0:
        raise ValueError('Empty evaluation split')
    old = model.training
    total, n = 0., 0
    try:
        # Every arm with the same config uses the same t/noise sequence.
        with isolated_rng(cfg.eval.seed):
            loss_fn = build_loss(cfg, sde, train=False)
            for part in ids.split(cfg.eval.batch_size):
                x = dataset.batch(part).to(device)
                loss = loss_fn(model, x)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite evaluation loss')
                total += float(loss) * len(part)
                n += len(part)
    finally:
        model.train(old)
    return {'dsm_mean': total / n, 'n_images': n,
            'reduction': 'pixel_mean' if cfg.training.reduce_mean else 'half_pixel_sum',
            'eval_seed': cfg.eval.seed, 'eval_batch_size': cfg.eval.batch_size}
