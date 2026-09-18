"""Generate EMA samples into bounded-memory uint8 NHWC NPZ shards."""
import argparse
from pathlib import Path
import math
import numpy as np
import torch
from sde.sampling import sample_batch
from utils.inference import load_model
from utils.util import seed_all, json_write, file_hash
from parse_config import plain


class BatchedScore(torch.nn.Module):
    """Split model forwards ONLY; preserve Langevin's effective-batch norms."""
    def __init__(self, model, batch_size):
        super().__init__()
        self.model, self.batch_size = model, batch_size

    def forward(self, x, labels):
        return torch.cat([self.model(a, b) for a, b in zip(x.split(self.batch_size), labels.split(self.batch_size))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-r', '--checkpoint', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--num-samples', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=64, help='Effective sampler batch, not model forward batch')
    p.add_argument('--model-batch-size', type=int, default=None)
    p.add_argument('--steps', type=int, help='Continuous sampler grid only; discrete grid cannot change')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--sampler', choices=('pc', 'ode'))
    p.add_argument('--corrector', choices=('none', 'langevin', 'ald'))
    a = p.parse_args()
    if min(a.num_samples, a.batch_size, a.model_batch_size or a.batch_size) < 1:
        p.error('Sample and batch counts must be positive')
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty: {out}')
    model, cfg, ck, device = load_model(a.checkpoint, a.device)
    if a.sampler:
        cfg.sampling.method = a.sampler
    if a.corrector:
        cfg.sampling.corrector = a.corrector
    if not cfg.training.continuous and a.steps not in (None, cfg.model.num_scales):
        p.error('Discrete sampling must keep the trained label grid')
    model = BatchedScore(model, a.model_batch_size or a.batch_size).eval()
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(checkpoint_sha256=file_hash(a.checkpoint), step=ck['step'], weights='EMA',
                dataset=cfg.data.dataset, reference=cfg.reference.mode, num_samples=a.num_samples,
                effective_batch_size=a.batch_size, model_batch_size=a.model_batch_size or a.batch_size,
                sampling_steps=a.steps or cfg.model.num_scales, sampler=plain(cfg.sampling),
                seed=a.seed, round_seed='seed + shard_index', device=str(device),
                quantization='clip(x*255,0,255).astype(uint8), NHWC', complete=False)
    del ck
    json_write(meta, out / 'settings.json')
    written, total_nfe = 0, 0
    for index in range(math.ceil(a.num_samples / a.batch_size)):
        seed_all(a.seed + index)
        # Always sample a FULL batch, then truncate exports, matching the source
        # evaluator's batch-coupled Langevin behavior even on the final round.
        shape = (a.batch_size, cfg.data.num_channels, cfg.data.image_size, cfg.data.image_size)
        images, nfe = sample_batch(model, cfg, shape, device, a.steps)
        keep = min(a.batch_size, a.num_samples - written)
        images = images[:keep]
        array = np.clip(images.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        np.savez_compressed(out / f'samples_{index:05d}.npz', samples=array)
        if index == 0:
            from torchvision.utils import save_image
            save_image(images[:64].cpu().clamp(0, 1), out / 'preview.png', nrow=8)
        written += keep
        total_nfe += nfe
        print(f'samples={written}/{a.num_samples}, shard_NFE={nfe}', flush=True)
    meta.update(complete=True, exported_samples=written, score_calls_total=total_nfe)
    json_write(meta, out / 'settings.json')


if __name__ == '__main__':
    main()
