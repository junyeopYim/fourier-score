"""Evaluate checkpoint EMA DSM; FID/IS is an optional separate metrics.py tool."""
import argparse
from data_loader.data_loaders import build_dataset, prepare_stats
from trainer.trainer import evaluation_source
from model.metric import evaluate_dsm
from sde.sde_lib import make_sde
from utils.inference import load_model
from utils.util import json_write, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-r', '--checkpoint', required=True)
    p.add_argument('-o', '--output', default='evaluation.json')
    p.add_argument('--device', default='auto')
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--data-dir', default=None)
    p.add_argument('--cache-dir', default=None)
    a = p.parse_args()
    model, cfg, ck, device = load_model(a.checkpoint, a.device)
    if a.max_images is not None:
        if a.max_images < 1:
            p.error('--max-images must be positive')
        cfg.eval.max_images = a.max_images
    if a.batch_size is not None:
        if a.batch_size < 1:
            p.error('--batch-size must be positive')
        cfg.eval.batch_size = a.batch_size
    if a.data_dir:
        cfg.data_loader.data_dir = a.data_dir
    if a.cache_dir:
        cfg.data_loader.cache_dir = a.cache_dir
    dataset = build_dataset(cfg)
    other = None
    try:
        stats = prepare_stats(cfg, dataset, ck['stats'])
        other, ids, split = evaluation_source(cfg, dataset, stats)
        result = evaluate_dsm(model, cfg, make_sde(cfg), other, ids[:cfg.eval.max_images], device)
        result.update(step=ck['step'], weights='EMA', split=split, reference=cfg.reference.mode,
                      dataset=cfg.data.dataset, checkpoint_sha256=file_hash(a.checkpoint))
        json_write(result, a.output)
        print(result)
    finally:
        dataset.close()
        if other is not None and other is not dataset:
            other.close()


if __name__ == '__main__':
    main()
