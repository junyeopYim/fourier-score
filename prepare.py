"""Download/verify MNIST or CIFAR-10 and fit training-only Gaussian statistics."""
import argparse
from parse_config import ConfigParser, config_arguments
from data_loader.data_loaders import build_dataset, prepare_stats
from utils.util import configure_runtime


def main():
    p = config_arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--download', action='store_true', help='Enable torchvision download of both official splits')
    a = p.parse_args()
    cfg = ConfigParser.from_file(a.config, a.set).config
    configure_runtime(cfg)
    if a.download:
        if cfg.data_loader.kind != 'torchvision':
            p.error('--download is for MNIST/CIFAR10; prepare face sources separately')
        cfg.data_loader.download = True
        test = build_dataset(cfg, 'test')
        test.close()
    dataset = build_dataset(cfg)
    try:
        stats = prepare_stats(cfg, dataset)
        print(f"Training statistics ready: train={stats['n_train']}, "
              f"validation={len(stats['val_indices'])}, moments={stats['n_stats_images']}")
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
