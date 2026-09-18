"""Indexed data access, train-only statistics and explicit split contracts."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from model.gaussian import estimate_stats, to_model_space
from utils.util import atomic_save, load_checkpoint, fingerprint


class TensorImages:
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        if cfg.data_loader.kind == 'synthetic':
            g = torch.Generator().manual_seed(12345 + (split != 'train'))
            n = cfg.data_loader.synthetic_size if split == 'train' else 16
            self.raw = torch.randint(0, 256, (n, 28, 28) if cfg.data.dataset.lower() == 'mnist'
                                     else (n, cfg.data.num_channels, cfg.data.image_size, cfg.data.image_size),
                                     dtype=torch.uint8, generator=g)
        else:
            from torchvision.datasets import MNIST, CIFAR10
            name = cfg.data.dataset.lower()
            if name not in ('mnist', 'cifar10'):
                raise ValueError('torchvision raw loader supports MNIST or CIFAR10')
            cls = MNIST if name == 'mnist' else CIFAR10
            ds = cls(cfg.data_loader.data_dir, train=split == 'train', download=cfg.data_loader.download)
            self.raw = (torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
                        if isinstance(ds.data, np.ndarray) else ds.data)
        digest = hashlib.sha256(memoryview(self.raw.numpy())).hexdigest()
        self.metadata = {'kind': cfg.data_loader.kind, 'dataset': cfg.data.dataset,
                         'split': split, 'raw_sha256': digest, 'count': len(self.raw)}
        self.fingerprint = fingerprint(self.metadata)

    def __len__(self):
        return len(self.raw)

    def batch(self, ids):
        if self.cfg.data.dataset.lower() in ('mnist', 'cifar10'):
            return to_model_space(self.raw[ids], self.cfg.data.dataset.lower(), self.cfg.data.centered)
        x = self.raw[ids].float() / 255.
        return 2 * x - 1 if self.cfg.data.centered else x

    def close(self):
        pass


class ImageFolder:
    """Already-preprocessed fixed-size RGB images; NO silent crop or resize.

    This is a custom dataset adapter, NOT an assertion of official LSUN/FFHQ
    preprocessing equivalence. Prepare the exact comparison images externally.
    """
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        root = Path(cfg.data_loader.data_dir) / split
        self.files = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
        if not self.files:
            raise FileNotFoundError(f'No images under {root}')
        self.metadata = {'preprocess': 'preprocessed_exact_size_rgb_v1', 'split': split,
                         'files': [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in self.files]}
        self.fingerprint = fingerprint(self.metadata)

    def __len__(self):
        return len(self.files)

    def batch(self, ids):
        from PIL import Image
        xs = []
        for i in ids:
            with Image.open(self.files[int(i)]) as im:
                im = im.convert('RGB')
                if im.size != (self.cfg.data.image_size, self.cfg.data.image_size):
                    raise ValueError(f'{self.files[int(i)]}: preprocess to the configured size first')
                xs.append(torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1))
        x = torch.stack(xs).float() / 255.
        return 2 * x - 1 if self.cfg.data.centered else x

    def close(self):
        pass


def build_dataset(cfg, split='train'):
    kind = cfg.data_loader.kind
    if kind == 'faces':
        from .face_data import FaceDataset
        return FaceDataset(cfg, split)
    if kind == 'image_folder':
        return ImageFolder(cfg, split)
    return TensorImages(cfg, split)


def split_indices(cfg, n):
    if cfg.data_loader.protocol == 'full_train':
        return torch.arange(n), torch.empty(0, dtype=torch.long)
    n_val = cfg.data_loader.validation_size
    if not 0 < n_val < n:
        raise ValueError('validation_size must be between zero and dataset size')
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.data_loader.split_seed))
    return perm[:-n_val], perm[-n_val:]


def stats_contract(cfg, dataset):
    return dict(dataset=cfg.data.dataset, image_size=cfg.data.image_size,
                num_channels=cfg.data.num_channels, centered=cfg.data.centered,
                random_flip=cfg.data.random_flip, floor=cfg.reference.floor,
                protocol=cfg.data_loader.protocol, validation_size=cfg.data_loader.validation_size,
                split_seed=cfg.data_loader.split_seed, data_fingerprint=dataset.fingerprint,
                preprocessing='score_sde_template_v1')


def prepare_stats(cfg, dataset, checkpoint_stats=None):
    contract = stats_contract(cfg, dataset)
    train_ids, val_ids = split_indices(cfg, len(dataset))
    path = Path(cfg.reference.stats_path) if cfg.reference.stats_path else (
        Path(cfg.data_loader.cache_dir) / f'{cfg.data.dataset.lower()}_{fingerprint(contract)[:16]}_gaussian.pt')
    if checkpoint_stats is not None:
        stats = checkpoint_stats
    elif path.exists():
        stats = load_checkpoint(path)
    else:
        def batches():
            for ids in train_ids.split(cfg.data_loader.stats_batch_size):
                x = dataset.batch(ids)
                yield x
                if cfg.data.random_flip:
                    yield x.flip(-1)
        stats = estimate_stats(batches(), cfg.reference.floor)
        stats.update(n_stats_images=stats['n_train'], n_train=len(train_ids),
                     train_indices=train_ids, val_indices=val_ids, contract=contract)
        atomic_save(stats, path)
    if stats.get('contract') != contract:
        raise ValueError('Statistics/data/preprocessing/split mismatch; use a new cache or the original data')
    if not torch.equal(stats['train_indices'].cpu(), train_ids) or not torch.equal(stats['val_indices'].cpu(), val_ids):
        raise ValueError('Statistics split mismatch')
    expected = (cfg.data.num_channels, cfg.data.image_size, cfg.data.image_size)
    if tuple(stats['mean'].shape) != expected or tuple(stats['power'].shape) != expected:
        raise ValueError('Statistics shape does not match model')
    return stats


def augment(batch, cfg):
    if cfg.data.random_flip:
        flip = torch.rand(len(batch)) < .5
        batch[flip] = batch[flip].flip(-1)
    return batch
