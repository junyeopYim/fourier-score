"""Safe local checkpoint IO and reproducible runtime helpers."""
import contextlib
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(cfg):
    if cfg.backend.precision != 'fp32' or cfg.backend.tf32:
        raise ValueError('Only strict FP32 is enabled')
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else torch.device(cfg.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: cpu, cuda, cuda:N, auto')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this PyTorch installation')
    if device.type == 'cpu' and cfg.backend.cpu_threads:
        torch.set_num_threads(cfg.backend.cpu_threads)
    return device


def rng_state():
    s = np.random.get_state()
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(),
            'numpy': [s[0], s[1].tolist(), s[2], s[3], s[4]]}


def restore_rng(s):
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] and torch.cuda.is_available():
        if len(s['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from checkpoint; exact resume is not supported')
        torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])
    random.setstate(s['python'])
    n = s['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


@contextlib.contextmanager
def isolated_rng(seed):
    state = rng_state()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(state)


def _plain_scalars(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _plain_scalars(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_plain_scalars(x) for x in v)
    return v


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(_plain_scalars(obj), temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_checkpoint(path):
    # No arbitrary pickle globals and no unsafe fallback.
    return torch.load(path, map_location='cpu', weights_only=True)


def json_write(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        if any(p in ('.venv', 'saved', '__pycache__') for p in path.relative_to(root).parts):
            continue
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()
