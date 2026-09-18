from pathlib import Path
import copy
import json
import random
import numpy as np
import pytest
import torch
from parse_config import ConfigParser, plain
from trainer.trainer import Trainer
from base.base_data_loader import BatchStream
from data_loader.data_loaders import build_dataset, prepare_stats
from data_loader.face_data import record_index
from utils.util import load_checkpoint, isolated_rng, seed_all, rng_state
from utils.inference import load_model
from utils.ema import ExponentialMovingAverage


@pytest.mark.parametrize('path', sorted(Path('configs').glob('*.json')))
def test_configs_load(path):
    cfg = ConfigParser.from_file(path).config
    assert cfg.model.name == 'ncsnpp'


def test_numerical_presets():
    cifar = ConfigParser.from_file('configs/cifar10_upstream.json').config
    paper = ConfigParser.from_file('configs/cifar10_paper.json').config
    celeba = ConfigParser.from_file('configs/celeba64.json').config
    ffhq = ConfigParser.from_file('configs/ffhq256.json').config
    assert (cifar.training.n_iters, paper.training.n_iters) == (1300001,950000)
    assert cifar.model.nf == 128 and not cifar.training.reduce_mean and not cifar.data.centered
    assert not celeba.training.continuous and celeba.model.sigma_max == 90
    assert (ffhq.training.n_iters, ffhq.training.batch_size, ffhq.model.num_scales) == (2400001,64,2000)
    assert ffhq.model.ch_mult == [1,1,2,2,2,2,2] and ffhq.model.sigma_max == 348


@pytest.mark.parametrize('override', ['model.unknown=32','training.batch_size=0','reference.mode=unknown','backend.tf32=true','model.sigma_min=-1'])
def test_invalid_configs(override):
    with pytest.raises((ValueError, KeyError)):
        ConfigParser.from_file('configs/smoke.json', [override])


def test_config_cycle(tmp_path):
    path = tmp_path/'cycle.json'
    path.write_text('{"extends":"cycle.json"}')
    with pytest.raises(ValueError): ConfigParser.from_file(path)


def test_stream_exact_resume_and_tail():
    a = BatchStream(torch.arange(11), 4, 17)
    a.next(); a.next()
    state = a.state_dict()
    b = BatchStream(torch.arange(11), 4, 999)
    b.load_state_dict(state)
    for _ in range(8):
        assert torch.equal(a.next(), b.next())


def test_train_only_stats_and_identity(smoke_config):
    ds = build_dataset(smoke_config)
    stats = prepare_stats(smoke_config, ds)
    assert len(stats['train_indices']) == 56 and len(stats['val_indices']) == 8
    assert not set(stats['train_indices'].tolist()) & set(stats['val_indices'].tolist())
    x = ds.batch(stats['train_indices'])
    torch.testing.assert_close(stats['mean'], x.mean(0))
    changed = copy.deepcopy(smoke_config)
    changed.data.centered = False
    with pytest.raises(ValueError): prepare_stats(changed, ds, stats)


def test_exact_flip_mixture(smoke_config):
    cfg = smoke_config
    cfg.data.random_flip = True
    ds = build_dataset(cfg)
    stats = prepare_stats(cfg, ds)
    torch.testing.assert_close(stats['mean'], stats['mean'].flip(-1))
    assert stats['n_stats_images'] == 2 * stats['n_train']


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b): assert_tree_equal(x,y)
    else:
        assert a == b


@pytest.mark.parametrize('microbatch', [None, 1])
def test_training_resume_bitwise(smoke_config, microbatch):
    full = copy.deepcopy(smoke_config)
    full.name = 'full'
    full.training.n_iters = 4
    full.trainer.microbatch_size = microbatch
    Trainer(full).train()
    partial = copy.deepcopy(full)
    partial.name = 'resumed'
    partial.training.n_iters = 2
    Trainer(partial).train()
    checkpoint = load_checkpoint(Path(partial.trainer.save_dir)/partial.name/'last.pt')
    partial.training.n_iters = 4
    Trainer(partial, checkpoint).train()
    expected = load_checkpoint(Path(full.trainer.save_dir)/full.name/'last.pt')
    actual = load_checkpoint(Path(partial.trainer.save_dir)/partial.name/'last.pt')
    for key in ('model','optimizer','ema','stream','rng'):
        assert_tree_equal(expected[key], actual[key])
    assert expected['step'] == actual['step'] == 4


def test_overwrite_and_mismatched_resume(smoke_config):
    Trainer(smoke_config).train()
    with pytest.raises(FileExistsError): Trainer(smoke_config)
    ck = load_checkpoint(Path(smoke_config.trainer.save_dir)/smoke_config.name/'last.pt')
    changed = copy.deepcopy(smoke_config)
    changed.reference.mode = 'spectral'
    with pytest.raises(ValueError, match='configuration mismatch'): Trainer(changed, ck)


def test_ema_checkpoint_inference_equivalence(smoke_config):
    Trainer(smoke_config).train()
    run = Path(smoke_config.trainer.save_dir)/smoke_config.name
    full, cfg, _, _ = load_model(run/'last.pt', 'cpu')
    snapshot, _, _, _ = load_model(run/'ema_step_0000003.pt', 'cpu')
    for a,b in zip(full.state_dict().values(), snapshot.state_dict().values()):
        assert torch.equal(a,b)


def test_ema_exception_safe_and_rng_isolation():
    model = torch.nn.Linear(4,4)
    ema = ExponentialMovingAverage(model.parameters(), .999)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError):
        with ema.average_parameters(model):
            raise RuntimeError('intentional')
    assert model.training
    for a,b in zip(before, model.parameters()): assert torch.equal(a,b)
    seed_all(1)
    state = rng_state()
    with isolated_rng(12):
        random.random();np.random.rand();torch.rand(3)
    assert_tree_equal(state, rng_state())


def test_tf_record_truncation(tmp_path):
    file = tmp_path/'bad.tfrecord'
    file.write_bytes(b'123')
    with pytest.raises(ValueError): record_index([file])
