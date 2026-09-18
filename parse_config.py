"""JSON configuration and strict dotted CLI overrides; no eval or dynamic imports."""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [namespace(v) for v in value]
    return value


def plain(value):
    if isinstance(value, SimpleNamespace):
        value = vars(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def deep_merge(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_json(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f'Cyclic config inheritance: {path}')
    seen.add(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    base = data.pop('extends', None)
    if base is not None:
        data = deep_merge(read_json(path.parent / base, seen), data)
    return data


def validate(c):
    from model.gaussian import REFERENCE_MODES
    if c['reference']['mode'] not in REFERENCE_MODES:
        raise ValueError('Unknown reference.mode')
    if c['model']['name'] != 'ncsnpp':
        raise ValueError('This compact template retains only the NCSN++/DDPM++ backbone')
    if c['training']['sde'] not in ('vesde', 'vpsde', 'subvpsde'):
        raise ValueError('Unknown SDE')
    if c['reference']['mode'] != 'baseline' and c['training']['sde'] != 'vesde':
        raise ValueError('Gaussian reference modes support VE/SMLD only')
    if c['data']['uniform_dequantization']:
        raise ValueError('Uniform dequantization is not supported by the Gaussian statistics protocol')
    if c['backend']['precision'] != 'fp32' or c['backend']['tf32']:
        raise ValueError('This comparison template deliberately requires FP32 and TF32=false')
    for key in ('batch_size', 'n_iters', 'log_freq', 'eval_freq', 'snapshot_freq', 'snapshot_freq_for_preemption'):
        if not isinstance(c['training'][key], int) or c['training'][key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    micro = c['trainer']['microbatch_size']
    if micro is not None and (not isinstance(micro, int) or micro < 1):
        raise ValueError('microbatch_size must be null or a positive integer')
    if c['data_loader']['kind'] not in ('torchvision', 'faces', 'synthetic', 'image_folder'):
        raise ValueError('Unknown data_loader.kind')
    if c['data_loader']['protocol'] not in ('holdout', 'full_train'):
        raise ValueError('protocol must be holdout or full_train')
    if c['data']['image_size'] % 2 ** (len(c['model']['ch_mult']) - 1):
        raise ValueError('image_size must be divisible by the model downsampling factor')
    if c['model']['nf'] < 4 or c['model']['nf'] % 4:
        raise ValueError('model.nf must be a positive multiple of 4 (GroupNorm)')
    if c['training']['continuous'] and c['training']['sde'] == 'vesde' and c['model']['embedding_type'] != 'fourier':
        raise ValueError('Continuous VE presets use Fourier noise embeddings')
    if not c['training']['continuous']:
        if c['model']['embedding_type'] != 'positional' or c['training']['likelihood_weighting']:
            raise ValueError('Discrete training requires positional embedding and no likelihood weighting')
        if c['training']['sde'] == 'subvpsde':
            raise ValueError('Discrete sub-VP is not supported')
    if c['sampling']['predictor'] not in ('none', 'reverse_diffusion', 'euler_maruyama', 'ancestral_sampling'):
        raise ValueError('Unknown predictor')
    if c['sampling']['corrector'] not in ('none', 'langevin', 'ald'):
        raise ValueError('Unknown corrector')
    if c['sampling']['method'] == 'ode' and not c['training']['continuous']:
        raise ValueError('ODE sampling requires continuous training')
    if c['training']['sde'] == 'subvpsde' and c['sampling']['corrector'] != 'none':
        raise ValueError('Sub-VP must use corrector=none')
    if c['sampling']['method'] not in ('pc', 'ode'):
        raise ValueError('Unknown sampling.method')
    if not 0 < c['sampling']['eps'] < 1:
        raise ValueError('sampling.eps must lie in (0,1)')
    if c['sampling']['n_steps_each'] < 0 or c['sampling']['snr'] <= 0:
        raise ValueError('Invalid corrector settings')
    if c['model']['num_scales'] < 2 or not 0 < c['model']['sigma_min'] < c['model']['sigma_max']:
        raise ValueError('Invalid noise schedule')
    if c['reference']['floor'] <= 0 or not math.isfinite(c['reference']['floor']):
        raise ValueError('reference.floor must be finite and positive')
    if c['eval']['batch_size'] < 1 or c['eval']['max_images'] < 1:
        raise ValueError('Evaluation limits must be positive')
    return c


class ConfigParser:
    def __init__(self, data):
        self.data = validate(copy.deepcopy(plain(data)))

    @classmethod
    def from_file(cls, path, overrides=()):
        return cls.from_dict(read_json(path), overrides)

    @classmethod
    def from_dict(cls, data, overrides=()):
        data = copy.deepcopy(plain(data))
        for entry in overrides:
            key, sep, raw = entry.partition('=')
            if not sep:
                raise ValueError('--set requires key=value')
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            node = data
            parts = key.split('.')
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise KeyError(f'Unknown configuration path: {key}')
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f'Unknown configuration key: {key}')
            node[parts[-1]] = value
        return cls(data)

    @property
    def config(self):
        return namespace(self.data)

    def save(self, path):
        Path(path).write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def config_arguments(parser, default='config.json'):
    parser.add_argument('-c', '--config', default=default)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    return parser
