"""Load EMA inference weights without putting Adam moments on the GPU."""
from parse_config import ConfigParser
from model.model import build_model
from utils.ema import ExponentialMovingAverage
from utils.util import load_checkpoint, configure_runtime


def load_model(path, device='auto', overrides=()):
    ck = load_checkpoint(path)
    if ck.get('format_version') != 1:
        raise ValueError('Expected template v1 checkpoint; legacy formats are not silently guessed')
    cfg = ConfigParser.from_dict(ck['config'], overrides).config
    cfg.device = device
    runtime = configure_runtime(cfg)
    model = build_model(cfg, ck['stats'], runtime)
    model.load_state_dict(ck['model'], strict=True)
    if ck.get('weights') != 'EMA':
        ema = ExponentialMovingAverage(model.parameters(), cfg.model.ema_rate)
        ema.load_state_dict(ck['ema'])
        ema.copy_to(model.parameters())
    model.eval()
    return model, cfg, ck, runtime
