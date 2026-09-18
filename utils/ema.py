"""EMA adapted from models/ema.py (originally fadel/pytorch_ema).

Preserves update-count warmup. Adds checked loading and exception-safe swapping.
"""
from contextlib import contextmanager
import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if not 0 <= decay <= 1:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    @torch.no_grad()
    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        parameters = [p for p in parameters if p.requires_grad]
        if len(parameters) != len(self.shadow_params):
            raise ValueError('EMA parameter count mismatch')
        for shadow, param in zip(self.shadow_params, parameters):
            shadow.sub_((1.0 - decay) * (shadow - param))

    @torch.no_grad()
    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        if len(parameters) != len(self.shadow_params):
            raise ValueError('EMA parameter count mismatch')
        for shadow, param in zip(self.shadow_params, parameters):
            param.copy_(shadow)

    @torch.no_grad()
    def store(self, parameters):
        self.collected_params = [p.detach().clone() for p in parameters]

    @torch.no_grad()
    def restore(self, parameters):
        for saved, param in zip(self.collected_params, parameters):
            param.copy_(saved)
        self.collected_params = []

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates, shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        incoming = state_dict['shadow_params']
        if len(incoming) != len(self.shadow_params):
            raise ValueError('EMA checkpoint parameter count mismatch')
        if any(a.shape != b.shape for a, b in zip(incoming, self.shadow_params)):
            raise ValueError('EMA checkpoint parameter shapes mismatch')
        if not 0 <= state_dict['decay'] <= 1:
            raise ValueError('Invalid EMA decay')
        self.decay, self.num_updates = state_dict['decay'], state_dict['num_updates']
        self.shadow_params = [a.detach().to(b).clone() for a, b in zip(incoming, self.shadow_params)]

    @contextmanager
    def average_parameters(self, model):
        training = model.training
        self.store(model.parameters())
        self.copy_to(model.parameters())
        try:
            model.eval()
            yield model
        finally:
            self.restore(model.parameters())
            model.train(training)
