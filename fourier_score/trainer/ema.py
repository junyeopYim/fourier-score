"""EMA with the score-SDE warm-start decay convention."""

from contextlib import contextmanager
import torch


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.detach(), 1 - decay)

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state, model):
        if state["decay"] != self.decay or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA mismatch")
        params = dict(model.named_parameters())
        self.shadow = {n: t.to(params[n]) for n, t in state["shadow"].items()}
        self.num_updates = state["num_updates"]

    @contextmanager
    def average_parameters(self, model):
        with torch.no_grad():
            params = dict(model.named_parameters())
            backup = {n: params[n].detach().clone() for n in self.shadow}
            for n, t in self.shadow.items():
                params[n].copy_(t)
        try:
            yield
        finally:
            with torch.no_grad():
                for n, t in backup.items():
                    params[n].copy_(t)
