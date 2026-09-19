"""Small BaseModel in the spirit of victoresque/pytorch-template."""
from torch import nn

class BaseModel(nn.Module):
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
