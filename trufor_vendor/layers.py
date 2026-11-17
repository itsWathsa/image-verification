"""Inference-only equivalents of the timm helpers used by TruFor."""

from torch import nn
from torch.nn.init import trunc_normal_


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0., **kwargs):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, value):
        if self.training:
            raise RuntimeError("This TruFor adapter supports evaluation only.")
        return value
