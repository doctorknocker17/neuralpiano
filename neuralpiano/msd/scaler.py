import torch
from torch import nn
import math


class Scaler(nn.Module):
    def __init__(self, init_min=math.inf, init_max=-math.inf):
        super().__init__()
        self.register_buffer("min", torch.tensor(init_min))
        self.register_buffer("max", torch.tensor(init_max))
        self.frozen = False

    def forward(self, x):
        if self.frozen:
            scale = (self.max - self.min).clamp(min=1e-5)
            return ((x - self.min) / scale * 2 - 1).clamp(-1.1, 1.1)
        return ((x - self.min) / (self.max - self.min) * 2 - 1).clamp(-1, 1)

    def freeze_scaler(self):
        self.mel[1].frozen = True
        print(f"Scaler frozen: min={self.mel[1].min.item():.3f}, max={self.mel[1].max.item():.3f}")

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(-1, 1) + 1) / 2 * (self.max - self.min) + self.min

def adaptive_update_hook(module: Scaler, input):
    x = input[0]
    if module.training:
        module.min.fill_(torch.min(module.min, x.min()))
        module.max.fill_(torch.max(module.max, x.max()))

def get_scaler(adaptive: bool = True, **kwargs) -> Scaler:
    scaler = Scaler(**kwargs)
    if adaptive:
        scaler.register_forward_pre_hook(adaptive_update_hook)
    return scaler