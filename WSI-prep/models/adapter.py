import torch.nn as nn

class Adapter(nn.Module):
    """
    Simple bottleneck adapter (Houlsby-style).
    """
    def __init__(self, in_dim: int, bottleneck: int = 64):
        super().__init__()
        self.down  = nn.Linear(in_dim, bottleneck, bias=False)
        self.nonlin = nn.ReLU()
        self.up    = nn.Linear(bottleneck, in_dim, bias=False)

    def forward(self, x):
        return x + self.up(self.nonlin(self.down(x)))
