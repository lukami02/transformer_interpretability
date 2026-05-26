import torch
import torch.nn as nn
from typing import List, Union

class Add(nn.Module):
    """
    LRP-compatible element-wise addition layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(inputs, dim=0).sum(dim=0)
    
class Clone(nn.Module):
    """
    LRP-compatible tensor cloning layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, num_clones: int) -> List[torch.Tensor]:
        return [x.clone() for _ in range(num_clones)]
    
class IndexSelect(nn.Module):
    """
    LRP-compatible index selection layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, dim: int, indices: torch.Tensor) -> torch.Tensor:
        return torch.index_select(inputs, dim, indices)
    
