import torch
from typing import List, Tuple


def from_chunks(chunks: List[torch.Tensor], inv: Tuple):
    return torch.cat([chunks[i] for i in inv], dim=-1)
