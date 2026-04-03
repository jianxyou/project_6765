from dataclasses import dataclass
import torch


@dataclass
class PruneResult:
    vectors: torch.Tensor
    pruning_ratio: float
    num_before: int
    num_after: int
