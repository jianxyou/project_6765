from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class PruneResult:
    vectors: torch.Tensor
    pruning_ratio: float
    num_before: int
    num_after: int
    # For visualization: which image_pad patches survived (indices into the image_pad subset)
    kept_indices: Optional[torch.Tensor] = None
    # For visualization: cluster label per kept patch (same length as kept_indices);
    # patches sharing a label were merged into one output vector.
    cluster_labels: Optional[torch.Tensor] = None
    # For visualization: bool per kept patch. True if this patch is the spatial
    # "representative" of its cluster (its location is where the merged vector lives).
    # For singletons (cluster size 1), this is always True.
    representative_mask: Optional[torch.Tensor] = None
