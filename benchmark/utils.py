import random
from typing import Optional

import torch
from PIL import Image


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)
