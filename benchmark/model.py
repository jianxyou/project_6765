"""
ColQwen2.5 model loading, embedding, and attention extraction.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|>


class AttentionHookManager:
    """Forward hook to capture attention weights from the last self-attention layer."""

    def __init__(self):
        self.attention_weights: Optional[torch.Tensor] = None
        self._handle = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_weights = output[1].detach()

    def register(self, attn_layer: nn.Module):
        self._handle = attn_layer.register_forward_hook(self._hook_fn)

    def clear(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self.attention_weights is not None:
            del self.attention_weights
            self.attention_weights = None


def load_colqwen25(model_name: str, device: torch.device,
                   need_attention: bool = False):
    """Load ColQwen2.5 model and processor."""
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    extra_kwargs = {}
    if need_attention:
        extra_kwargs["attn_implementation"] = "eager"

    model = ColQwen2_5.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map=str(device), **extra_kwargs,
    ).eval()

    if need_attention:
        model.config.output_attentions = True
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
            model.language_model.config.output_attentions = True

    processor = ColQwen2_5_Processor.from_pretrained(model_name)

    if need_attention:
        attn_layer = model.language_model.layers[-1].self_attn
        print(f"  Attention layer located: model.language_model.layers[-1].self_attn")

    return model, processor


@torch.no_grad()
def embed_images_with_attention(
    model, processor, images: List[Image.Image],
    device: torch.device, extract_attention: bool = False,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]],
           Optional[List[torch.Tensor]], Optional[List[Tuple[int, int]]]]:
    """
    Encode document images, optionally extract EOS attention scores.

    Returns:
        emb_list:  [Tensor[seq_len, D], ...]
        attn_list: [Tensor[seq_len], ...] or None
        mask_list: [Tensor[seq_len], ...] (True = image_pad token)
        grid_list: [(H, W), ...] post-spatial-merge grid dimensions per image,
                   or None if not available from the processor
    """
    batch = processor.process_images(images).to(device)
    input_ids = batch["input_ids"]

    hook = None
    if extract_attention:
        hook = AttentionHookManager()
        hook.register(model.language_model.layers[-1].self_attn)

    try:
        embeddings = model(**batch)
    finally:
        if hook:
            raw_attn = hook.attention_weights
            hook.clear()

    emb_list = []
    mask_list = []
    for j in range(embeddings.shape[0]):
        emb_list.append(embeddings[j].detach().cpu())
        mask_list.append((input_ids[j] == IMAGE_PAD_TOKEN_ID).detach().cpu())

    attn_list = None
    if extract_attention and raw_attn is not None:
        avg_attn = raw_attn.mean(dim=1)
        eos_attn = avg_attn[:, -1, :]
        attn_list = []
        for j in range(embeddings.shape[0]):
            attn_list.append(eos_attn[j, :embeddings.shape[1]].detach().cpu().float())

    # Extract post-merge grid (H, W) per image from image_grid_thw
    grid_list = None
    if "image_grid_thw" in batch:
        thw = batch["image_grid_thw"].detach().cpu()
        merge_size = getattr(
            getattr(model.config, "vision_config", None), "spatial_merge_size", 2
        ) or 2
        grid_list = []
        for j in range(thw.shape[0]):
            t, h, w = thw[j].tolist()
            grid_list.append((h // merge_size, w // merge_size))

    return emb_list, attn_list, mask_list, grid_list


@torch.no_grad()
def embed_queries(model, processor, queries: List[str],
                  device: torch.device) -> List[torch.Tensor]:
    batch = processor.process_queries(queries).to(device)
    embeddings = model(**batch)
    return [embeddings[j].detach().cpu() for j in range(embeddings.shape[0])]


def encode_corpus(model, processor, images: List[Image.Image],
                  device: torch.device, batch_size: int = 4):
    """Encode full corpus with attention extraction.
    Returns (emb_list, attn_list, mask_list, grid_list)."""
    all_emb, all_attn, all_mask, all_grid = [], [], [], []
    for i in tqdm(range(0, len(images), batch_size), desc="Embedding docs"):
        batch_imgs = images[i:i + batch_size]
        embs, attns, masks, grids = embed_images_with_attention(
            model, processor, batch_imgs, device=device, extract_attention=True,
        )
        for j, emb in enumerate(embs):
            all_emb.append(emb)
            all_attn.append(attns[j] if attns else None)
            all_mask.append(masks[j] if masks else None)
            all_grid.append(grids[j] if grids else None)
    return all_emb, all_attn, all_mask, all_grid


def encode_queries(model, processor, texts: List[str],
                   device: torch.device, batch_size: int = 8):
    """Encode all queries. Returns list of tensors."""
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding queries"):
        batch_q = texts[i:i + batch_size]
        all_emb.extend(embed_queries(model, processor, batch_q, device=device))
    return all_emb
