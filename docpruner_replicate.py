#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DocPruner 严格复现脚本

严格对齐论文 Section 4.1 的实验设置：
  - 模型: ColQwen2.5 (vidore/colqwen2.5-v0.2)
  - 数据集: ViDoRe-V2 (BEIR 格式)
      - vidore/esg_reports_v2
      - vidore/biomedical_lectures_eng_v2
      - vidore/economics_reports_v2
      - vidore/esg_reports_human_labeled_v2
  - 指标: nDCG@5 (pytrec_eval)
  - 剪枝方法: DocPruner (EOS attention adaptive threshold)
  - adaptation factor k: {-0.5, -0.25, 0, 0.25, 0.5, 1}
  - Attention 提取: forward hook on model.language_model.layers[-1].self_attn
  - EOS token: 序列最后一个 token ([:, -1, :])

依赖安装:
  pip install "colpali-engine>=0.3.1" datasets torch pillow tqdm pytrec_eval-terrier

用法:
  # 跑单个数据集 + 单个 k 值
  python docpruner_replicate.py \
      --dataset vidore/esg_reports_v2 \
      --pruner docpruner --k -0.25

  # 跑所有数据集（用 run_all_experiments.sh）
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# pytrec_eval 用于计算 nDCG@5（论文使用的标准评估方式）
try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None
    print("警告: pytrec_eval 未安装，将使用简易 nDCG 计算。")
    print("推荐安装: pip install pytrec_eval-terrier")


# =============================================================================
# 论文中 ViDoRe-V2 的 4 个英文数据集（Section 4.1）
# =============================================================================

VIDORE_V2_DATASETS = [
    "vidore/esg_reports_v2",
    "vidore/biomedical_lectures_eng_v2",
    "vidore/economics_reports_v2",
    "vidore/esg_reports_human_labeled_v2",
]

# DocPruner 论文中的 k 值范围（Section 4.1 Implementation Details）
DOCPRUNER_K_VALUES = [-0.5, -0.25, 0, 0.25, 0.5, 1.0]


# =============================================================================
# 工具函数
# =============================================================================

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


# =============================================================================
# Attention Hook — 按作者邮件中的实现细节
# =============================================================================

class AttentionHookManager:
    """
    用 forward hook 从最后一层 self-attention 提取注意力权重。

    作者邮件原文要点：
    1. model config 设置 output_attentions=True
    2. 对最后一层 self_attn 注册 forward hook
    3. hook 捕获 output[1]，shape = (batch, heads, seq_len, seq_len)
    4. 用完立即清理 hook 防止显存泄漏
    """

    def __init__(self):
        self.attention_weights: Optional[torch.Tensor] = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_weights = output[1].detach()

    def register(self, attn_layer: nn.Module):
        self._hook_handle = attn_layer.register_forward_hook(self._hook_fn)

    def clear(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if self.attention_weights is not None:
            del self.attention_weights
            self.attention_weights = None


# =============================================================================
# 模型加载（ColQwen2.5 — 论文使用的模型）
# =============================================================================

def load_colqwen25(model_name: str, device: torch.device, need_attention: bool = False):
    """
    加载 ColQwen2.5 模型，论文 Section 4.1 的 base model 之一。

    使用 colpali_engine 的 ColQwen2_5 / ColQwen2_5_Processor。
    模型名: vidore/colqwen2.5-v0.2
    Attention 层路径: model.language_model.layers[-1].self_attn（作者邮件确认）
    """
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    # SDPA 不支持 output_attentions，需要用 eager 模式
    extra_kwargs = {}
    if need_attention:
        extra_kwargs["attn_implementation"] = "eager"

    model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
        **extra_kwargs,
    ).eval()

    # 作者邮件要点 1: 设置 output_attentions=True
    if need_attention:
        model.config.output_attentions = True
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
            model.language_model.config.output_attentions = True

    processor = ColQwen2_5_Processor.from_pretrained(model_name)

    # 验证 attention 层路径
    if need_attention:
        try:
            attn_layer = model.language_model.layers[-1].self_attn
            print(f"  Attention 层定位成功: model.language_model.layers[-1].self_attn")
        except AttributeError:
            raise RuntimeError(
                "无法定位 ColQwen2.5 的最后一层 self_attn。"
                "请确认 colpali_engine 版本 >= 0.3.1。"
            )

    return model, processor


# =============================================================================
# 嵌入 + 注意力提取
# =============================================================================

@torch.no_grad()
def embed_images_with_attention(
    model, processor, images: List[Image.Image],
    device: torch.device,
    extract_attention: bool = False,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """
    编码文档图片，可选地提取 EOS token 注意力分数。

    关键发现：模型输出的 embedding 序列中包含 text prompt tokens 和 vision patch tokens。
    DocPruner 只对 vision patch tokens（<|image_pad|>, id=151655）做剪枝。

    Returns:
        emb_list: List[Tensor[seq_len, D]]，每个文档的完整 embeddings
        attn_list: List[Tensor[seq_len]]，EOS token 对所有位置的注意力
        imgpad_mask_list: List[Tensor[seq_len]]，True = image_pad token（可剪枝）
    """
    IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|>

    batch = processor.process_images(images).to(device)
    input_ids = batch["input_ids"]  # [B, S]

    hook_manager = None
    if extract_attention:
        hook_manager = AttentionHookManager()
        attn_layer = model.language_model.layers[-1].self_attn
        hook_manager.register(attn_layer)

    try:
        embeddings = model(**batch)  # [B, S, D]
    finally:
        if hook_manager:
            raw_attn = hook_manager.attention_weights
            hook_manager.clear()

    # 拆分 per-image
    emb_list = []
    imgpad_mask_list = []
    for j in range(embeddings.shape[0]):
        emb_list.append(embeddings[j].detach().cpu())
        # 标记哪些位置是 image_pad token
        mask = (input_ids[j] == IMAGE_PAD_TOKEN_ID).detach().cpu()
        imgpad_mask_list.append(mask)

    # 提取注意力分数
    attn_list = None
    if extract_attention and raw_attn is not None:
        avg_attn = raw_attn.mean(dim=1)  # [B, S, S]
        eos_attn = avg_attn[:, -1, :]     # [B, S]

        attn_list = []
        for j in range(embeddings.shape[0]):
            scores = eos_attn[j, :embeddings.shape[1]].detach().cpu().float()
            attn_list.append(scores)

    return emb_list, attn_list, imgpad_mask_list


@torch.no_grad()
def embed_queries(model, processor, queries: List[str],
                  device: torch.device) -> List[torch.Tensor]:
    """编码查询文本，返回 per-query embeddings"""
    batch = processor.process_queries(queries).to(device)
    embeddings = model(**batch)  # [B, L, D]
    emb_list = []
    for j in range(embeddings.shape[0]):
        emb_list.append(embeddings[j].detach().cpu())
    return emb_list


# =============================================================================
# DocPruner 剪枝（论文 Section 3.2）
# =============================================================================

@dataclass
class PruneResult:
    vectors: torch.Tensor
    pruning_ratio: float
    num_before: int
    num_after: int


def docpruner_prune(embeddings: torch.Tensor,
                    attention_scores: torch.Tensor,
                    k: float,
                    imgpad_mask: Optional[torch.Tensor] = None) -> PruneResult:
    """
    DocPruner 自适应剪枝，严格按论文 Section 3.2。
    
    关键：只对 image_pad tokens 做剪枝，text tokens 保留不动。

    公式 (3): μ = mean(I(dj))  — 只算 image_pad 位置
    公式 (4): σ = std(I(dj))   — 只算 image_pad 位置
    阈值:     τ = μ + k·σ
    公式 (5): 保留 I(dj) > τ 的 image_pad patch
    公式 (6): 至少保留一个 image_pad patch
    """
    n = embeddings.shape[0]
    if n == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=0, num_after=0)

    if imgpad_mask is None:
        # 没有 mask，对所有 token 做剪枝（fallback）
        imgpad_mask = torch.ones(n, dtype=torch.bool)

    # 分离 text tokens 和 image_pad tokens
    text_mask = ~imgpad_mask
    img_indices = imgpad_mask.nonzero(as_tuple=True)[0]
    text_indices = text_mask.nonzero(as_tuple=True)[0]

    n_img = img_indices.shape[0]
    if n_img == 0:
        # 没有 image_pad token，不剪枝
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=n, num_after=n)

    # 只取 image_pad 位置的注意力分数
    img_attn = attention_scores[img_indices]

    # 处理 attention sink: 序列开头的少量 image_pad token 会吸收极端高注意力
    # （如 <|vision_start|> 旁边的 token），但不携带有意义的视觉信息。
    # 用 IQR 方法裁剪 outlier，使 μ 和 σ 反映真实的 patch 注意力分布。
    q75 = img_attn.quantile(0.75).item()
    q25 = img_attn.quantile(0.25).item()
    iqr = q75 - q25
    upper_fence = q75 + 3.0 * iqr  # 宽松上界，只裁掉极端值
    img_attn_clipped = img_attn.clamp(max=upper_fence)

    # 论文公式 (3)(4): 对裁剪后的分布计算 μ 和 σ
    mu = img_attn_clipped.mean().item()
    sigma = img_attn_clipped.std().item() if n_img > 1 else 0.0
    tau = mu + k * sigma

    # 论文公式 (5): 在 image_pad 中保留注意力 > τ 的
    keep_img_mask = img_attn > tau

    # 论文公式 (6): 至少保留一个 image_pad patch
    if keep_img_mask.sum() == 0:
        keep_img_mask[img_attn.argmax()] = True

    # 合并：保留的 image_pad patches + 全部 text tokens
    kept_img_indices = img_indices[keep_img_mask]
    all_kept_indices = torch.cat([text_indices, kept_img_indices]).sort().values
    pruned = embeddings[all_kept_indices]

    n_img_kept = kept_img_indices.shape[0]
    pruning_ratio = 1.0 - (n_img_kept / n_img) if n_img > 0 else 0.0

    return PruneResult(
        vectors=pruned,
        pruning_ratio=pruning_ratio,
        num_before=n_img,       # 只报告 image_pad 的数量
        num_after=n_img_kept,
    )


def identity_prune(embeddings: torch.Tensor, **kwargs) -> PruneResult:
    """不剪枝 baseline"""
    n = embeddings.shape[0]
    return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                       num_before=n, num_after=n)


def random_prune(embeddings: torch.Tensor, ratio: float = 0.5,
                 seed: int = 0, **kwargs) -> PruneResult:
    """随机剪枝 baseline（论文 Section 4.1 的 Random 方法）"""
    n = embeddings.shape[0]
    num_keep = max(1, int(n * (1.0 - ratio)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)[:num_keep].sort().values
    pruned = embeddings[perm]
    return PruneResult(vectors=pruned, pruning_ratio=1.0 - (num_keep / n),
                       num_before=n, num_after=num_keep)


def docmerger_compress(embeddings: torch.Tensor,
                       attention_scores: torch.Tensor,
                       k1: float = 0.5,
                       k2: float = 0.25,
                       merge_ratio: float = 0.25,
                       imgpad_mask: Optional[torch.Tensor] = None,
                       weighted: bool = True) -> PruneResult:
    """
    DocMerger: tri-level partitioning + attention-weighted merging.

    P_preserve = { d_j | I(d_j) > μ + k1·σ }           → keep as-is
    P_merge    = { d_j | μ - k2·σ < I(d_j) ≤ μ + k1·σ } → cluster + merge
    P_discard  = { d_j | I(d_j) ≤ μ - k2·σ }            → remove

    Args:
        k1: upper threshold factor (preserve vs merge boundary)
        k2: lower threshold factor (merge vs discard boundary)
        merge_ratio: reduce P_merge to this fraction of its size via clustering
        weighted: if True, use attention-weighted centroids; else simple average
    """
    from sklearn.cluster import AgglomerativeClustering

    n = embeddings.shape[0]
    if n == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=0, num_after=0)

    if imgpad_mask is None:
        imgpad_mask = torch.ones(n, dtype=torch.bool)

    text_indices = (~imgpad_mask).nonzero(as_tuple=True)[0]
    img_indices = imgpad_mask.nonzero(as_tuple=True)[0]
    n_img = img_indices.shape[0]

    if n_img == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=n, num_after=n)

    img_attn = attention_scores[img_indices]

    # IQR clipping (same as docpruner) for robust μ/σ
    q75 = img_attn.quantile(0.75).item()
    q25 = img_attn.quantile(0.25).item()
    iqr = q75 - q25
    img_attn_clipped = img_attn.clamp(max=q75 + 3.0 * iqr)

    mu = img_attn_clipped.mean().item()
    sigma = img_attn_clipped.std().item() if n_img > 1 else 0.0

    tau_high = mu + k1 * sigma
    tau_low = mu - k2 * sigma

    # Tri-level partition
    preserve_mask = img_attn > tau_high
    discard_mask = img_attn <= tau_low
    merge_mask = ~preserve_mask & ~discard_mask

    preserve_idx = img_indices[preserve_mask]
    merge_idx = img_indices[merge_mask]

    # Ensure at least one patch survives
    if preserve_idx.shape[0] == 0 and merge_idx.shape[0] == 0:
        preserve_idx = img_indices[img_attn.argmax().unsqueeze(0)]
        merge_idx = torch.tensor([], dtype=torch.long)

    # Merge P_merge via agglomerative clustering
    merged_vectors = torch.empty(0, embeddings.shape[1])
    n_merge = merge_idx.shape[0]
    if n_merge > 0:
        n_clusters = max(1, int(n_merge * merge_ratio))
        if n_clusters >= n_merge:
            # No reduction needed, keep all
            merged_vectors = embeddings[merge_idx]
        else:
            merge_emb = embeddings[merge_idx].float().numpy()
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            )
            labels = clustering.fit_predict(merge_emb)
            merge_attn = img_attn[merge_mask]

            centroids = []
            for c in range(n_clusters):
                cmask = torch.tensor(labels == c)
                c_emb = embeddings[merge_idx[cmask]]
                if weighted:
                    w = merge_attn[cmask].unsqueeze(1)
                    w = w / w.sum()
                    centroid = (c_emb * w).sum(dim=0)
                else:
                    centroid = c_emb.mean(dim=0)
                centroids.append(centroid)
            merged_vectors = torch.stack(centroids)

    # Combine: text tokens + preserved patches + merged centroids
    parts = [embeddings[text_indices], embeddings[preserve_idx]]
    if merged_vectors.shape[0] > 0:
        parts.append(merged_vectors)
    result = torch.cat(parts, dim=0)

    n_img_after = preserve_idx.shape[0] + merged_vectors.shape[0]
    pruning_ratio = 1.0 - (n_img_after / n_img) if n_img > 0 else 0.0

    return PruneResult(
        vectors=result,
        pruning_ratio=pruning_ratio,
        num_before=n_img,
        num_after=int(n_img_after),
    )


# =============================================================================
# MaxSim 检索（Late Interaction）
# =============================================================================

def pad_3d(list_2d: List[torch.Tensor], pad_value: float = 0.0
           ) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(list_2d) == 0:
        return torch.zeros(0, 0, 0), torch.zeros(0, 0, dtype=torch.bool)
    b = len(list_2d)
    d = list_2d[0].shape[1]
    lmax = max(x.shape[0] for x in list_2d)
    X = torch.full((b, lmax, d), pad_value, dtype=list_2d[0].dtype)
    mask = torch.zeros((b, lmax), dtype=torch.bool)
    for i, x in enumerate(list_2d):
        li = x.shape[0]
        if li > 0:
            X[i, :li] = x
            mask[i, :li] = True
    return X, mask


def maxsim_retrieval(
    Q_list: List[torch.Tensor],
    D_list: List[torch.Tensor],
    query_ids: List[str],
    corpus_ids: List[str],
    device: torch.device,
    batch_q: int = 4,
    batch_d: int = 16,
) -> Dict[str, Dict[str, float]]:
    """
    MaxSim 全量检索，返回 pytrec_eval 格式的 scores。

    Returns:
        {query_id: {corpus_id: score, ...}, ...}
    """
    nq = len(Q_list)
    nd = len(D_list)
    results: Dict[str, Dict[str, float]] = {}

    for qs in tqdm(range(0, nq, batch_q), desc="MaxSim retrieval"):
        q_batch = Q_list[qs:qs + batch_q]
        Q, qmask = pad_3d(q_batch)
        Q = Q.to(device)
        qmask = qmask.to(device)
        bq = Q.shape[0]
        Lq = Q.shape[1]

        # 为这批 query 收集所有 doc 的分数
        all_scores = torch.zeros(bq, nd, device=device)

        for ds in range(0, nd, batch_d):
            d_batch = D_list[ds:ds + batch_d]
            D, dmask = pad_3d(d_batch)
            D = D.to(device)
            dmask = dmask.to(device)
            bd = D.shape[0]
            Ld = D.shape[1]

            # sim: [bq, bd, Lq, Ld]
            sim = torch.einsum("qld,bmd->qblm", Q, D)
            sim = sim.masked_fill(~dmask.view(1, bd, 1, Ld), -1e9)

            # max over doc tokens → [bq, bd, Lq]
            sim_max = sim.max(dim=-1).values
            sim_max = sim_max * qmask.view(bq, 1, Lq).to(sim_max.dtype)

            # sum over query tokens → [bq, bd]
            scores = sim_max.sum(dim=-1)
            all_scores[:, ds:ds + bd] = scores

        # 转换为 pytrec_eval 格式
        all_scores_cpu = all_scores.detach().cpu()
        for i in range(bq):
            qi = qs + i
            if qi >= nq:
                break
            qid = query_ids[qi]
            results[qid] = {}
            for j in range(nd):
                results[qid][corpus_ids[j]] = float(all_scores_cpu[i, j])

    return results


# =============================================================================
# 评估（nDCG@5，论文的主要指标）
# =============================================================================

def evaluate_ndcg5(run: Dict[str, Dict[str, float]],
                   qrels: Dict[str, Dict[str, int]]) -> float:
    """
    用 pytrec_eval 计算 nDCG@5（论文的 primary metric）。
    如果 pytrec_eval 不可用，用简易实现。
    """
    if pytrec_eval is not None:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_5"})
        results = evaluator.evaluate(run)
        ndcg5_scores = [v["ndcg_cut_5"] for v in results.values()]
        return sum(ndcg5_scores) / max(1, len(ndcg5_scores))
    else:
        # 简易 nDCG@5 fallback
        ndcg_sum = 0.0
        count = 0
        for qid, doc_scores in run.items():
            if qid not in qrels:
                continue
            relevant = qrels[qid]
            ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:5]
            dcg = 0.0
            for rank, (did, _) in enumerate(ranked, 1):
                if did in relevant and relevant[did] > 0:
                    dcg += 1.0 / math.log2(rank + 1)
            # ideal DCG
            ideal_gains = sorted(relevant.values(), reverse=True)[:5]
            idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal_gains) if g > 0)
            ndcg_sum += dcg / max(idcg, 1e-10)
            count += 1
        return ndcg_sum / max(1, count)


# =============================================================================
# ViDoRe-V2 BEIR 格式数据集加载
# =============================================================================

def load_vidore_v2_dataset(dataset_name: str, split: str = "test", language: str = "english"):
    """
    加载 ViDoRe-V2 BEIR 格式数据集。

    BEIR 格式有 3 个子集: corpus, queries, qrels
    - corpus: {doc_id, image}
    - queries: {query_id, text/query, language?}
    - qrels: {query_id, doc_id, score}
    """

    def _get_field(row, candidates):
        """从 row 中按优先级查找字段"""
        for c in candidates:
            if c in row:
                return row[c]
        raise KeyError(f"找不到字段，尝试了 {candidates}，实际字段: {list(row.keys())}")

    print(f"  加载 corpus...")
    corpus_ds = load_dataset(dataset_name, "corpus", split=split)

    print(f"  加载 queries...")
    queries_ds = load_dataset(dataset_name, "queries", split=split)

    print(f"  加载 qrels...")
    qrels_ds = load_dataset(dataset_name, "qrels", split=split)

    # 打印字段名帮助调试
    print(f"  Corpus 字段: {corpus_ds.column_names}")
    print(f"  Queries 字段: {queries_ds.column_names}")
    print(f"  Qrels 字段: {qrels_ds.column_names}")

    # 构建 corpus: {doc_id: PIL.Image}
    corpus_ids = []
    corpus_images = []
    for row in corpus_ds:
        doc_id = str(_get_field(row, ["corpus-id", "corpus_id", "doc-id", "doc_id", "_id"]))
        corpus_ids.append(doc_id)
        corpus_images.append(to_pil(row["image"]))

    # 构建 queries: {query_id: text}（按语言过滤）
    query_ids = []
    query_texts = []
    for row in queries_ds:
        qid = str(_get_field(row, ["query-id", "query_id", "_id", "id", "qid"]))
        # 按语言过滤（论文实验用英文）
        row_lang = row.get("language", "english")
        if language and row_lang != language:
            continue
        query_ids.append(qid)
        text = str(_get_field(row, ["query", "text", "question", "content"]))
        query_texts.append(text)

    # 构建 qrels: {query_id: {doc_id: relevance}}
    qrels = {}
    for row in qrels_ds:
        qid = str(_get_field(row, ["query-id", "query_id", "qid"]))
        did = str(_get_field(row, ["corpus-id", "corpus_id", "doc-id", "doc_id", "docid"]))
        score = int(_get_field(row, ["score", "relevance", "label"]))
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = score

    # 只保留在 qrels 中有标注的 query
    filtered_qids = []
    filtered_texts = []
    for qid, text in zip(query_ids, query_texts):
        if qid in qrels:
            filtered_qids.append(qid)
            filtered_texts.append(text)

    print(f"  Corpus: {len(corpus_ids)} 个文档")
    print(f"  Queries: {len(filtered_qids)} 个查询 (语言={language})")
    print(f"  Qrels: {sum(len(v) for v in qrels.values())} 条标注")

    return corpus_ids, corpus_images, filtered_qids, filtered_texts, qrels


# =============================================================================
# 主实验流程
# =============================================================================

def run_experiment(args):
    """对单个数据集 + 单个 k 值运行实验"""

    set_seed(args.seed)
    device = pick_device(args.device)
    need_attention = (args.pruner in ("docpruner", "docmerger", "docmerger_avg"))

    dataset_short = args.dataset.split("/")[-1]

    print(f"\n{'='*70}")
    print(f"DocPruner 严格复现实验")
    print(f"{'='*70}")
    print(f"模型: {args.model_name}")
    print(f"数据集: {args.dataset}")
    print(f"剪枝器: {args.pruner} (k={args.k})")
    print(f"设备: {device}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 1. 加载数据集（ViDoRe-V2 BEIR 格式）
    # ------------------------------------------------------------------
    print(f"\n[1/5] 加载数据集 ({dataset_short})...")
    corpus_ids, corpus_images, query_ids, query_texts, qrels = \
        load_vidore_v2_dataset(args.dataset, split=args.split, language=args.language)

    n_corpus = len(corpus_ids)
    n_queries = len(query_ids)

    model = None
    processor = None

    # ------------------------------------------------------------------
    # 3. 编码 corpus（文档图片 + 注意力提取）— 带缓存
    # ------------------------------------------------------------------
    cache_dir = Path(args.outdir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_name.replace("/", "_")
    cache_emb_path = cache_dir / f"{dataset_short}_{model_short}_emb.pt"
    cache_attn_path = cache_dir / f"{dataset_short}_{model_short}_attn.pt"
    cache_mask_path = cache_dir / f"{dataset_short}_{model_short}_imgpad_mask.pt"

    if cache_emb_path.exists():
        if args.clear_cache:
            print(f"  清除缓存: {cache_emb_path}")
            cache_emb_path.unlink()
            if cache_attn_path.exists():
                cache_attn_path.unlink()
            if cache_mask_path.exists():
                cache_mask_path.unlink()

    if cache_emb_path.exists():
        print(f"\n[2-3/5] 从缓存加载 corpus embeddings...")
        print(f"  缓存文件: {cache_emb_path}")
        doc_emb_list = torch.load(cache_emb_path, weights_only=False)
        if cache_attn_path.exists():
            doc_attn_list = torch.load(cache_attn_path, weights_only=False)
            print(f"  Attention 缓存已加载")
        else:
            doc_attn_list = [None] * len(doc_emb_list)
        if cache_mask_path.exists():
            doc_mask_list = torch.load(cache_mask_path, weights_only=False)
            print(f"  Image_pad mask 缓存已加载")
        else:
            doc_mask_list = [None] * len(doc_emb_list)
        avg_patches = sum(e.shape[0] for e in doc_emb_list) / max(1, len(doc_emb_list))
        print(f"  已加载 {len(doc_emb_list)} 个文档, 平均 token 数: {avg_patches:.0f}")
    else:
        # 编码时永远提取 attention（eager 模式），这样缓存对所有 pruner 通用
        print(f"\n[2/5] 加载 ColQwen2.5 模型 (eager 模式, 提取 attention)...")
        model, processor = load_colqwen25(
            args.model_name, device=device, need_attention=True
        )
        print(f"  模型加载完成")

        print(f"\n[3/5] 编码 corpus ({n_corpus} 个文档)...")
        doc_emb_list: List[torch.Tensor] = []
        doc_attn_list: List[Optional[torch.Tensor]] = []
        doc_mask_list: List[Optional[torch.Tensor]] = []

        for i in tqdm(range(0, n_corpus, args.batch_doc), desc="Embedding docs"):
            batch_imgs = corpus_images[i:i + args.batch_doc]
            embs, attns, masks = embed_images_with_attention(
                model, processor, batch_imgs,
                device=device, extract_attention=True,
            )
            for j, emb in enumerate(embs):
                doc_emb_list.append(emb)
                if attns is not None:
                    doc_attn_list.append(attns[j])
                else:
                    doc_attn_list.append(None)
                if masks is not None:
                    doc_mask_list.append(masks[j])
                else:
                    doc_mask_list.append(None)

        avg_patches = sum(e.shape[0] for e in doc_emb_list) / max(1, len(doc_emb_list))
        n_imgpad = sum(m.sum().item() for m in doc_mask_list if m is not None) / max(1, len(doc_mask_list))
        print(f"  平均 token 数: {avg_patches:.0f} (其中 image_pad: {n_imgpad:.0f})")

        # 保存缓存
        print(f"  保存缓存到: {cache_emb_path}")
        torch.save(doc_emb_list, cache_emb_path)
        torch.save(doc_attn_list, cache_attn_path)
        torch.save(doc_mask_list, cache_mask_path)
        print(f"  保存 attention + mask 缓存完成")

    # ------------------------------------------------------------------
    # 4. 剪枝
    # ------------------------------------------------------------------
    print(f"\n[4/5] 剪枝 ({args.pruner}, k={args.k})...")
    pruned_docs: List[torch.Tensor] = []
    per_doc_stats = []

    for i in tqdm(range(n_corpus), desc="Pruning"):
        if args.pruner == "docpruner":
            result = docpruner_prune(
                doc_emb_list[i], doc_attn_list[i], k=args.k,
                imgpad_mask=doc_mask_list[i] if i < len(doc_mask_list) else None,
            )
        elif args.pruner in ("docmerger", "docmerger_avg"):
            result = docmerger_compress(
                doc_emb_list[i], doc_attn_list[i],
                k1=args.k1, k2=args.k2, merge_ratio=args.merge_ratio,
                imgpad_mask=doc_mask_list[i] if i < len(doc_mask_list) else None,
                weighted=(args.pruner == "docmerger"),
            )
        elif args.pruner == "random":
            result = random_prune(doc_emb_list[i], ratio=args.random_ratio, seed=args.seed)
        else:
            result = identity_prune(doc_emb_list[i])

        pruned_docs.append(result.vectors)
        per_doc_stats.append({
            "doc_id": corpus_ids[i],
            "num_before": result.num_before,
            "num_after": result.num_after,
            "pruning_ratio": round(result.pruning_ratio, 4),
        })

    avg_pruning = sum(s["pruning_ratio"] for s in per_doc_stats) / max(1, len(per_doc_stats))
    avg_after = sum(s["num_after"] for s in per_doc_stats) / max(1, len(per_doc_stats))
    print(f"  平均剪枝率: {avg_pruning*100:.1f}%")
    print(f"  平均剩余 patch: {avg_after:.0f}")

    # 释放原始 embeddings 和 attention 节省显存
    del doc_emb_list, doc_attn_list, doc_mask_list
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------
    # 5. 编码 queries + MaxSim 检索 + 评估
    # ------------------------------------------------------------------
    # 如果从缓存加载了 corpus（没有加载模型），这里才加载
    if model is None:
        print(f"\n  加载 ColQwen2.5 模型（用于 query 编码）...")
        model, processor = load_colqwen25(
            args.model_name, device=device, need_attention=False
        )
        print(f"  模型加载完成")

    print(f"\n[5/5] 编码 queries + MaxSim 检索...")
    q_emb_list: List[torch.Tensor] = []
    for i in tqdm(range(0, n_queries, args.batch_query), desc="Embedding queries"):
        batch_q = query_texts[i:i + args.batch_query]
        embs = embed_queries(model, processor, batch_q, device=device)
        q_emb_list.extend(embs)

    # MaxSim 全量检索
    run = maxsim_retrieval(
        Q_list=q_emb_list,
        D_list=pruned_docs,
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        device=device,
        batch_q=args.batch_score_q,
        batch_d=args.batch_score_d,
    )

    # 计算 nDCG@5
    ndcg5 = evaluate_ndcg5(run, qrels)

    # ------------------------------------------------------------------
    # 输出结果
    # ------------------------------------------------------------------
    metrics = {
        "dataset": args.dataset,
        "dataset_short": dataset_short,
        "model_name": args.model_name,
        "pruner": args.pruner,
        "k": args.k if args.pruner == "docpruner" else None,
        "ndcg@5": round(ndcg5, 4),
        "avg_pruning_ratio": round(avg_pruning, 4),
        "avg_patches_before": round(avg_patches, 1),
        "avg_patches_after": round(avg_after, 1),
        "num_corpus": n_corpus,
        "num_queries": n_queries,
        "language": args.language,
    }

    # 保存
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.pruner}_k{args.k}" if args.pruner == "docpruner" else args.pruner
    fname = f"{dataset_short}_{suffix}"

    (outdir / f"metrics_{fname}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    (outdir / f"per_doc_{fname}.json").write_text(
        json.dumps(per_doc_stats[:20], indent=2, ensure_ascii=False))  # 只存前 20 条示例

    print(f"\n{'='*70}")
    print(f"结果: nDCG@5 = {ndcg5:.4f} | 剪枝率 = {avg_pruning*100:.1f}%")
    print(f"{'='*70}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    return metrics


# =============================================================================
# 入口
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="DocPruner 严格复现（论文 Figure 2）")

    # 数据集 — ViDoRe-V2 BEIR 格式
    p.add_argument("--dataset", default="vidore/esg_reports_v2",
                   help=f"数据集名称，可选: {VIDORE_V2_DATASETS}")
    p.add_argument("--split", default="test")
    p.add_argument("--language", default="english",
                   help="Query 语言过滤（论文 Figure 2 用 english）")

    # 模型 — ColQwen2.5（论文 Figure 2 Left）
    p.add_argument("--model-name", default="vidore/colqwen2.5-v0.2",
                   help="论文使用 vidore/colqwen2.5-v0.2")
    p.add_argument("--device", default=None)

    # 批处理
    p.add_argument("--batch-doc", type=int, default=4)
    p.add_argument("--batch-query", type=int, default=8)
    p.add_argument("--batch-score-q", type=int, default=4)
    p.add_argument("--batch-score-d", type=int, default=16)

    # 剪枝
    p.add_argument("--pruner", choices=["identity", "docpruner", "random", "docmerger", "docmerger_avg"],
                   default="docpruner")
    p.add_argument("--k", type=float, default=-0.25,
                   help="DocPruner adaptation factor（论文推荐 -0.25）")
    p.add_argument("--random-ratio", type=float, default=0.5)

    # DocMerger 参数
    p.add_argument("--k1", type=float, default=0.5,
                   help="DocMerger upper threshold (preserve vs merge)")
    p.add_argument("--k2", type=float, default=0.25,
                   help="DocMerger lower threshold (merge vs discard)")
    p.add_argument("--merge-ratio", type=float, default=0.25,
                   help="DocMerger: reduce P_merge to this fraction")

    # 其他
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs_replicate")
    p.add_argument("--clear-cache", action="store_true",
                   help="清除缓存，强制重新编码 corpus")

    # 是否跑全部数据集 + 全部 k 值
    p.add_argument("--run-all", action="store_true",
                   help="跑论文 Figure 2 的完整实验: 4 个数据集 × 7 个配置")

    args = p.parse_args()

    if args.run_all:
        # 完整复现 Figure 2（ColQwen2.5 列）
        all_metrics = []

        # 1. Baseline: identity
        for ds in VIDORE_V2_DATASETS:
            args.dataset = ds
            args.pruner = "identity"
            m = run_experiment(args)
            all_metrics.append(m)

        # 2. DocPruner: 所有 k 值
        for k_val in DOCPRUNER_K_VALUES:
            for ds in VIDORE_V2_DATASETS:
                args.dataset = ds
                args.pruner = "docpruner"
                args.k = k_val
                m = run_experiment(args)
                all_metrics.append(m)

        # 3. Random baseline (50%)
        for ds in VIDORE_V2_DATASETS:
            args.dataset = ds
            args.pruner = "random"
            args.random_ratio = 0.5
            m = run_experiment(args)
            all_metrics.append(m)

        # 汇总输出
        outdir = Path(args.outdir)
        (outdir / "all_metrics.json").write_text(
            json.dumps(all_metrics, indent=2, ensure_ascii=False))
        print(f"\n全部实验完成！结果保存在 {outdir}/all_metrics.json")

        # 打印汇总表
        print(f"\n{'='*90}")
        print(f"{'Method':<20} {'k':>6} {'ESG':>8} {'Bio':>8} {'Econ':>8} {'ESG-H':>8} {'Avg':>8} {'Prune%':>8}")
        print("-" * 90)

        # 按 (pruner, k) 分组
        from collections import defaultdict
        grouped = defaultdict(dict)
        for m in all_metrics:
            key = (m["pruner"], m.get("k"))
            grouped[key][m["dataset_short"]] = m

        ds_order = ["esg_reports_v2", "biomedical_lectures_eng_v2",
                     "economics_reports_v2", "esg_reports_human_labeled_v2"]

        for (pruner, k_val), ds_dict in sorted(grouped.items()):
            k_str = f"{k_val}" if k_val is not None else "-"
            scores = [ds_dict.get(ds, {}).get("ndcg@5", 0) for ds in ds_order]
            prune_ratios = [ds_dict.get(ds, {}).get("avg_pruning_ratio", 0) for ds in ds_order]
            avg_score = sum(scores) / max(1, len([s for s in scores if s > 0]))
            avg_prune = sum(prune_ratios) / max(1, len([p for p in prune_ratios if p >= 0]))
            print(f"{pruner:<20} {k_str:>6} "
                  f"{scores[0]:>8.4f} {scores[1]:>8.4f} {scores[2]:>8.4f} {scores[3]:>8.4f} "
                  f"{avg_score:>8.4f} {avg_prune*100:>7.1f}%")
        print(f"{'='*90}")

    else:
        run_experiment(args)


if __name__ == "__main__":
    main()