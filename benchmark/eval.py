"""
MaxSim late-interaction retrieval and nDCG@5 evaluation.
"""

import math
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None


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
    """Full MaxSim retrieval, returns pytrec_eval-format scores."""
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

        all_scores = torch.zeros(bq, nd, device=device)

        for ds in range(0, nd, batch_d):
            d_batch = D_list[ds:ds + batch_d]
            D, dmask = pad_3d(d_batch)
            D = D.to(device)
            dmask = dmask.to(device)
            bd = D.shape[0]
            Ld = D.shape[1]

            sim = torch.einsum("qld,bmd->qblm", Q, D)
            sim = sim.masked_fill(~dmask.view(1, bd, 1, Ld), -1e9)
            sim_max = sim.max(dim=-1).values
            sim_max = sim_max * qmask.view(bq, 1, Lq).to(sim_max.dtype)
            scores = sim_max.sum(dim=-1)
            all_scores[:, ds:ds + bd] = scores

        all_scores_cpu = all_scores.detach().cpu()
        for i in range(bq):
            qi = qs + i
            if qi >= nq:
                break
            qid = query_ids[qi]
            results[qid] = {corpus_ids[j]: float(all_scores_cpu[i, j]) for j in range(nd)}

    return results


def evaluate_ndcg5(run: Dict[str, Dict[str, float]],
                   qrels: Dict[str, Dict[str, int]]) -> float:
    """Compute nDCG@5 using pytrec_eval (or fallback)."""
    if pytrec_eval is not None:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_5"})
        results = evaluator.evaluate(run)
        scores = [v["ndcg_cut_5"] for v in results.values()]
        return sum(scores) / max(1, len(scores))

    # Fallback
    ndcg_sum, count = 0.0, 0
    for qid, doc_scores in run.items():
        if qid not in qrels:
            continue
        relevant = qrels[qid]
        ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:5]
        dcg = sum(1.0 / math.log2(r + 2) for r, (did, _) in enumerate(ranked)
                  if did in relevant and relevant[did] > 0)
        ideal = sorted(relevant.values(), reverse=True)[:5]
        idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal) if g > 0)
        ndcg_sum += dcg / max(idcg, 1e-10)
        count += 1
    return ndcg_sum / max(1, count)
