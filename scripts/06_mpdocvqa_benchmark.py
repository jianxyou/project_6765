#!/usr/bin/env python3
"""
MP-DocVQA benchmark adapter for DocPruner/DocMerger evaluation.

Task: per-question page retrieval within a multi-page document.
Each question has 1-20 page images; answer_page_idx is the ground truth.
We encode all pages, compress, then retrieve via MaxSim.

Metric: Retrieval Accuracy@1 and nDCG@5.
"""

import argparse, json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from docpruner_replicate import (
    load_colqwen25, embed_images_with_attention, embed_queries,
    pick_device, set_seed, pad_3d,
    docpruner_prune, docmerger_compress, identity_prune,
)


def load_mpdocvqa(split="val", max_questions=None):
    """Load MP-DocVQA and return list of question dicts."""
    ds = load_dataset("lmms-lab/MP-DocVQA", split=split)
    questions = []
    for i in range(len(ds)):
        if max_questions and i >= max_questions:
            break
        ex = ds[i]
        images = []
        for j in range(1, 21):
            img = ex[f"image_{j}"]
            if img is not None:
                images.append(img)
        answer_idx = int(ex["answer_page_idx"])
        if answer_idx >= len(images):
            continue
        questions.append({
            "qid": str(ex["questionId"]),
            "question": ex["question"],
            "images": images,
            "answer_page_idx": answer_idx,
            "doc_id": ex["doc_id"],
            "n_pages": len(images),
        })
    return questions


def encode_and_cache(questions, model, processor, device, cache_path, batch_doc=1):
    """Encode all page images and queries, cache to disk."""
    if cache_path.exists():
        print(f"Loading cache from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"Encoding {sum(q['n_pages'] for q in questions)} pages across {len(questions)} questions...")
    all_page_embs = []  # list of lists
    all_page_attns = []
    all_page_masks = []
    all_q_embs = []

    for q in tqdm(questions, desc="Encoding questions"):
        # Encode pages
        page_embs, page_attns, page_masks = [], [], []
        for img in q["images"]:
            embs, attns, masks = embed_images_with_attention(
                model, processor, [img], device=device, extract_attention=True,
            )
            page_embs.append(embs[0])
            page_attns.append(attns[0] if attns else None)
            page_masks.append(masks[0] if masks else None)
        all_page_embs.append(page_embs)
        all_page_attns.append(page_attns)
        all_page_masks.append(page_masks)

        # Encode query
        q_emb = embed_queries(model, processor, [q["question"]], device=device)
        all_q_embs.append(q_emb[0])

    data = {
        "page_embs": all_page_embs,
        "page_attns": all_page_attns,
        "page_masks": all_page_masks,
        "q_embs": all_q_embs,
        "questions": [{k: v for k, v in q.items() if k != "images"} for q in questions],
    }
    print(f"Saving cache to {cache_path}")
    torch.save(data, cache_path)
    return data


def maxsim_score(q_emb, d_emb, device):
    """Single query-doc MaxSim score."""
    q = q_emb.unsqueeze(0).to(device)  # [1, Lq, D]
    d = d_emb.unsqueeze(0).to(device)  # [1, Ld, D]
    sim = torch.einsum("qld,bmd->qblm", q, d)  # [1,1,Lq,Ld]
    return sim.max(dim=-1).values.sum().item()


def evaluate(data, pruner, device, global_retrieval=False, **kwargs):
    """Run compression + retrieval, return accuracy@1 and nDCG@5.
    
    If global_retrieval=True, pool all pages into a single corpus and
    retrieve across all documents (harder, more realistic).
    """
    questions = data["questions"]

    # Compress all pages
    all_compressed = []  # list of lists (per question)
    for i in tqdm(range(len(questions)), desc="Compressing"):
        n_pages = questions[i]["n_pages"]
        compressed = []
        for j in range(n_pages):
            emb = data["page_embs"][i][j]
            attn = data["page_attns"][i][j]
            mask = data["page_masks"][i][j]
            if pruner == "identity":
                result = identity_prune(emb)
            elif pruner == "docpruner":
                result = docpruner_prune(emb, attn, k=kwargs["k"], imgpad_mask=mask)
            elif pruner == "docmerger":
                result = docmerger_compress(
                    emb, attn, k1=kwargs["k1"], k2=kwargs["k2"],
                    merge_ratio=kwargs["merge_ratio"], imgpad_mask=mask,
                )
            compressed.append(result.vectors)
        all_compressed.append(compressed)

    if global_retrieval:
        # Build global corpus: flatten all pages with (question_idx, page_idx) mapping
        global_pages = []
        page_origin = []  # (question_idx, local_page_idx)
        for i, pages in enumerate(all_compressed):
            for j, page in enumerate(pages):
                global_pages.append(page)
                page_origin.append((i, j))

        n_global = len(global_pages)
        print(f"  Global retrieval: {n_global} pages in corpus")

        correct_at1 = 0
        ndcg5_sum = 0.0
        total = 0

        for i in tqdm(range(len(questions)), desc="Global retrieval"):
            q_emb = data["q_embs"][i]
            answer_qi = i
            answer_pi = questions[i]["answer_page_idx"]

            # Score all global pages
            scores = []
            for gi, page in enumerate(global_pages):
                s = maxsim_score(q_emb, page, device)
                scores.append((s, gi))
            scores.sort(reverse=True)

            # Check if the correct page is ranked first
            target = (answer_qi, answer_pi)
            if page_origin[scores[0][1]] == target:
                correct_at1 += 1

            dcg = 0.0
            for rank, (_, gi) in enumerate(scores[:5]):
                if page_origin[gi] == target:
                    dcg = 1.0 / (rank + 1)
                    break
            ndcg5_sum += dcg
            total += 1

        return correct_at1 / total, ndcg5_sum / total, total

    if global_retrieval:
        # Build global corpus
        global_pages = []
        page_origin = []  # (question_idx, local_page_idx)
        for i, pages in enumerate(all_compressed):
            for j, page in enumerate(pages):
                global_pages.append(page)
                page_origin.append((i, j))

        n_global = len(global_pages)
        print(f"  Global retrieval: {len(questions)} queries × {n_global} pages")

        # Build query/corpus ids for maxsim_retrieval
        q_ids = [str(i) for i in range(len(questions))]
        c_ids = [f"{qi}_{pi}" for qi, pi in page_origin]
        q_embs = [data["q_embs"][i] for i in range(len(questions))]

        run = maxsim_retrieval(q_embs, global_pages, q_ids, c_ids, device=device)

        # Evaluate
        correct_at1 = 0
        ndcg5_sum = 0.0
        for i in range(len(questions)):
            target_cid = f"{i}_{questions[i]['answer_page_idx']}"
            qid = str(i)
            ranked = sorted(run[qid].items(), key=lambda x: -x[1])

            if ranked[0][0] == target_cid:
                correct_at1 += 1

            dcg = 0.0
            for rank, (cid, _) in enumerate(ranked[:5]):
                if cid == target_cid:
                    dcg = 1.0 / (rank + 1)
                    break
            ndcg5_sum += dcg

        total = len(questions)
        return correct_at1 / total, ndcg5_sum / total, total

    # Per-question retrieval (original behavior)
    correct_at1 = 0
    ndcg5_sum = 0.0
    total = 0

    for i in range(len(questions)):
        q_emb = data["q_embs"][i]
        answer_idx = questions[i]["answer_page_idx"]

        scores = []
        for j, page in enumerate(all_compressed[i]):
            s = maxsim_score(q_emb, page, device)
            scores.append((s, j))
        scores.sort(reverse=True)

        if scores[0][1] == answer_idx:
            correct_at1 += 1

        dcg = 0.0
        for rank, (_, idx) in enumerate(scores[:5]):
            if idx == answer_idx:
                dcg = 1.0 / (rank + 1)
                break
        ndcg5_sum += dcg
        total += 1

    return correct_at1 / total, ndcg5_sum / total, total


def main():
    p = argparse.ArgumentParser(description="MP-DocVQA benchmark for DocPruner/DocMerger")
    p.add_argument("--split", default="val")
    p.add_argument("--max-questions", type=int, default=500,
                   help="Limit questions (full val=5187, use 500 for fast iteration)")
    p.add_argument("--pruner", choices=["identity", "docpruner", "docmerger"], default="identity")
    p.add_argument("--k", type=float, default=-0.25)
    p.add_argument("--k1", type=float, default=1.0)
    p.add_argument("--k2", type=float, default=0.0)
    p.add_argument("--merge-ratio", type=float, default=0.25)
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default="outputs_replicate/cache_mpdocvqa")
    p.add_argument("--global-retrieval", action="store_true",
                   help="Retrieve across all pages globally (not per-question)")
    args = p.parse_args()

    set_seed(0)
    device = pick_device(args.device)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"mpdocvqa_{args.split}_n{args.max_questions}.pt"

    # Load dataset
    print("Loading MP-DocVQA...")
    questions = load_mpdocvqa(args.split, args.max_questions)
    print(f"Loaded {len(questions)} questions, {sum(q['n_pages'] for q in questions)} total pages")

    # Encode (or load cache)
    if not cache_path.exists():
        model, processor = load_colqwen25("vidore/colqwen2.5-v0.2", device=device, need_attention=True)
        data = encode_and_cache(questions, model, processor, device, cache_path)
        del model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        data = encode_and_cache(questions, None, None, device, cache_path)

    # Evaluate
    mode = "global" if args.global_retrieval else "per-question"
    print(f"\nEvaluating: pruner={args.pruner}, mode={mode}")
    acc1, ndcg5, total = evaluate(
        data, args.pruner, device,
        global_retrieval=args.global_retrieval,
        k=args.k, k1=args.k1, k2=args.k2, merge_ratio=args.merge_ratio,
    )

    print(f"\n{'='*50}")
    print(f"MP-DocVQA ({args.split}, n={total}, {mode})")
    print(f"Pruner: {args.pruner}")
    print(f"Accuracy@1: {acc1:.4f} ({int(acc1*total)}/{total})")
    print(f"nDCG@5:     {ndcg5:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
