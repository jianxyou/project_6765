"""
ViDoRe-V2 BEIR-format dataset loading.
"""

from typing import Dict, List, Tuple

from datasets import load_dataset
from PIL import Image

from .utils import to_pil

VIDORE_V2_DATASETS = [
    "vidore/esg_reports_v2",
    "vidore/biomedical_lectures_eng_v2",
    "vidore/economics_reports_v2",
    "vidore/esg_reports_human_labeled_v2",
]

DOCPRUNER_K_VALUES = [-0.5, -0.25, 0, 0.25, 0.5, 1.0]


def _get_field(row, candidates):
    for c in candidates:
        if c in row:
            return row[c]
    raise KeyError(f"Field not found, tried {candidates}, actual: {list(row.keys())}")


def load_vidore_v2(
    dataset_name: str, split: str = "test", language: str = "english"
) -> Tuple[List[str], List[Image.Image], List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Load a ViDoRe-V2 BEIR-format dataset.

    Returns:
        corpus_ids, corpus_images, query_ids, query_texts, qrels
    """
    print(f"  Loading corpus...")
    corpus_ds = load_dataset(dataset_name, "corpus", split=split)
    print(f"  Loading queries...")
    queries_ds = load_dataset(dataset_name, "queries", split=split)
    print(f"  Loading qrels...")
    qrels_ds = load_dataset(dataset_name, "qrels", split=split)

    print(f"  Corpus fields: {corpus_ds.column_names}")
    print(f"  Queries fields: {queries_ds.column_names}")
    print(f"  Qrels fields: {qrels_ds.column_names}")

    # Corpus
    corpus_ids = []
    corpus_images = []
    for row in corpus_ds:
        doc_id = str(_get_field(row, ["corpus-id", "corpus_id", "doc-id", "doc_id", "_id"]))
        corpus_ids.append(doc_id)
        corpus_images.append(to_pil(row["image"]))

    # Queries (filtered by language)
    query_ids = []
    query_texts = []
    for row in queries_ds:
        qid = str(_get_field(row, ["query-id", "query_id", "_id", "id", "qid"]))
        if language and row.get("language", "english") != language:
            continue
        query_ids.append(qid)
        query_texts.append(str(_get_field(row, ["query", "text", "question", "content"])))

    # Qrels
    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(_get_field(row, ["query-id", "query_id", "qid"]))
        did = str(_get_field(row, ["corpus-id", "corpus_id", "doc-id", "doc_id", "docid"]))
        score = int(_get_field(row, ["score", "relevance", "label"]))
        qrels.setdefault(qid, {})[did] = score

    # Filter queries to those with qrels
    filtered_qids, filtered_texts = [], []
    for qid, text in zip(query_ids, query_texts):
        if qid in qrels:
            filtered_qids.append(qid)
            filtered_texts.append(text)

    print(f"  Corpus: {len(corpus_ids)} docs")
    print(f"  Queries: {len(filtered_qids)} (language={language})")
    print(f"  Qrels: {sum(len(v) for v in qrels.values())} annotations")

    return corpus_ids, corpus_images, filtered_qids, filtered_texts, qrels
