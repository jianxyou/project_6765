#!/usr/bin/env python3
"""
Index creation script (Qdrant backend) for multimodal RAG system.
- Render PDF pages -> ColPali embeddings (multivector) -> upsert to Qdrant#!/usr/bin/env python3
Index creation script (Qdrant backend) for multimodal RAG system.
- Render PDF pages -> ColPali embeddings (multivector) -> upsert to Qdrant
- Save metadata.json for change detection (hash, model name, docs list, etc.)
"""
import os


os.environ["HF_HOME"] = "/active_work/hf_models/models--davanstrien--finetune_colpali_v1_2-ufo-4bit"
os.environ["HF_HUB_CACHE"] = "/active_work/hf_models/models--davanstrien--finetune_colpali_v1_2-ufo-4bit"
os.environ["TRANSFORMERS_CACHE"] = "/active_work/hf_models/models--davanstrien--finetune_colpali_v1_2-ufo-4bit"
import json
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


import torch
from PIL import Image
import fitz  # PyMuPDF


from qdrant_client import QdrantClient
from qdrant_client.http import models
import stamina


from DocumentNumberExtractor import extract_document_number_from_pdf




# --------------------- Helpers ---------------------


def pick_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"




def infer_doc_class(pdf_path: str, data_root: Path) -> str:
    """
    infer doc_class from folder under data_root:
    data_root/internal/*.pdf -> internal
    data_root/external/*.pdf -> external
    else unknown
    """
    p = Path(pdf_path).resolve()
    root = data_root.resolve()
    try:
        rel = p.relative_to(root)
        top = rel.parts[0].lower() if rel.parts else "unknown"
        if top in ("internal", "external"):
            return top
        return "unknown"
    except Exception:
        return "unknown"




def render_pdf_pages(pdf_path: str, dpi: int = 150, max_pages: Optional[int] = None) -> Iterable[Tuple[Image.Image, int]]:
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        if max_pages is not None:
            page_count = min(page_count, max_pages)


        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)


        for page_idx in range(page_count):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield img, page_idx
    finally:
        doc.close()




def make_point_id(pdf_path: str, page_index: int, doc_class: str, user: str) -> str:
    ns = uuid.NAMESPACE_URL
    key = f"{doc_class}::{pdf_path}::page::{page_index}::user::{user}"
    return str(uuid.uuid5(ns, key))




def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 128,
    recreate: bool = False,
    indexing_threshold: Optional[int] = None,
):
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False


    if not exists:
        client.create_collection(
            collection_name=collection_name,
            on_disk_payload=True,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                ),
            ),
        )


    if indexing_threshold is not None:
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
        )




@stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(client: QdrantClient, collection_name: str, points: List[models.PointStruct]):
    client.upsert(collection_name=collection_name, points=points, wait=False)
    return True




# --------------------- IndexBuilder (Qdrant) ---------------------


class QdrantIndexBuilder:
    def __init__(
        self,
        data_path: str,
        # Qdrant
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "local-pdf-colpali-binary-new",
        recreate_collection: bool = False,
        indexing_threshold: Optional[int] = 10,
        # ColPali
        model_name: str = "davanstrien/finetune_colpali_v1_2-ufo-4bit",
        processor_name: str = "vidore/colpaligemma-3b-pt-448-base",
        device: Optional[str] = None,
        # rendering
        render_dpi: int = 150,
        max_pages_per_pdf: Optional[int] = None,
        batch_size: int = 4,
        # metadata / rebuild
        metadata_path: str = "/active_work/rag/qdrant_index.metadata.json",
        force_rebuild: bool = False,
        # payload
        user: str = "jianxin",
    ):
        self.data_root = Path(data_path)
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.recreate_collection = recreate_collection
        self.indexing_threshold = indexing_threshold


        self.model_name = model_name
        self.processor_name = processor_name
        self.device = device or pick_device()
        self.render_dpi = render_dpi
        self.max_pages_per_pdf = max_pages_per_pdf
        self.batch_size = batch_size


        self.metadata_path = Path(metadata_path)
        self.force_rebuild = force_rebuild
        self.user = user


        if self.device.startswith("cuda") or self.device == "mps":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32


        self.pdf_files: List[Path] = []
        self.document_numbers: Dict[str, Optional[str]] = {}


    def _scan_pdfs(self):
        self.pdf_files = sorted(self.data_root.rglob("*.pdf"))
        if not self.pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_root}")


    def _get_documents_hash(self) -> str:
        hash_content = ""
        for pdf in self.pdf_files:
            rel = pdf.relative_to(self.data_root)
            st = pdf.stat()
            hash_content += f"{rel}:{st.st_mtime}:{st.st_size}\n"
        return hashlib.md5(hash_content.encode()).hexdigest()


    def _extract_document_numbers(self):
        for pdf in self.pdf_files:
            rel = str(pdf.relative_to(self.data_root))
            try:
                self.document_numbers[rel] = extract_document_number_from_pdf(str(pdf))
            except Exception:
                self.document_numbers[rel] = None


    def _should_rebuild(self) -> bool:
        if self.force_rebuild:
            return True
        if not self.metadata_path.exists():
            return True
        try:
            meta = json.loads(self.metadata_path.read_text())
            if meta.get("documents_hash") != self._get_documents_hash():
                return True
            if meta.get("model_name") != self.model_name:
                return True
            if meta.get("processor_name") != self.processor_name:
                return True
            return False
        except Exception:
            return True


    def _save_metadata(self):
        meta = {
            "data_path": str(self.data_root),
            "documents_hash": self._get_documents_hash(),
            "model_name": self.model_name,
            "processor_name": self.processor_name,
            "collection_name": self.collection_name,
            "qdrant_url": self.qdrant_url,
            "render_dpi": self.render_dpi,
            "num_pdfs": len(self.pdf_files),
            "document_list": [
                {
                    "path": str(pdf.relative_to(self.data_root)),
                    "filename": pdf.name,
                    "document_number": self.document_numbers.get(str(pdf.relative_to(self.data_root))),
                    "doc_class": infer_doc_class(str(pdf), self.data_root),
                }
                for pdf in self.pdf_files
            ],
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(meta, indent=2))


    def build(self) -> bool:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data path {self.data_root} does not exist")


        self._scan_pdfs()
        self._extract_document_numbers()


        if not self._should_rebuild():
            print("[info] Qdrant index up-to-date (by metadata hash). Skipping.")
            return True


        qdrant = QdrantClient(url=self.qdrant_url)
        ensure_collection(
            qdrant,
            self.collection_name,
            vector_size=128,
            recreate=self.recreate_collection,
            indexing_threshold=self.indexing_threshold,
        )


        from colpali_engine.models import ColPali, ColPaliProcessor


        colpali_model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )
        colpali_processor = ColPaliProcessor.from_pretrained(self.processor_name)


        # iterate pages
        pages = []
        for pdf in self.pdf_files:
            pdf_path = str(pdf.resolve())
            doc_class = infer_doc_class(pdf_path, self.data_root)
            for img, page_idx in render_pdf_pages(pdf_path, dpi=self.render_dpi, max_pages=self.max_pages_per_pdf):
                pages.append({
                    "pdf_path": pdf_path,
                    "page_index": page_idx,
                    "image": img,
                    "doc_class": doc_class,
                    "user": self.user,
                    "pdf_rel_path": str(pdf.relative_to(self.data_root)),
                    "document_number": self.document_numbers.get(str(pdf.relative_to(self.data_root))),
                })


        print(f"[info] indexing {len(pages)} pages into Qdrant: {self.collection_name}")


        for i in range(0, len(pages), self.batch_size):
            batch = pages[i:i + self.batch_size]
            images = [x["image"] for x in batch]


            with torch.no_grad():
                batch_images = colpali_processor.process_images(images).to(colpali_model.device)
                image_embeddings = colpali_model(**batch_images)


            points: List[models.PointStruct] = []
            for j, emb in enumerate(image_embeddings):
                meta = batch[j]
                pid = make_point_id(meta["pdf_path"], meta["page_index"], meta["doc_class"], meta["user"])
                multivector = emb.detach().cpu().float().numpy().tolist()


                points.append(
                    models.PointStruct(
                        id=pid,
                        vector=multivector,
                        payload={
                            "source": "local_pdf",
                            "pdf_path": meta["pdf_path"],
                            "pdf_rel_path": meta["pdf_rel_path"],
                            "page_index": int(meta["page_index"]),
                            "render_dpi": int(self.render_dpi),
                            "doc_class": meta["doc_class"],
                            "user": meta["user"],
                            "document_number": meta["document_number"],
                        },
                    )
                )


            upsert_to_qdrant(qdrant, self.collection_name, points)


        self._save_metadata()
        print("[info] ✓ Qdrant indexing complete + metadata saved.")
        return True




def create_index_qdrant(
    data_path: str,
    qdrant_url: str,
    collection_name: str,
    force_rebuild: bool = False,
):
    builder = QdrantIndexBuilder(
        data_path=data_path,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        force_rebuild=force_rebuild,
    )
    return builder.build()




if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="local-pdf-colpali-binary-new")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()


    ok = create_index_qdrant(
        data_path=args.data_path,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        force_rebuild=args.force,
    )
    raise SystemExit(0 if ok else 1)


# - Save metadata.json for change detection (hash, model name, docs list, etc.)
import os


os.environ["HF_HOME"] = "/active_work/hf_models/models--davanstrien--finetune_colpali_v1_2-ufo-4bit"
os.environ["HF_HUB_CACHE"] = "/active_work/hf_models/models--davanstrien--finetune_colpali_v1_2-ufo-4bit"
os.environ["TRANSFORMERS_CACHE"] = "/active_work/hf_models/models--davanstrien--finetune_colpali_v1_2-ufo-4bit"
import json
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


import torch
from PIL import Image
import fitz  # PyMuPDF


from qdrant_client import QdrantClient
from qdrant_client.http import models
import stamina


from DocumentNumberExtractor import extract_document_number_from_pdf




# --------------------- Helpers ---------------------


def pick_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"




def infer_doc_class(pdf_path: str, data_root: Path) -> str:
    """
    infer doc_class from folder under data_root:
    data_root/internal/*.pdf -> internal
    data_root/external/*.pdf -> external
    else unknown
    """
    p = Path(pdf_path).resolve()
    root = data_root.resolve()
    try:
        rel = p.relative_to(root)
        top = rel.parts[0].lower() if rel.parts else "unknown"
        if top in ("internal", "external"):
            return top
        return "unknown"
    except Exception:
        return "unknown"




def render_pdf_pages(pdf_path: str, dpi: int = 150, max_pages: Optional[int] = None) -> Iterable[Tuple[Image.Image, int]]:
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        if max_pages is not None:
            page_count = min(page_count, max_pages)


        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)


        for page_idx in range(page_count):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield img, page_idx
    finally:
        doc.close()




def make_point_id(pdf_path: str, page_index: int, doc_class: str, user: str) -> str:
    ns = uuid.NAMESPACE_URL
    key = f"{doc_class}::{pdf_path}::page::{page_index}::user::{user}"
    return str(uuid.uuid5(ns, key))




def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 128,
    recreate: bool = False,
    indexing_threshold: Optional[int] = None,
):
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False


    if not exists:
        client.create_collection(
            collection_name=collection_name,
            on_disk_payload=True,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                ),
            ),
        )


    if indexing_threshold is not None:
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
        )




@stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(client: QdrantClient, collection_name: str, points: List[models.PointStruct]):
    client.upsert(collection_name=collection_name, points=points, wait=False)
    return True




# --------------------- IndexBuilder (Qdrant) ---------------------


class QdrantIndexBuilder:
    def __init__(
        self,
        data_path: str,
        # Qdrant
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "local-pdf-colpali-binary-new",
        recreate_collection: bool = False,
        indexing_threshold: Optional[int] = 10,
        # ColPali
        model_name: str = "davanstrien/finetune_colpali_v1_2-ufo-4bit",
        processor_name: str = "vidore/colpaligemma-3b-pt-448-base",
        device: Optional[str] = None,
        # rendering
        render_dpi: int = 150,
        max_pages_per_pdf: Optional[int] = None,
        batch_size: int = 4,
        # metadata / rebuild
        metadata_path: str = "/active_work/rag/qdrant_index.metadata.json",
        force_rebuild: bool = False,
        # payload
        user: str = "jianxin",
    ):
        self.data_root = Path(data_path)
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.recreate_collection = recreate_collection
        self.indexing_threshold = indexing_threshold


        self.model_name = model_name
        self.processor_name = processor_name
        self.device = device or pick_device()
        self.render_dpi = render_dpi
        self.max_pages_per_pdf = max_pages_per_pdf
        self.batch_size = batch_size


        self.metadata_path = Path(metadata_path)
        self.force_rebuild = force_rebuild
        self.user = user


        if self.device.startswith("cuda") or self.device == "mps":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32


        self.pdf_files: List[Path] = []
        self.document_numbers: Dict[str, Optional[str]] = {}


    def _scan_pdfs(self):
        self.pdf_files = sorted(self.data_root.rglob("*.pdf"))
        if not self.pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_root}")


    def _get_documents_hash(self) -> str:
        hash_content = ""
        for pdf in self.pdf_files:
            rel = pdf.relative_to(self.data_root)
            st = pdf.stat()
            hash_content += f"{rel}:{st.st_mtime}:{st.st_size}\n"
        return hashlib.md5(hash_content.encode()).hexdigest()


    def _extract_document_numbers(self):
        for pdf in self.pdf_files:
            rel = str(pdf.relative_to(self.data_root))
            try:
                self.document_numbers[rel] = extract_document_number_from_pdf(str(pdf))
            except Exception:
                self.document_numbers[rel] = None


    def _should_rebuild(self) -> bool:
        if self.force_rebuild:
            return True
        if not self.metadata_path.exists():
            return True
        try:
            meta = json.loads(self.metadata_path.read_text())
            if meta.get("documents_hash") != self._get_documents_hash():
                return True
            if meta.get("model_name") != self.model_name:
                return True
            if meta.get("processor_name") != self.processor_name:
                return True
            return False
        except Exception:
            return True


    def _save_metadata(self):
        meta = {
            "data_path": str(self.data_root),
            "documents_hash": self._get_documents_hash(),
            "model_name": self.model_name,
            "processor_name": self.processor_name,
            "collection_name": self.collection_name,
            "qdrant_url": self.qdrant_url,
            "render_dpi": self.render_dpi,
            "num_pdfs": len(self.pdf_files),
            "document_list": [
                {
                    "path": str(pdf.relative_to(self.data_root)),
                    "filename": pdf.name,
                    "document_number": self.document_numbers.get(str(pdf.relative_to(self.data_root))),
                    "doc_class": infer_doc_class(str(pdf), self.data_root),
                }
                for pdf in self.pdf_files
            ],
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(meta, indent=2))


    def build(self) -> bool:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data path {self.data_root} does not exist")


        self._scan_pdfs()
        self._extract_document_numbers()


        if not self._should_rebuild():
            print("[info] Qdrant index up-to-date (by metadata hash). Skipping.")
            return True


        qdrant = QdrantClient(url=self.qdrant_url)
        ensure_collection(
            qdrant,
            self.collection_name,
            vector_size=128,
            recreate=self.recreate_collection,
            indexing_threshold=self.indexing_threshold,
        )


        from colpali_engine.models import ColPali, ColPaliProcessor


        colpali_model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )
        colpali_processor = ColPaliProcessor.from_pretrained(self.processor_name)


        # iterate pages
        pages = []
        for pdf in self.pdf_files:
            pdf_path = str(pdf.resolve())
            doc_class = infer_doc_class(pdf_path, self.data_root)
            for img, page_idx in render_pdf_pages(pdf_path, dpi=self.render_dpi, max_pages=self.max_pages_per_pdf):
                pages.append({
                    "pdf_path": pdf_path,
                    "page_index": page_idx,
                    "image": img,
                    "doc_class": doc_class,
                    "user": self.user,
                    "pdf_rel_path": str(pdf.relative_to(self.data_root)),
                    "document_number": self.document_numbers.get(str(pdf.relative_to(self.data_root))),
                })


        print(f"[info] indexing {len(pages)} pages into Qdrant: {self.collection_name}")


        for i in range(0, len(pages), self.batch_size):
            batch = pages[i:i + self.batch_size]
            images = [x["image"] for x in batch]


            with torch.no_grad():
                batch_images = colpali_processor.process_images(images).to(colpali_model.device)
                image_embeddings = colpali_model(**batch_images)


            points: List[models.PointStruct] = []
            for j, emb in enumerate(image_embeddings):
                meta = batch[j]
                pid = make_point_id(meta["pdf_path"], meta["page_index"], meta["doc_class"], meta["user"])
                multivector = emb.detach().cpu().float().numpy().tolist()


                points.append(
                    models.PointStruct(
                        id=pid,
                        vector=multivector,
                        payload={
                            "source": "local_pdf",
                            "pdf_path": meta["pdf_path"],
                            "pdf_rel_path": meta["pdf_rel_path"],
                            "page_index": int(meta["page_index"]),
                            "render_dpi": int(self.render_dpi),
                            "doc_class": meta["doc_class"],
                            "user": meta["user"],
                            "document_number": meta["document_number"],
                        },
                    )
                )


            upsert_to_qdrant(qdrant, self.collection_name, points)


        self._save_metadata()
        print("[info] ✓ Qdrant indexing complete + metadata saved.")
        return True




def create_index_qdrant(
    data_path: str,
    qdrant_url: str,
    collection_name: str,
    force_rebuild: bool = False,
):
    builder = QdrantIndexBuilder(
        data_path=data_path,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        force_rebuild=force_rebuild,
    )
    return builder.build()




if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="local-pdf-colpali-binary-new")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()


    ok = create_index_qdrant(
        data_path=args.data_path,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        force_rebuild=args.force,
    )
    raise SystemExit(0 if ok else 1)



