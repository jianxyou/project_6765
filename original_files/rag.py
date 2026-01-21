# !/usr/bin/env python3
# """#!/usr/bin/env python3
"""
Query script for multimodal RAG system (Qdrant backend).
- Query text -> ColPali query embedding (multivector) -> Qdrant query_points
- Fetch top-k pages via payload(pdf_path, page_index) -> render -> feed images to VLM
"""
import os


os.environ["HF_HOME"] = "/active_work/hf_models"
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from pathlib import Path
import time


import torch
from PIL import Image
import fitz  # PyMuPDF


from qdrant_client import QdrantClient
from qdrant_client.http import models


from framework.models.model_provider import get_model
from langchain_core.messages import HumanMessage




class MultimodalRAGQuery:
    """RAG query system that uses a pre-built Qdrant ColPali collection."""


    def __init__(
        self,
        data_path: str,
        model_name: str = "qwen2.5vl:7b",
        # kept arg name "index_name" to minimize main changes, but now it means "collection_name"
        index_name: str = "local-pdf-colpali-binary-new",
        device: str = "cuda:0",
        qdrant_url: str = "http://localhost:6333",
        colpali_model_name: str = "davanstrien/finetune_colpali_v1_2-ufo-4bit",
        colpali_processor_name: str = "vidore/colpaligemma-3b-pt-448-base",
        render_dpi: int = 150,
        query_timeout: int = 100,
        filter_doc_class: Optional[str] = None,  # "internal"/"external"/None
    ):
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.collection_name = index_name  # <-- interpret as Qdrant collection
        self.device = device
        self.qdrant_url = qdrant_url


        self.colpali_model_name = colpali_model_name
        self.colpali_processor_name = colpali_processor_name


        self.render_dpi = int(render_dpi)
        self.query_timeout = int(query_timeout)
        self.filter_doc_class = filter_doc_class


        # Cache for rendered pages: (pdf_path, page_index0) -> PIL.Image
        self.page_cache: Dict[tuple, Image.Image] = {}


        # Initialize the VLM
        print(f"Initializing VLM: {model_name}", flush=True)
        self.vlm = get_model(model_name)


        # Load retrieval stack (Qdrant + ColPali query encoder)
        self._load_index()


    # -------------------- Loading --------------------


    def _load_index(self):
        """Initialize Qdrant client and ColPali query encoder."""
        print(f"Connecting to Qdrant: {self.qdrant_url}", flush=True)
        self.qdrant = QdrantClient(url=self.qdrant_url)


        # Sanity: collection exists?
        try:
            exists = self.qdrant.collection_exists(self.collection_name)
        except Exception as e:
            raise RuntimeError(f"Failed to reach Qdrant at {self.qdrant_url}: {e}")


        if not exists:
            raise FileNotFoundError(
                f"Qdrant collection not found: {self.collection_name}\n"
                f"Please create it first using your indexing script."
            )


        # Load ColPali
        from colpali_engine.models import ColPali, ColPaliProcessor


        # Resolve device + dtype
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device_map = self.device if ":" in self.device else "cuda:0"
            torch_dtype = torch.float16
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
            torch_dtype = torch.float16
        else:
            device_map = "cpu"
            torch_dtype = torch.float32


        self._device_map = device_map


        print(f"Loading ColPali model on {device_map} (dtype={torch_dtype})", flush=True)
        self.colpali_model = ColPali.from_pretrained(
            self.colpali_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.colpali_processor = ColPaliProcessor.from_pretrained(self.colpali_processor_name)


        print(f"✓ Loaded Qdrant collection '{self.collection_name}' and ColPali encoder", flush=True)


    # -------------------- Image helpers --------------------


    def _pil_to_base64(self, img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


    def _create_image_message_content(self, images: List[Image.Image]) -> List[Dict]:
        content = []
        for img in images:
            img_b64 = self._pil_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
        return content


    def _render_pdf_page(self, pdf_path: str, page_index0: int) -> Image.Image:
        """Render a single PDF page to PIL (cached). page_index0 is 0-based."""
        key = (pdf_path, page_index0)
        if key in self.page_cache:
            return self.page_cache[key]


        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(page_index0)
            zoom = self.render_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()


        self.page_cache[key] = img
        return img


    def clear_page_cache(self):
        self.page_cache.clear()
        print("Page cache cleared", flush=True)


    # -------------------- Qdrant retrieval --------------------


    def _build_doc_class_filter(self, doc_class: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_class",
                    match=models.MatchValue(value=doc_class),
                )
            ]
        )


    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search Qdrant for relevant PDF pages using ColPali query multivector."""
        if not query:
            return []


        # optional payload filter
        query_filter = None
        if self.filter_doc_class:
            query_filter = self._build_doc_class_filter(self.filter_doc_class)


        # encode query -> multivector
        with torch.no_grad():
            batch_query = self.colpali_processor.process_queries([query]).to(self.colpali_model.device)
            query_embedding = self.colpali_model(**batch_query)


        multivector_query = query_embedding[0].detach().cpu().float().numpy().tolist()


        # qdrant query_points
        start = time.time()
        res = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=multivector_query,
            limit=k,
            timeout=self.query_timeout,
            query_filter=query_filter,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
        )
        elapsed = time.time() - start
        print(f"[info] Qdrant search done in {elapsed:.3f}s, got {len(res.points or [])} hits", flush=True)


        processed = []
        for p in (res.points or []):
            payload = p.payload or {}
            pdf_path = payload.get("pdf_path")
            page_index = payload.get("page_index", 0)


            try:
                page_index = int(page_index)
            except Exception:
                page_index = 0


            processed.append({
                # keep keys similar to old byaldi output to minimize changes below
                "pdf_path": pdf_path,
                "page_num": page_index,   # NOTE: your index stores 0-based page_index
                "score": float(p.score) if p.score is not None else 0.0,
                "payload": payload,
                "id": str(p.id),
            })


        return processed


    def _get_grouped_images(self, results: List[Dict]) -> List[Image.Image]:
        """Render images for top-k results."""
        grouped_images = []
        for r in results:
            pdf_path = r.get("pdf_path") or (r.get("payload") or {}).get("pdf_path")
            page_index0 = int(r.get("page_num", 0))


            if not pdf_path:
                continue


            if not Path(pdf_path).exists():
                print(f"[warn] pdf_path not found on this machine: {pdf_path}", flush=True)
                continue


            img = self._render_pdf_page(pdf_path, page_index0)
            grouped_images.append(img)
            print(f"  Retrieved: {Path(pdf_path).name}, page_index={page_index0}", flush=True)


        return grouped_images


    # -------------------- Optional image-to-text for retrieval --------------------


    def _generate_image_description(self, image: Image.Image) -> str:
        """Use VLM to generate a search-optimized description (same as your old logic)."""
        img_b64 = self._pil_to_base64(image)


        prompt = """Analyze this image carefully and describe what you see.


        Focus on:
        1. What programming language or technical domain is shown?
        2. What specific topics, functions, or concepts are visible?
        3. What type of content is this? (code, diagram, documentation, etc.)


        Be VERY SPECIFIC. Include:
        - Exact programming language if it's code
        - Visible function names, class names, or technical terms
        - The main topic or purpose


        Do NOT hallucinate or add information not visible in the image.
        Generate a detailed, specific description:"""


        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]
        messages = [HumanMessage(content=content)]
        response = self.vlm.invoke(messages)
        desc = response.content if hasattr(response, "content") else str(response)
        print(f"Generated image description: {desc}", flush=True)
        return desc.strip()


    # -------------------- RAG answer --------------------


    def query(self, query: str = "", query_image=None, summary_focus: str = "", k: int = 3) -> Dict[str, Any]:
        search_query = query
        query_images = []


        if query_image is not None:
            print("\nProcessing query image...", flush=True)
            img_description = self._generate_image_description(query_image)
            query_images.append(query_image)


            if query:
                search_query = f"{query} {img_description}"
                print("Combined query: text + image description", flush=True)
            else:
                search_query = img_description
                print("Image-only query (via description)", flush=True)


        if not search_query:
            return {"answer": "Please provide a text query or image.", "results": [], "metadata": {"total_results": 0, "k_retrieved": k}}


        print(f"\nSearching for: {search_query}", flush=True)
        results = self.search(search_query, k=k)


        if not results:
            return {"answer": "No relevant pages found.", "results": [], "metadata": {"total_results": 0, "k_retrieved": k}}


        print(f"Found {len(results)} results, retrieving top {k}...", flush=True)
        grouped_images = self._get_grouped_images(results[:k])


        image_content = []
        if query_images:
            image_content.extend(self._create_image_message_content(query_images))
        image_content.extend(self._create_image_message_content(grouped_images))


        rag_prompt = f"""You are a helpful assistant. Given images and user input, address the user input to the best of your ability.


Context:
UCC stands for Ultra Compact Core.
The user searched for "{query if query else '[image query]'}"
"""
        if query_images:
            rag_prompt += "The user provided an image as part of their query (shown first).\n"


        rag_prompt += f"""Use the document images to answer the user's input.


User: {summary_focus if summary_focus else (query if query else 'Please analyze the provided image and retrieved documents.')}
"""


        messages = [HumanMessage(content=image_content + [{"type": "text", "text": rag_prompt}])]


        print("Generating answer...", flush=True)
        try:
            response = self.vlm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Error querying VLM: {str(e)}"


        return {
            "answer": answer,
            "results": results[:k],
            "metadata": {
                "total_results": len(results),
                "k_retrieved": k,
                "query_type": "image+text" if (query and query_image) else ("image" if query_image else "text"),
            },
        }


    # --- keep these methods for ask_rag compatibility (minimal changes) ---


    def get_document_name(self, doc_id_unused: int) -> str:
        return "unknown"


    def get_document_number(self, doc_id_unused: int) -> Optional[str]:
        return None




def create_rag_query(
    data_path: str,
    model_name: str = "qwen2.5vl:7b",
    index_name: str = "local-pdf-colpali-binary-new",  # now collection name
    device: str = "cuda:0",
) -> MultimodalRAGQuery:
    return MultimodalRAGQuery(
        data_path=data_path,
        model_name=model_name,
        index_name=index_name,
        device=device,
    )




def ask_rag(
    query: str,
    summary_focus: str,
    k: int,
    rag_instance: MultimodalRAGQuery,
    query_image=None,
) -> Dict[str, Any]:
    start_time = time.time()
    query_result = rag_instance.query(query, query_image=query_image, summary_focus=summary_focus, k=k)
    answer = query_result["answer"]
    results = query_result["results"]


    formatted_results = []
    for r in results:
        payload = r.get("payload", {}) or {}
        pdf_path = r.get("pdf_path") or payload.get("pdf_path", "")
        page_num = r.get("page_num", 0)
        score = r.get("score", 0.0)


        formatted_results.append({
            "content": "",
            "document_number": payload.get("document_number", None),
            "source": Path(pdf_path).name if pdf_path else "unknown",
            "page": int(page_num),
            "reranker_score": float(score),
        })


    query_key = query if query else "[image query]"
    return {
        query_key: {
            "full_response": answer,
            "final_answer": answer,
            "before_reranking": formatted_results,
            "after_reranking": formatted_results,
            "query_time": time.time() - start_time,
            "metadata": {
                "total_chunks": query_result["metadata"]["total_results"],
                "candidates_considered": k,
                "quality_filtered": 0,
                "final_sources": len(formatted_results),
                "reranker_used": False,
                "query_type": query_result["metadata"].get("query_type", "text"),
            },
        }
    }




# Example usage
if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Query multimodal RAG system (Qdrant backend)")
    parser.add_argument("data_path", help="Path to folder containing PDF files (used only for local rendering)")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="VLM model to use")
    # keep flag name for minimal change; now it's collection name
    parser.add_argument("--index-name", default="local-pdf-colpali-binary-new", help="Qdrant collection name")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--query", required=True, help="Single text query (text-only mode)")
    parser.add_argument("--doc-class", default=None, help="Optional payload filter: internal/external")


    args = parser.parse_args()


    rag = MultimodalRAGQuery(
        data_path=args.data_path,
        model_name=args.model,
        index_name=args.index_name,
        device=args.device,
        filter_doc_class=args.doc_class,
    )


    result = ask_rag(args.query, "", k=args.k, rag_instance=rag)
    print(f"\nAnswer: {result[args.query]['final_answer']}")
    print(f"Query time: {result[args.query]['query_time']:.2f}s")
    print(f"Sources: {result[args.query]['metadata']['final_sources']}")



"""
Query script for multimodal RAG system (Qdrant backend).
- Query text -> ColPali query embedding (multivector) -> Qdrant query_points
- Fetch top-k pages via payload(pdf_path, page_index) -> render -> feed images to VLM
"""
import os


os.environ["HF_HOME"] = "/active_work/hf_models"
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from pathlib import Path
import time


import torch
from PIL import Image
import fitz  # PyMuPDF


from qdrant_client import QdrantClient
from qdrant_client.http import models


from framework.models.model_provider import get_model
from langchain_core.messages import HumanMessage




class MultimodalRAGQuery:
    """RAG query system that uses a pre-built Qdrant ColPali collection."""


    def __init__(
        self,
        data_path: str,
        model_name: str = "qwen2.5vl:7b",
        # kept arg name "index_name" to minimize main changes, but now it means "collection_name"
        index_name: str = "local-pdf-colpali-binary-new",
        device: str = "cuda:0",
        qdrant_url: str = "http://localhost:6333",
        colpali_model_name: str = "davanstrien/finetune_colpali_v1_2-ufo-4bit",
        colpali_processor_name: str = "vidore/colpaligemma-3b-pt-448-base",
        render_dpi: int = 150,
        query_timeout: int = 100,
        filter_doc_class: Optional[str] = None,  # "internal"/"external"/None
    ):
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.collection_name = index_name  # <-- interpret as Qdrant collection
        self.device = device
        self.qdrant_url = qdrant_url


        self.colpali_model_name = colpali_model_name
        self.colpali_processor_name = colpali_processor_name


        self.render_dpi = int(render_dpi)
        self.query_timeout = int(query_timeout)
        self.filter_doc_class = filter_doc_class


        # Cache for rendered pages: (pdf_path, page_index0) -> PIL.Image
        self.page_cache: Dict[tuple, Image.Image] = {}


        # Initialize the VLM
        print(f"Initializing VLM: {model_name}", flush=True)
        self.vlm = get_model(model_name)


        # Load retrieval stack (Qdrant + ColPali query encoder)
        self._load_index()


    # -------------------- Loading --------------------


    def _load_index(self):
        """Initialize Qdrant client and ColPali query encoder."""
        print(f"Connecting to Qdrant: {self.qdrant_url}", flush=True)
        self.qdrant = QdrantClient(url=self.qdrant_url)


        # Sanity: collection exists?
        try:
            exists = self.qdrant.collection_exists(self.collection_name)
        except Exception as e:
            raise RuntimeError(f"Failed to reach Qdrant at {self.qdrant_url}: {e}")


        if not exists:
            raise FileNotFoundError(
                f"Qdrant collection not found: {self.collection_name}\n"
                f"Please create it first using your indexing script."
            )


        # Load ColPali
        from colpali_engine.models import ColPali, ColPaliProcessor


        # Resolve device + dtype
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device_map = self.device if ":" in self.device else "cuda:0"
            torch_dtype = torch.float16
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
            torch_dtype = torch.float16
        else:
            device_map = "cpu"
            torch_dtype = torch.float32


        self._device_map = device_map


        print(f"Loading ColPali model on {device_map} (dtype={torch_dtype})", flush=True)
        self.colpali_model = ColPali.from_pretrained(
            self.colpali_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.colpali_processor = ColPaliProcessor.from_pretrained(self.colpali_processor_name)


        print(f"✓ Loaded Qdrant collection '{self.collection_name}' and ColPali encoder", flush=True)


    # -------------------- Image helpers --------------------


    def _pil_to_base64(self, img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


    def _create_image_message_content(self, images: List[Image.Image]) -> List[Dict]:
        content = []
        for img in images:
            img_b64 = self._pil_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
        return content


    def _render_pdf_page(self, pdf_path: str, page_index0: int) -> Image.Image:
        """Render a single PDF page to PIL (cached). page_index0 is 0-based."""
        key = (pdf_path, page_index0)
        if key in self.page_cache:
            return self.page_cache[key]


        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(page_index0)
            zoom = self.render_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()


        self.page_cache[key] = img
        return img


    def clear_page_cache(self):
        self.page_cache.clear()
        print("Page cache cleared", flush=True)


    # -------------------- Qdrant retrieval --------------------


    def _build_doc_class_filter(self, doc_class: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_class",
                    match=models.MatchValue(value=doc_class),
                )
            ]
        )


    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search Qdrant for relevant PDF pages using ColPali query multivector."""
        if not query:
            return []


        # optional payload filter
        query_filter = None
        if self.filter_doc_class:
            query_filter = self._build_doc_class_filter(self.filter_doc_class)


        # encode query -> multivector
        with torch.no_grad():
            batch_query = self.colpali_processor.process_queries([query]).to(self.colpali_model.device)
            query_embedding = self.colpali_model(**batch_query)


        multivector_query = query_embedding[0].detach().cpu().float().numpy().tolist()


        # qdrant query_points
        start = time.time()
        res = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=multivector_query,
            limit=k,
            timeout=self.query_timeout,
            query_filter=query_filter,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
        )
        elapsed = time.time() - start
        print(f"[info] Qdrant search done in {elapsed:.3f}s, got {len(res.points or [])} hits", flush=True)


        processed = []
        for p in (res.points or []):
            payload = p.payload or {}
            pdf_path = payload.get("pdf_path")
            page_index = payload.get("page_index", 0)


            try:
                page_index = int(page_index)
            except Exception:
                page_index = 0


            processed.append({
                # keep keys similar to old byaldi output to minimize changes below
                "pdf_path": pdf_path,
                "page_num": page_index,   # NOTE: your index stores 0-based page_index
                "score": float(p.score) if p.score is not None else 0.0,
                "payload": payload,
                "id": str(p.id),
            })


        return processed


    def _get_grouped_images(self, results: List[Dict]) -> List[Image.Image]:
        """Render images for top-k results."""
        grouped_images = []
        for r in results:
            pdf_path = r.get("pdf_path") or (r.get("payload") or {}).get("pdf_path")
            page_index0 = int(r.get("page_num", 0))


            if not pdf_path:
                continue


            if not Path(pdf_path).exists():
                print(f"[warn] pdf_path not found on this machine: {pdf_path}", flush=True)
                continue


            img = self._render_pdf_page(pdf_path, page_index0)
            grouped_images.append(img)
            print(f"  Retrieved: {Path(pdf_path).name}, page_index={page_index0}", flush=True)


        return grouped_images


    # -------------------- Optional image-to-text for retrieval --------------------


    def _generate_image_description(self, image: Image.Image) -> str:
        """Use VLM to generate a search-optimized description (same as your old logic)."""
        img_b64 = self._pil_to_base64(image)


        prompt = """Analyze this image carefully and describe what you see.


        Focus on:
        1. What programming language or technical domain is shown?
        2. What specific topics, functions, or concepts are visible?
        3. What type of content is this? (code, diagram, documentation, etc.)


        Be VERY SPECIFIC. Include:
        - Exact programming language if it's code
        - Visible function names, class names, or technical terms
        - The main topic or purpose


        Do NOT hallucinate or add information not visible in the image.
        Generate a detailed, specific description:"""


        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]
        messages = [HumanMessage(content=content)]
        response = self.vlm.invoke(messages)
        desc = response.content if hasattr(response, "content") else str(response)
        print(f"Generated image description: {desc}", flush=True)
        return desc.strip()


    # -------------------- RAG answer --------------------


    def query(self, query: str = "", query_image=None, summary_focus: str = "", k: int = 3) -> Dict[str, Any]:
        search_query = query
        query_images = []


        if query_image is not None:
            print("\nProcessing query image...", flush=True)
            img_description = self._generate_image_description(query_image)
            query_images.append(query_image)


            if query:
                search_query = f"{query} {img_description}"
                print("Combined query: text + image description", flush=True)
            else:
                search_query = img_description
                print("Image-only query (via description)", flush=True)


        if not search_query:
            return {"answer": "Please provide a text query or image.", "results": [], "metadata": {"total_results": 0, "k_retrieved": k}}


        print(f"\nSearching for: {search_query}", flush=True)
        results = self.search(search_query, k=k)


        if not results:
            return {"answer": "No relevant pages found.", "results": [], "metadata": {"total_results": 0, "k_retrieved": k}}


        print(f"Found {len(results)} results, retrieving top {k}...", flush=True)
        grouped_images = self._get_grouped_images(results[:k])


        image_content = []
        if query_images:
            image_content.extend(self._create_image_message_content(query_images))
        image_content.extend(self._create_image_message_content(grouped_images))


        rag_prompt = f"""You are a helpful assistant. Given images and user input, address the user input to the best of your ability.


Context:
UCC stands for Ultra Compact Core.
The user searched for "{query if query else '[image query]'}"
"""
        if query_images:
            rag_prompt += "The user provided an image as part of their query (shown first).\n"


        rag_prompt += f"""Use the document images to answer the user's input.


User: {summary_focus if summary_focus else (query if query else 'Please analyze the provided image and retrieved documents.')}
"""


        messages = [HumanMessage(content=image_content + [{"type": "text", "text": rag_prompt}])]


        print("Generating answer...", flush=True)
        try:
            response = self.vlm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Error querying VLM: {str(e)}"


        return {
            "answer": answer,
            "results": results[:k],
            "metadata": {
                "total_results": len(results),
                "k_retrieved": k,
                "query_type": "image+text" if (query and query_image) else ("image" if query_image else "text"),
            },
        }


    # --- keep these methods for ask_rag compatibility (minimal changes) ---


    def get_document_name(self, doc_id_unused: int) -> str:
        return "unknown"


    def get_document_number(self, doc_id_unused: int) -> Optional[str]:
        return None




def create_rag_query(
    data_path: str,
    model_name: str = "qwen2.5vl:7b",
    index_name: str = "local-pdf-colpali-binary-new",  # now collection name
    device: str = "cuda:0",
) -> MultimodalRAGQuery:
    return MultimodalRAGQuery(
        data_path=data_path,
        model_name=model_name,
        index_name=index_name,
        device=device,
    )




def ask_rag(
    query: str,
    summary_focus: str,
    k: int,
    rag_instance: MultimodalRAGQuery,
    query_image=None,
) -> Dict[str, Any]:
    start_time = time.time()
    query_result = rag_instance.query(query, query_image=query_image, summary_focus=summary_focus, k=k)
    answer = query_result["answer"]
    results = query_result["results"]


    formatted_results = []
    for r in results:
        payload = r.get("payload", {}) or {}
        pdf_path = r.get("pdf_path") or payload.get("pdf_path", "")
        page_num = r.get("page_num", 0)
        score = r.get("score", 0.0)


        formatted_results.append({
            "content": "",
            "document_number": payload.get("document_number", None),
            "source": Path(pdf_path).name if pdf_path else "unknown",
            "page": int(page_num),
            "reranker_score": float(score),
        })


    query_key = query if query else "[image query]"
    return {
        query_key: {
            "full_response": answer,
            "final_answer": answer,
            "before_reranking": formatted_results,
            "after_reranking": formatted_results,
            "query_time": time.time() - start_time,
            "metadata": {
                "total_chunks": query_result["metadata"]["total_results"],
                "candidates_considered": k,
                "quality_filtered": 0,
                "final_sources": len(formatted_results),
                "reranker_used": False,
                "query_type": query_result["metadata"].get("query_type", "text"),
            },
        }
    }




# Example usage
if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Query multimodal RAG system (Qdrant backend)")
    parser.add_argument("data_path", help="Path to folder containing PDF files (used only for local rendering)")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="VLM model to use")
    # keep flag name for minimal change; now it's collection name
    parser.add_argument("--index-name", default="local-pdf-colpali-binary-new", help="Qdrant collection name")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--query", required=True, help="Single text query (text-only mode)")
    parser.add_argument("--doc-class", default=None, help="Optional payload filter: internal/external")


    args = parser.parse_args()


    rag = MultimodalRAGQuery(
        data_path=args.data_path,
        model_name=args.model,
        index_name=args.index_name,
        device=args.device,
        filter_doc_class=args.doc_class,
    )


    result = ask_rag(args.query, "", k=args.k, rag_instance=rag)
    print(f"\nAnswer: {result[args.query]['final_answer']}")
    print(f"Query time: {result[args.query]['query_time']:.2f}s")
    print(f"Sources: {result[args.query]['metadata']['final_sources']}")



