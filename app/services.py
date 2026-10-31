"""
Service layer coordinating SQLite persistence, in-memory Knowledge Bases,
file ingestion, extraction, and RAG execution.
"""

from __future__ import annotations
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.database import Database, sanitize_filename
from app.llm import LLMClient, LLMError
from ragcore import Chunk, IngestError, KnowledgeBase, chunk_text, extract


SUMMARY_TRIGGERS = (
    "summary", "summarize", "overview", "what is this document",
    "what is the document", "explain this document", "about this document",
    "tell me about this document", "what does this document", "what is in this document",
    "what is this file", "what is the pdf", "what is this pdf", "main points",
    "brief this", "what is it about"
)


class IngestionValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class KnowledgeBaseManager:
    def __init__(self, db: Database, llm_client: Optional[LLMClient] = None):
        self.db = db
        self.llm = llm_client or LLMClient()
        self.kbs: Dict[str, KnowledgeBase] = {}
        self.uploads_dir = Path(settings.UPLOADS_DIR)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.reload_all_from_db()

    def reload_all_from_db(self) -> None:
        """Reload all knowledge bases and chunks from SQLite on startup."""
        kb_records = self.db.list_kbs()
        self.kbs.clear()
        for rec in kb_records:
            kb_id = rec["id"]
            kb = KnowledgeBase(id=kb_id, name=rec["name"])
            chunks = self.db.get_chunks_for_kb(kb_id)
            kb.chunks = chunks
            kb.reindex()
            self.kbs[kb_id] = kb

    def get_kb_instance(self, kb_id: str) -> Optional[KnowledgeBase]:
        if kb_id in self.kbs:
            return self.kbs[kb_id]
        rec = self.db.get_kb(kb_id)
        if not rec:
            return None
        kb = KnowledgeBase(id=kb_id, name=rec["name"])
        kb.chunks = self.db.get_chunks_for_kb(kb_id)
        kb.reindex()
        self.kbs[kb_id] = kb
        return kb

    def create_kb(self, name: str, description: str = "") -> Dict[str, Any]:
        kb_id = str(uuid.uuid4())
        rec = self.db.create_kb(kb_id=kb_id, name=name, description=description)
        self.kbs[kb_id] = KnowledgeBase(id=kb_id, name=name)
        return rec

    def delete_kb(self, kb_id: str) -> bool:
        if kb_id in self.kbs:
            del self.kbs[kb_id]
        return self.db.delete_kb(kb_id)

    def ingest_file(
        self,
        kb_id: str,
        filename: str,
        content: bytes,
    ) -> Dict[str, Any]:
        """
        Ingest an uploaded file into a Knowledge Base with size, extension,
        and content validation, preventing path traversal and duplicate uploads.
        """
        # 1. Verify KB exists
        kb_rec = self.db.get_kb(kb_id)
        if not kb_rec:
            raise IngestionValidationError(f"Knowledge Base '{kb_id}' not found.", status_code=404)

        # 2. Size validation
        if len(content) > settings.max_upload_bytes:
            raise IngestionValidationError(
                f"File size exceeds maximum permitted limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
                status_code=413,
            )
        if len(content) == 0:
            raise IngestionValidationError("Uploaded file is empty (0 bytes).", status_code=400)

        # 3. Extension validation
        clean_name = sanitize_filename(filename)
        ext = Path(clean_name).suffix.lower()
        allowed_exts = {".txt", ".docx", ".xlsx", ".pdf"}
        if ext not in allowed_exts:
            raise IngestionValidationError(
                f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(allowed_exts))}",
                status_code=400,
            )

        # 4. Duplicate upload check via SHA-256
        sha256 = hashlib.sha256(content).hexdigest()
        existing = self.db.find_document_by_hash(kb_id=kb_id, sha256=sha256)
        if existing:
            raise IngestionValidationError(
                f"Duplicate document detected: '{existing['name']}' already exists in this Knowledge Base.",
                status_code=409,
            )

        # 5. Save upload with sanitized unique filename
        doc_id = str(uuid.uuid4())
        safe_disk_filename = f"{doc_id}_{clean_name}"
        saved_path = self.uploads_dir / safe_disk_filename
        saved_path.write_bytes(content)

        # 6. Insert initial document in 'processing' status
        self.db.insert_document_initial(
            doc_id=doc_id,
            kb_id=kb_id,
            name=clean_name,
            sha256=sha256,
            size=len(content),
            file_path=str(saved_path),
        )

        start_time = time.time()
        try:
            # 7. Extract pages
            pages = extract(saved_path, filename=clean_name)

            # 8. Chunk text
            chunks: List[Chunk] = []
            chunk_global_idx = 0
            for page_num, text in pages:
                p_chunks = chunk_text(
                    text=text,
                    doc_id=doc_id,
                    doc_name=clean_name,
                    page_number=page_num,
                    start_chunk_idx=chunk_global_idx,
                )
                chunks.extend(p_chunks)
                chunk_global_idx += len(p_chunks)

            processing_ms = int((time.time() - start_time) * 1000)

            # 9. Transactional commit: chunks inserted and document marked 'indexed'
            self.db.commit_document_chunks(
                doc_id=doc_id,
                kb_id=kb_id,
                chunks=chunks,
                page_count=len(pages),
                processing_ms=processing_ms,
            )

            # 10. Update in-memory KB
            kb = self.get_kb_instance(kb_id)
            if kb:
                # Remove any existing chunks for doc_id
                kb.chunks = [c for c in kb.chunks if c.doc_id != doc_id]
                kb.chunks.extend(chunks)
                kb.reindex()

            return self.db.get_document(doc_id)  # type: ignore

        except Exception as e:
            proc_ms = int((time.time() - start_time) * 1000)
            self.db.mark_document_failed(doc_id, error_msg=str(e), processing_ms=proc_ms)
            if isinstance(e, IngestError):
                raise IngestionValidationError(str(e), status_code=422)
            raise IngestionValidationError(f"Failed to process document: {e}", status_code=500)

    def reprocess_document(self, doc_id: str) -> Dict[str, Any]:
        """Reprocess an existing document from its original upload file."""
        doc = self.db.get_document(doc_id)
        if not doc:
            raise IngestionValidationError(f"Document '{doc_id}' not found.", status_code=404)

        file_path = doc.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise IngestionValidationError("Original uploaded file is missing from disk.", status_code=404)

        start_time = time.time()
        try:
            pages = extract(file_path, filename=doc["name"])
            chunks: List[Chunk] = []
            chunk_global_idx = 0
            for page_num, text in pages:
                p_chunks = chunk_text(
                    text=text,
                    doc_id=doc_id,
                    doc_name=doc["name"],
                    page_number=page_num,
                    start_chunk_idx=chunk_global_idx,
                )
                chunks.extend(p_chunks)
                chunk_global_idx += len(p_chunks)

            processing_ms = int((time.time() - start_time) * 1000)
            self.db.commit_document_chunks(
                doc_id=doc_id,
                kb_id=doc["kb_id"],
                chunks=chunks,
                page_count=len(pages),
                processing_ms=processing_ms,
            )

            kb = self.get_kb_instance(doc["kb_id"])
            if kb:
                kb.chunks = [c for c in kb.chunks if c.doc_id != doc_id]
                kb.chunks.extend(chunks)
                kb.reindex()

            return self.db.get_document(doc_id)  # type: ignore
        except Exception as e:
            proc_ms = int((time.time() - start_time) * 1000)
            self.db.mark_document_failed(doc_id, error_msg=str(e), processing_ms=proc_ms)
            raise IngestionValidationError(f"Reprocessing failed: {e}", status_code=422)

    def ask(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query a specific Knowledge Base with KB boundary isolation.
        Returns answer, citations, and debug metrics.
        """
        start_t = time.time()
        kb = self.get_kb_instance(kb_id)
        if not kb:
            raise IngestionValidationError(f"Knowledge Base '{kb_id}' not found.", status_code=404)

        # Enforce doc_ids belong to kb_id if supplied
        if doc_ids:
            kb_docs = {d["id"] for d in self.db.list_documents(kb_id)}
            invalid_docs = [did for did in doc_ids if did not in kb_docs]
            if invalid_docs:
                raise IngestionValidationError(
                    f"Document IDs not in this knowledge base: {invalid_docs}",
                    status_code=400,
                )

        retrieval_start = time.time()
        retrieved_chunks = kb.retrieve(query=query, top_k=top_k, doc_ids=doc_ids)

        # Handle summary/overview requests gracefully
        q_clean = query.lower().strip()
        if any(trig in q_clean for trig in SUMMARY_TRIGGERS) and (not retrieved_chunks or all(s <= 0.0 for _, s in retrieved_chunks)):
            allowed_docs = set(doc_ids) if doc_ids is not None else None
            summary_chunks = []
            for c in kb.chunks:
                if allowed_docs is not None and c.doc_id not in allowed_docs:
                    continue
                summary_chunks.append((c, 1.0))
                if len(summary_chunks) >= top_k:
                    break
            if summary_chunks:
                retrieved_chunks = summary_chunks

        retrieval_ms = (time.time() - retrieval_start) * 1000

        llm_start = time.time()
        answer, provider_name, model_name = self.llm.generate_answer(query, retrieved_chunks)
        llm_ms = (time.time() - llm_start) * 1000

        total_ms = (time.time() - start_t) * 1000

        # Build citations from retrieved chunks
        citations = []
        seen = set()
        for chunk, score in retrieved_chunks:
            if score <= 0.0:
                continue
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            excerpt = chunk.text[:280].strip() + ("..." if len(chunk.text) > 280 else "")
            citations.append({
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "score": round(score, 4),
                "excerpt": excerpt,
            })

        has_relevant = any(score > 0.0 for _, score in retrieved_chunks)
        status_val = "answered" if (has_relevant or self.llm.provider != "extractive") else "no_relevant_context"

        return {
            "answer": answer,
            "status": status_val,
            "provider": provider_name,
            "model": model_name,
            "citations": citations,
            "debug": {
                "query": query,
                "retrieved_chunks": [c.id for c, _ in retrieved_chunks],
                "scores": [round(s, 4) for _, s in retrieved_chunks],
                "timings_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "llm_generation": round(llm_ms, 2),
                    "total": round(total_ms, 2),
                },
            },
        }

    def get_chunk_viewer(self, doc_id: str, chunk_id: str) -> Dict[str, Any]:
        """Fetch chunk details ensuring doc_id boundary."""
        chunk = self.db.get_chunk(chunk_id)
        if not chunk or chunk["doc_id"] != doc_id:
            raise IngestionValidationError(
                f"Chunk '{chunk_id}' not found for document '{doc_id}'.",
                status_code=404,
            )
        return chunk
