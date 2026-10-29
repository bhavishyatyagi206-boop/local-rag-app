"""
FastAPI application for Local RAG Web App.
Includes all REST endpoints, CORS, static file serving, and error handlers.
"""

from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import Database
from app.llm import LLMClient, LLMError
from app.schemas import (
    AskRequest,
    AskResponse,
    ChunkResponse,
    ConversationCreate,
    ConversationResponse,
    ConversationUpdate,
    DocumentRename,
    DocumentResponse,
    HealthResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    MessageCreate,
    MessageResponse,
)
from app.services import IngestionValidationError, KnowledgeBaseManager

app = FastAPI(
    title="Local RAG Web App",
    description="Full-featured local RAG engine with multi-format ingestion, lexical/hybrid search, and grounded citations.",
    version="1.0.0",
)

# CORS configuration
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services initialization
db = Database(settings.DATABASE_PATH)
llm_client = LLMClient()
kb_manager = KnowledgeBaseManager(db=db, llm_client=llm_client)

# Mount static folder
static_path = Path(__file__).parent / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.exception_handler(IngestionValidationError)
async def ingestion_error_handler(request, exc: IngestionValidationError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(LLMError)
async def llm_error_handler(request, exc: LLMError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


# ------------------ UI Home Route ------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "Local RAG API is running. UI not found at static/index.html."})


# ------------------ Health Endpoint ------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    kbs = db.list_kbs()
    total_docs = 0
    for kb in kbs:
        total_docs += len(db.list_documents(kb["id"]))

    return HealthResponse(
        status="healthy",
        active_provider=settings.LLM_PROVIDER,
        active_model=settings.LLM_MODEL if settings.LLM_PROVIDER != "extractive" else "extractive",
        database_status="connected",
        total_kbs=len(kbs),
        total_documents=total_docs,
    )


# ------------------ Knowledge Bases ------------------

@app.post("/kbs", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(payload: KnowledgeBaseCreate):
    rec = kb_manager.create_kb(name=payload.name, description=payload.description or "")
    rec["document_count"] = 0
    rec["total_chunks"] = 0
    return rec


@app.get("/kbs", response_model=List[KnowledgeBaseResponse])
def list_knowledge_bases():
    kbs = db.list_kbs()
    results = []
    for kb in kbs:
        docs = db.list_documents(kb["id"])
        total_chunks = sum(d["chunk_count"] for d in docs)
        results.append({
            **kb,
            "document_count": len(docs),
            "total_chunks": total_chunks,
        })
    return results


@app.get("/kbs/{kb_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(kb_id: str):
    kb = db.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    docs = db.list_documents(kb_id)
    total_chunks = sum(d["chunk_count"] for d in docs)
    return {
        **kb,
        "document_count": len(docs),
        "total_chunks": total_chunks,
    }


@app.patch("/kbs/{kb_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(kb_id: str, payload: KnowledgeBaseUpdate):
    kb = db.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    updated = db.update_kb(kb_id, name=payload.name, description=payload.description)
    docs = db.list_documents(kb_id)
    return {
        **updated,
        "document_count": len(docs),
        "total_chunks": sum(d["chunk_count"] for d in docs),
    }


@app.delete("/kbs/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge_base(kb_id: str):
    success = kb_manager.delete_kb(kb_id)
    if not success:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return None


# ------------------ Documents ------------------

@app.post("/kbs/{kb_id}/documents", response_model=List[DocumentResponse])
async def upload_documents(
    kb_id: str,
    files: List[UploadFile] = File(...),
):
    kb = db.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload")

    created_docs = []
    for file in files:
        content = await file.read()
        doc = kb_manager.ingest_file(
            kb_id=kb_id,
            filename=file.filename or "upload.txt",
            content=content,
        )
        created_docs.append(doc)
    return created_docs


@app.get("/kbs/{kb_id}/documents", response_model=List[DocumentResponse])
def list_documents(kb_id: str):
    kb = db.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return db.list_documents(kb_id)


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.patch("/documents/{doc_id}", response_model=DocumentResponse)
def rename_document(doc_id: str, payload: DocumentRename):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    renamed = db.rename_document(doc_id, payload.name)
    kb = kb_manager.get_kb_instance(doc["kb_id"])
    if kb:
        for c in kb.chunks:
            if c.doc_id == doc_id:
                c.doc_name = payload.name
        kb.reindex()
    return renamed


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    kb_id = doc["kb_id"]
    db.delete_document(doc_id)
    kb = kb_manager.get_kb_instance(kb_id)
    if kb:
        kb.remove_document(doc_id)
    return None


@app.post("/documents/{doc_id}/reprocess", response_model=DocumentResponse)
def reprocess_document(doc_id: str):
    return kb_manager.reprocess_document(doc_id)


# ------------------ Citation Viewer ------------------

@app.get("/documents/{doc_id}/chunks/{chunk_id}", response_model=ChunkResponse)
def get_chunk_viewer(doc_id: str, chunk_id: str):
    return kb_manager.get_chunk_viewer(doc_id=doc_id, chunk_id=chunk_id)


# ------------------ Ask Query ------------------

@app.post("/kbs/{kb_id}/ask", response_model=AskResponse)
def ask_question(kb_id: str, payload: AskRequest):
    return kb_manager.ask(
        kb_id=kb_id,
        query=payload.query,
        top_k=payload.top_k or 5,
        doc_ids=payload.doc_ids,
    )


# ------------------ Conversations & Messages ------------------

@app.post("/kbs/{kb_id}/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(kb_id: str, payload: ConversationCreate):
    kb = db.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    conv_id = str(uuid.uuid4())
    return db.create_conversation(conv_id=conv_id, kb_id=kb_id, title=payload.title or "New Conversation")


@app.get("/kbs/{kb_id}/conversations", response_model=List[ConversationResponse])
def list_conversations(kb_id: str):
    kb = db.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return db.list_conversations(kb_id)


@app.patch("/conversations/{conv_id}", response_model=ConversationResponse)
def rename_conversation(conv_id: str, payload: ConversationUpdate):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db.rename_conversation(conv_id, payload.title)


@app.delete("/conversations/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete_conversation(conv_id)
    return None


@app.get("/conversations/{conv_id}/messages", response_model=List[MessageResponse])
def get_messages(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db.list_messages(conv_id)


@app.post("/conversations/{conv_id}/messages", response_model=MessageResponse)
def send_message(conv_id: str, payload: MessageCreate):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    kb_id = conv["kb_id"]

    # 1. Store user message
    user_msg_id = str(uuid.uuid4())
    db.add_message(
        msg_id=user_msg_id,
        conv_id=conv_id,
        role="user",
        content=payload.content,
    )

    # 2. Run RAG query
    rag_result = kb_manager.ask(
        kb_id=kb_id,
        query=payload.content,
        top_k=payload.top_k or 5,
        doc_ids=payload.doc_ids,
    )

    # 3. Store assistant message
    asst_msg_id = str(uuid.uuid4())
    asst_msg = db.add_message(
        msg_id=asst_msg_id,
        conv_id=conv_id,
        role="assistant",
        content=rag_result["answer"],
        citations=rag_result["citations"],
        debug=rag_result["debug"],
    )

    return asst_msg


@app.get("/conversations/search", response_model=List[ConversationResponse])
def search_conversations(
    kb_id: str = Query(..., description="Knowledge base ID"),
    q: str = Query(..., min_length=1, description="Search query"),
):
    kb = db.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return db.search_conversations(kb_id=kb_id, query=q)
