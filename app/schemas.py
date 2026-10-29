"""
Pydantic schemas for request and response validation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    active_provider: str
    active_model: str
    database_status: str
    total_kbs: int
    total_documents: int


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(default="", max_length=500)


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    created_at: str
    document_count: Optional[int] = 0
    total_chunks: Optional[int] = 0


class DocumentResponse(BaseModel):
    id: str
    kb_id: str
    name: str
    sha256: str
    size: int
    page_count: int
    status: str
    error: Optional[str] = None
    chunk_count: int
    created_at: str
    processing_ms: int


class DocumentRename(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class CitationItem(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    score: float
    excerpt: str


class DebugInfo(BaseModel):
    query: str
    retrieved_chunks: List[str]
    scores: List[float]
    timings_ms: Optional[Dict[str, float]] = None


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    doc_ids: Optional[List[str]] = None


class AskResponse(BaseModel):
    answer: str
    status: str
    provider: str
    model: str
    citations: List[CitationItem]
    debug: DebugInfo


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default="New Conversation", max_length=200)


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ConversationResponse(BaseModel):
    id: str
    kb_id: str
    title: str
    created_at: str


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    doc_ids: Optional[List[str]] = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    citations: Optional[List[CitationItem]] = None
    debug: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class ChunkResponse(BaseModel):
    id: str
    doc_id: str
    kb_id: str
    doc_name: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int
    embedding_model: Optional[str] = None
