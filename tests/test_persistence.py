"""
Tests for Step 3 Persistence:
- SQLite storage
- Sanitized filenames preventing path traversal
- Document becomes status 'indexed' ONLY after its chunks are committed
- Reloading everything on startup from SQLite and confirming retrieval works
"""

import os
import tempfile
import pytest
from app.database import Database, sanitize_filename
from ragcore import KnowledgeBase, Chunk, extract


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("../../../malicious_file.pdf") == "malicious_file.pdf"
    assert sanitize_filename("safe_doc.txt") == "safe_doc.txt"
    assert sanitize_filename("doc with spaces!@#$.pdf") == "doc_with_spaces____.pdf"
    assert sanitize_filename("....hidden") == "hidden"


def test_sqlite_persistence_and_reload(tmp_path):
    db_file = tmp_path / "test_rag.db"
    db = Database(db_file)

    # 1. Create KB
    kb_record = db.create_kb(kb_id="kb_1", name="Engineering Docs", description="Core engineering handbook")
    assert kb_record["id"] == "kb_1"
    assert kb_record["name"] == "Engineering Docs"

    # 2. Add document record (starts in processing)
    doc_record = db.insert_document_initial(
        doc_id="doc_1",
        kb_id="kb_1",
        name="deployment.txt",
        sha256="abc123sha",
        size=1024,
        file_path="/tmp/deployment.txt",
    )
    assert doc_record["status"] == "processing"

    # 3. Simulate chunk generation
    chunks = [
        Chunk(
            id="doc_1_p1_c0",
            doc_id="doc_1",
            doc_name="deployment.txt",
            page_number=1,
            chunk_index=0,
            text="Antigravity deployments utilize zero-downtime rolling updates via blue-green deployments.",
            token_count=12,
        ),
        Chunk(
            id="doc_1_p2_c1",
            doc_id="doc_1",
            doc_name="deployment.txt",
            page_number=2,
            chunk_index=1,
            text="Database schema migrations run strictly backward-compatible changes first.",
            token_count=10,
        ),
    ]

    # Status must NOT be indexed yet
    doc_pre = db.get_document("doc_1")
    assert doc_pre["status"] == "processing"

    # Commit chunks transactionally
    db.commit_document_chunks(
        doc_id="doc_1",
        kb_id="kb_1",
        chunks=chunks,
        page_count=2,
        processing_ms=45,
    )

    # Status becomes 'indexed' ONLY after chunks are committed
    doc_post = db.get_document("doc_1")
    assert doc_post["status"] == "indexed"
    assert doc_post["chunk_count"] == 2
    assert doc_post["page_count"] == 2

    # 4. Simulate application restart: load KB fresh from database
    loaded_chunks = db.get_chunks_for_kb("kb_1")
    assert len(loaded_chunks) == 2

    reloaded_kb = KnowledgeBase(id="kb_1", name="Engineering Docs")
    reloaded_kb.chunks = loaded_chunks
    reloaded_kb.reindex()

    # Query the reloaded KnowledgeBase
    res = reloaded_kb.ask("How are zero-downtime rolling updates performed?")
    assert res["status"] == "answered"
    assert "blue-green deployments" in res["answer"]
    assert len(res["citations"]) >= 1
    assert res["citations"][0]["doc_id"] == "doc_1"
    assert res["citations"][0]["page_number"] == 1


def test_conversation_and_messages_persistence(tmp_path):
    db_file = tmp_path / "test_rag.db"
    db = Database(db_file)
    db.create_kb("kb_2", "Support KB")

    conv = db.create_conversation("conv_1", "kb_2", "User Inquiry")
    assert conv["id"] == "conv_1"

    msg1 = db.add_message("msg_1", "conv_1", "user", "What is the refund policy?")
    msg2 = db.add_message(
        "msg_2",
        "conv_1",
        "assistant",
        "Refunds are processed within 14 business days.",
        citations=[{"doc_id": "policy_doc", "page_number": 3}],
        debug={"timings_ms": 12},
    )

    history = db.list_messages("conv_1")
    assert len(history) == 2
    assert history[0]["content"] == "What is the refund policy?"
    assert history[1]["citations"][0]["page_number"] == 3
    assert history[1]["debug"]["timings_ms"] == 12

    # Search conversations
    search_results = db.search_conversations("kb_2", "refund")
    assert len(search_results) == 1
    assert search_results[0]["id"] == "conv_1"
