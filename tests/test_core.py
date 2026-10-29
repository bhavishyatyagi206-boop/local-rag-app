"""
Core test suite for ragcore: extraction, chunking, retrieval, and extractive answering.
All 9 core tests must pass.
"""

import os
import tempfile
import pytest
import docx
import openpyxl

from ragcore import (
    IngestError,
    Chunk,
    extract,
    chunk_text,
    BM25Index,
    KnowledgeBase,
)


def test_1_extract_txt(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Antigravity local RAG system is fast and private.", encoding="utf-8")
    
    pages = extract(txt_file)
    assert len(pages) == 1
    page_num, text = pages[0]
    assert page_num == 1
    assert "Antigravity local RAG system" in text


def test_2_extract_docx(tmp_path):
    docx_file = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_heading("Architecture Overview", level=1)
    doc.add_paragraph("The vector index stores high-dimensional embeddings.")
    doc.save(str(docx_file))

    pages = extract(docx_file)
    assert len(pages) == 1
    page_num, text = pages[0]
    assert page_num == 1
    assert "Architecture Overview" in text
    assert "high-dimensional embeddings" in text


def test_3_extract_xlsx(tmp_path):
    xlsx_file = tmp_path / "data.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Employees"
    ws.append(["Name", "Department", "Role"])
    ws.append(["Alice", "Engineering", "Staff Architect"])
    ws.append(["Bob", "Product", "Technical PM"])
    wb.save(str(xlsx_file))

    pages = extract(xlsx_file)
    assert len(pages) == 1
    page_num, text = pages[0]
    assert page_num == 1
    assert "Staff Architect" in text
    assert "Employees" in text


def test_4_extract_unsupported(tmp_path):
    bad_file = tmp_path / "image.png"
    bad_file.write_bytes(b"\x89PNG\r\n\x1a\n...")
    
    with pytest.raises(IngestError) as exc_info:
        extract(bad_file)
    assert "Unsupported file format" in str(exc_info.value)


def test_5_extract_empty_or_missing(tmp_path):
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("", encoding="utf-8")

    with pytest.raises(IngestError):
        extract(empty_txt)

    missing = tmp_path / "non_existent.txt"
    with pytest.raises(IngestError):
        extract(missing)


def test_6_chunk_text_preserves_page_and_metadata():
    text = (
        "Retrieval-Augmented Generation improves LLM reliability by grounding responses in verified facts. "
        "It minimizes hallucinations by supplying authoritative context alongside the user prompt. "
        "This ensures that domain-specific questions receive exact, verifiable answers."
    )
    chunks = chunk_text(
        text=text,
        doc_id="doc_123",
        doc_name="rag_guide.txt",
        page_number=3,
        start_chunk_idx=0,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == "doc_123"
        assert c.doc_name == "rag_guide.txt"
        assert c.page_number == 3
        assert len(c.text) > 0
        assert c.token_count > 0


def test_7_knowledge_base_add_document():
    kb = KnowledgeBase(name="Test Engineering KB")
    pages = [
        (1, "PostgreSQL supports JSONB for semi-structured document storage."),
        (2, "SQLite is an in-process library that implements a self-contained SQL database engine."),
    ]
    chunks = kb.add_document("doc_db", "Databases.txt", pages)

    assert len(chunks) == 2
    assert "doc_db" in kb.documents
    assert kb.documents["doc_db"]["chunk_count"] == 2
    assert len(kb.chunks) == 2


def test_8_knowledge_base_retrieval_and_filtering():
    kb = KnowledgeBase(name="Multi-Doc KB")
    kb.add_document("doc_python", "python.txt", [(1, "Python is dynamically typed and emphasizes readability.")])
    kb.add_document("doc_rust", "rust.txt", [(1, "Rust guarantees memory safety without needing garbage collection.")])

    # Query Python
    results = kb.retrieve("memory safety garbage collection", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.doc_id == "doc_rust"
    assert score > 0

    # Query with filter doc_ids=['doc_python']
    filtered = kb.retrieve("memory safety", top_k=2, doc_ids=["doc_python"])
    assert len(filtered) == 1
    assert filtered[0][0].doc_id == "doc_python"


def test_9_extractive_answer_and_citations():
    kb = KnowledgeBase(name="QA KB")
    kb.add_document(
        "doc_specs",
        "specs.txt",
        [(1, "The maximum upload payload size allowed in standard mode is 50 Megabytes per request.")]
    )

    res = kb.ask("What is the maximum upload payload size?")
    assert res["status"] == "answered"
    assert "50 Megabytes" in res["answer"]
    assert len(res["citations"]) == 1
    citation = res["citations"][0]
    assert citation["doc_id"] == "doc_specs"
    assert citation["doc_name"] == "specs.txt"
    assert citation["page_number"] == 1
    assert "50 Megabytes" in citation["excerpt"]
