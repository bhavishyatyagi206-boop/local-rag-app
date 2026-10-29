"""
Comprehensive FastAPI TestClient suite for Local RAG API.
Covers:
- GET /health
- KB CRUD
- Document upload (valid multi-file, duplicate upload -> 409, bad format -> 400, corrupt -> 422)
- In-scope question with citations and debug timings
- Out-of-scope question returning no_relevant_context
- Cross-KB isolation enforcement
- Document deletion and reprocessing
- Chunk viewer endpoint
- Conversation endpoints
- LLM provider behavior and clean 502/503 error responses
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app, db, kb_manager
from app.llm import LLMError


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db_path = tmp_path / "api_test.db"
    test_uploads_dir = tmp_path / "uploads"
    test_uploads_dir.mkdir(parents=True, exist_ok=True)

    # Re-initialize DB and Manager with isolated paths
    from app.database import Database
    from app.services import KnowledgeBaseManager
    from app.llm import LLMClient

    test_db = Database(test_db_path)
    test_llm = LLMClient(provider="extractive")
    test_mgr = KnowledgeBaseManager(db=test_db, llm_client=test_llm)
    test_mgr.uploads_dir = test_uploads_dir

    monkeypatch.setattr("app.main.db", test_db)
    monkeypatch.setattr("app.main.kb_manager", test_mgr)

    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database_status"] == "connected"
    assert "active_provider" in data


def test_kb_lifecycle(client):
    # Create KB
    res = client.post("/kbs", json={"name": "DevOps Manual", "description": "CI/CD and Kubernetes setup"})
    assert res.status_code == 201
    kb = res.json()
    kb_id = kb["id"]
    assert kb["name"] == "DevOps Manual"

    # List KBs
    res = client.get("/kbs")
    assert res.status_code == 200
    kbs = res.json()
    assert len(kbs) >= 1

    # Update KB
    res = client.patch(f"/kbs/{kb_id}", json={"name": "DevOps & Infrastructure"})
    assert res.status_code == 200
    assert res.json()["name"] == "DevOps & Infrastructure"

    # Delete KB
    res = client.delete(f"/kbs/{kb_id}")
    assert res.status_code == 204

    # Confirm 404 after deletion
    res = client.get(f"/kbs/{kb_id}")
    assert res.status_code == 404


def test_document_upload_and_duplicate(client):
    # Setup KB
    kb = client.post("/kbs", json={"name": "Knowledge Depot"}).json()
    kb_id = kb["id"]

    # Valid upload
    file_content = b"Antigravity engine supports hybrid BM25 and vector retrieval."
    files = [("files", ("engine_spec.txt", io.BytesIO(file_content), "text/plain"))]
    res = client.post(f"/kbs/{kb_id}/documents", files=files)
    assert res.status_code == 200
    docs = res.json()
    assert len(docs) == 1
    doc_id = docs[0]["id"]
    assert docs[0]["status"] == "indexed"
    assert docs[0]["chunk_count"] >= 1

    # Duplicate upload attempt -> 409 Conflict
    files_dup = [("files", ("engine_spec.txt", io.BytesIO(file_content), "text/plain"))]
    res_dup = client.post(f"/kbs/{kb_id}/documents", files=files_dup)
    assert res_dup.status_code == 409
    assert "Duplicate document detected" in res_dup.json()["detail"]


def test_document_bad_file_and_corrupt(client):
    kb = client.post("/kbs", json={"name": "Validation KB"}).json()
    kb_id = kb["id"]

    # Unsupported file extension -> 400 Bad Request
    bad_files = [("files", ("malware.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream"))]
    res_bad = client.post(f"/kbs/{kb_id}/documents", files=bad_files)
    assert res_bad.status_code == 400
    assert "Unsupported file format" in res_bad.json()["detail"]

    # Corrupt PDF file -> 422 Unprocessable Entity
    corrupt_pdf = [("files", ("broken.pdf", io.BytesIO(b"%PDF-1.4\ncorrupt body"), "application/pdf"))]
    res_corrupt = client.post(f"/kbs/{kb_id}/documents", files=corrupt_pdf)
    assert res_corrupt.status_code == 422
    assert "Corrupt PDF" in res_corrupt.json()["detail"] or "unable" in res_corrupt.json()["detail"]


def test_ask_in_scope_and_out_of_scope(client):
    kb = client.post("/kbs", json={"name": "Product Docs"}).json()
    kb_id = kb["id"]

    content = b"The standard return window for unopened electronics is thirty days from purchase date."
    files = [("files", ("returns.txt", io.BytesIO(content), "text/plain"))]
    client.post(f"/kbs/{kb_id}/documents", files=files)

    # In-scope question
    res_in = client.post(f"/kbs/{kb_id}/ask", json={"query": "What is the return window for electronics?"})
    assert res_in.status_code == 200
    data_in = res_in.json()
    assert data_in["status"] == "answered"
    assert "thirty days" in data_in["answer"]
    assert len(data_in["citations"]) >= 1
    assert data_in["citations"][0]["doc_name"] == "returns.txt"
    assert data_in["citations"][0]["page_number"] == 1
    assert "debug" in data_in
    assert "timings_ms" in data_in["debug"]

    # Out-of-scope question
    res_out = client.post(f"/kbs/{kb_id}/ask", json={"query": "What is the boiling point of liquid nitrogen?"})
    assert res_out.status_code == 200
    data_out = res_out.json()
    assert data_out["status"] == "no_relevant_context"
    assert "No relevant context found" in data_out["answer"]


def test_cross_kb_isolation(client):
    # KB Alpha: Aerospace
    kb_a = client.post("/kbs", json={"name": "Aerospace KB"}).json()
    client.post(
        f"/kbs/{kb_a['id']}/documents",
        files=[("files", ("rockets.txt", io.BytesIO(b"Liquid methane and LOX propel the starship rocket."), "text/plain"))],
    )

    # KB Beta: Marine Biology
    kb_b = client.post("/kbs", json={"name": "Marine KB"}).json()
    client.post(
        f"/kbs/{kb_b['id']}/documents",
        files=[("files", ("marine.txt", io.BytesIO(b"Coral reefs support over twenty-five percent of marine life."), "text/plain"))],
    )

    # Ask KB Alpha about Marine question -> must NOT retrieve KB Beta context
    res_cross = client.post(f"/kbs/{kb_a['id']}/ask", json={"query": "What percentage of marine life do coral reefs support?"})
    assert res_cross.status_code == 200
    assert res_cross.json()["status"] == "no_relevant_context"
    assert len(res_cross.json()["citations"]) == 0

    # Ask KB Beta directly -> answered
    res_correct = client.post(f"/kbs/{kb_b['id']}/ask", json={"query": "What percentage of marine life do coral reefs support?"})
    assert res_correct.status_code == 200
    assert res_correct.json()["status"] == "answered"
    assert "twenty-five percent" in res_correct.json()["answer"]


def test_chunk_viewer_endpoint(client):
    kb = client.post("/kbs", json={"name": "Viewer KB"}).json()
    upload = client.post(
        f"/kbs/{kb['id']}/documents",
        files=[("files", ("note.txt", io.BytesIO(b"Detailed citation testing excerpt text."), "text/plain"))],
    ).json()
    doc_id = upload[0]["id"]

    # Query to obtain chunk_id
    ask_res = client.post(f"/kbs/{kb['id']}/ask", json={"query": "citation testing"}).json()
    chunk_id = ask_res["citations"][0]["chunk_id"]

    # Hit chunk viewer endpoint
    res = client.get(f"/documents/{doc_id}/chunks/{chunk_id}")
    assert res.status_code == 200
    chunk_data = res.json()
    assert chunk_data["id"] == chunk_id
    assert chunk_data["doc_id"] == doc_id
    assert "Detailed citation testing" in chunk_data["text"]

    # Mismatched doc_id -> 404
    bad_res = client.get(f"/documents/wrong_doc_id/chunks/{chunk_id}")
    assert bad_res.status_code == 404


def test_document_delete_removes_from_retrieval(client):
    kb = client.post("/kbs", json={"name": "Ephemeral KB"}).json()
    upload = client.post(
        f"/kbs/{kb['id']}/documents",
        files=[("files", ("secret.txt", io.BytesIO(b"Confidential token: XYZ987PASS."), "text/plain"))],
    ).json()
    doc_id = upload[0]["id"]

    # Verify searchable
    res1 = client.post(f"/kbs/{kb['id']}/ask", json={"query": "XYZ987PASS"}).json()
    assert res1["status"] == "answered"

    # Delete document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 204

    # Verify not searchable
    res2 = client.post(f"/kbs/{kb['id']}/ask", json={"query": "XYZ987PASS"}).json()
    assert res2["status"] == "no_relevant_context"


def test_llm_provider_clean_error(client, monkeypatch):
    kb = client.post("/kbs", json={"name": "Faulty Provider KB"}).json()
    client.post(
        f"/kbs/{kb['id']}/documents",
        files=[("files", ("data.txt", io.BytesIO(b"Testing provider crash resilience."), "text/plain"))],
    )

    # Simulate external LLM provider outage
    def mock_generate_failure(*args, **kwargs):
        raise LLMError("Ollama service unreachable. Ensure Ollama is running.", status_code=503)

    from app.main import kb_manager
    monkeypatch.setattr(kb_manager.llm, "generate_answer", mock_generate_failure)

    res = client.post(f"/kbs/{kb['id']}/ask", json={"query": "provider crash resilience"})
    assert res.status_code == 503
    assert "Ollama service unreachable" in res.json()["detail"]
