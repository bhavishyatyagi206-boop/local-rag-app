"""
Live Application Verification:
Simulates real HTTP client against the live FastAPI app:
1. Health check
2. Create KB
3. Upload TXT, DOCX, and multi-page PDF from sample_docs
4. In-scope queries with exact page-level citations
5. Out-of-scope queries
6. Multi-document query
"""

from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

sample_dir = Path("/Users/bhavishyatyagi/local-rag-app/sample_docs")

client = TestClient(app)

print("=== 1. Health Status ===")
h = client.get("/health").json()
print("Health:", h)

print("\n=== 2. Creating Knowledge Base ===")
kb = client.post("/kbs", json={"name": "Production Verification KB", "description": "End-to-end live testing"}).json()
kb_id = kb["id"]
print(f"Created KB: {kb['name']} (ID: {kb_id})")

print("\n=== 3. Uploading Sample Documents ===")
with open(sample_dir / "sample.txt", "rb") as f1, \
     open(sample_dir / "sample.docx", "rb") as f2, \
     open(sample_dir / "sample.pdf", "rb") as f3:
    files = [
        ("files", ("sample.txt", f1, "text/plain")),
        ("files", ("sample.docx", f2, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ("files", ("sample.pdf", f3, "application/pdf")),
    ]
    docs = client.post(f"/kbs/{kb_id}/documents", files=files).json()

print(f"Successfully uploaded and indexed {len(docs)} documents:")
for d in docs:
    print(f"  * {d['name']}: status={d['status']}, pages={d['page_count']}, chunks={d['chunk_count']}, size={d['size']} bytes")

print("\n=== 4. In-Scope Query 1: TXT (Maintenance Windows) ===")
q1 = "When does scheduled cluster maintenance occur?"
res1 = client.post(f"/kbs/{kb_id}/ask", json={"query": q1}).json()
print(f"Query: \"{q1}\"")
print(f"Status: {res1['status']}")
print(f"Answer: {res1['answer']}")
print(f"Citations ({len(res1['citations'])}):")
for c in res1['citations']:
    print(f"  - Doc: {c['doc_name']}, Page: {c['page_number']}, Score: {c['score']}")
    print(f"    Excerpt: {c['excerpt']}")

print("\n=== 5. In-Scope Query 2: DOCX (Database SLA) ===")
q2 = "What is the automatic failover SLA?"
res2 = client.post(f"/kbs/{kb_id}/ask", json={"query": q2}).json()
print(f"Query: \"{q2}\"")
print(f"Status: {res2['status']}")
print(f"Answer: {res2['answer']}")
print(f"Citations ({len(res2['citations'])}):")
for c in res2['citations']:
    print(f"  - Doc: {c['doc_name']}, Page: {c['page_number']}, Score: {c['score']}")
    print(f"    Excerpt: {c['excerpt']}")

print("\n=== 6. In-Scope Query 3: PDF Page 2 (Mars MOXIE Payload) ===")
q3 = "What is the oxygen production rate of MOXIE?"
res3 = client.post(f"/kbs/{kb_id}/ask", json={"query": q3}).json()
print(f"Query: \"{q3}\"")
print(f"Status: {res3['status']}")
print(f"Answer: {res3['answer']}")
print(f"Citations ({len(res3['citations'])}):")
for c in res3['citations']:
    print(f"  - Doc: {c['doc_name']}, Page: {c['page_number']}, Score: {c['score']}")
    print(f"    Excerpt: {c['excerpt']}")

print("\n=== 7. Out-of-Scope Query ===")
q4 = "How do you make chocolate soufflé?"
res4 = client.post(f"/kbs/{kb_id}/ask", json={"query": q4}).json()
print(f"Query: \"{q4}\"")
print(f"Status: {res4['status']}")
print(f"Answer: {res4['answer']}")
print(f"Citations Count: {len(res4['citations'])}")

print("\n=== 8. Multi-Document Query ===")
q5 = "Tell me about cluster maintenance and Mars 2020 mission."
res5 = client.post(f"/kbs/{kb_id}/ask", json={"query": q5}).json()
print(f"Query: \"{q5}\"")
print(f"Status: {res5['status']}")
print(f"Answer: {res5['answer']}")
print(f"Citations Count: {len(res5['citations'])}")
for c in res5['citations']:
    print(f"  - Doc: {c['doc_name']}, Page: {c['page_number']}, Score: {c['score']}")
    print(f"    Excerpt: {c['excerpt']}")

print("\n=== 9. Citation Viewer Verification ===")
if res3['citations']:
    top_c = res3['citations'][0]
    chunk_res = client.get(f"/documents/{top_c['doc_id']}/chunks/{top_c['chunk_id']}").json()
    print(f"Fetched chunk from DB: doc={chunk_res['doc_name']}, page={chunk_res['page_number']}, text={chunk_res['text'][:80]}...")
