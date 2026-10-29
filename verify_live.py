import httpx
from pathlib import Path

base_url = "http://127.0.0.1:8000"
sample_dir = Path("/Users/bhavishyatyagi/local-rag-app/sample_docs")

with httpx.Client(base_url=base_url, timeout=15.0) as client:
    # 1. Health
    h = client.get("/health").json()
    print("1. Health Status:", h)

    # 2. Create KB
    kb = client.post("/kbs", json={"name": "Production Verification KB", "description": "End-to-end live testing"}).json()
    kb_id = kb["id"]
    kb_name = kb["name"]
    print(f"\n2. Created KB: {kb_name} (ID: {kb_id})")

    # 3. Upload sample documents
    with open(sample_dir / "sample.txt", "rb") as f1, \
         open(sample_dir / "sample.docx", "rb") as f2, \
         open(sample_dir / "sample.pdf", "rb") as f3:
        files = [
            ("files", ("sample.txt", f1, "text/plain")),
            ("files", ("sample.docx", f2, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("sample.pdf", f3, "application/pdf")),
        ]
        docs = client.post(f"/kbs/{kb_id}/documents", files=files).json()

    print(f"\n3. Uploaded {len(docs)} documents:")
    for d in docs:
        print(f"   - {d['name']}: status={d['status']}, pages={d['page_count']}, chunks={d['chunk_count']}")

    # 4. In-scope query 1 (sample.txt)
    q1 = "When does scheduled cluster maintenance occur?"
    res1 = client.post(f"/kbs/{kb_id}/ask", json={"query": q1}).json()
    print(f"\n4. Query: \"{q1}\"")
    print(f"   Status: {res1['status']}")
    print(f"   Answer: {res1['answer']}")
    print(f"   Citations: {res1['citations']}")

    # 5. In-scope query 2 (sample.docx)
    q2 = "What is the automatic failover SLA?"
    res2 = client.post(f"/kbs/{kb_id}/ask", json={"query": q2}).json()
    print(f"\n5. Query: \"{q2}\"")
    print(f"   Status: {res2['status']}")
    print(f"   Answer: {res2['answer']}")
    print(f"   Citations: {res2['citations']}")

    # 6. In-scope query 3 (sample.pdf, page 2)
    q3 = "What is the oxygen production rate of MOXIE?"
    res3 = client.post(f"/kbs/{kb_id}/ask", json={"query": q3}).json()
    print(f"\n6. Query: \"{q3}\"")
    print(f"   Status: {res3['status']}")
    print(f"   Answer: {res3['answer']}")
    print(f"   Citations: {res3['citations']}")

    # 7. Out-of-scope query
    q4 = "How do you make chocolate soufflé?"
    res4 = client.post(f"/kbs/{kb_id}/ask", json={"query": q4}).json()
    print(f"\n7. Out-of-scope Query: \"{q4}\"")
    print(f"   Status: {res4['status']}")
    print(f"   Answer: {res4['answer']}")
    print(f"   Citations: {res4['citations']}")

    # 8. Multi-document query
    q5 = "Tell me about cluster maintenance and Mars 2020 mission."
    res5 = client.post(f"/kbs/{kb_id}/ask", json={"query": q5}).json()
    print(f"\n8. Multi-doc Query: \"{q5}\"")
    print(f"   Status: {res5['status']}")
    print(f"   Answer: {res5['answer']}")
    print(f"   Retrieved Citations Count: {len(res5['citations'])}")
    for cit in res5['citations']:
        print(f"     * Doc: {cit['doc_name']}, Page: {cit['page_number']}, Score: {cit['score']}")
