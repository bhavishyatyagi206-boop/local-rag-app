"""
Tests for Step 8:
- EmbeddingProvider interface
- SQLite vector storage
- BM25 and RRF (Reciprocal Rank Fusion)
- Tracking embedding model name per chunk
- Triggering re-indexing when model changes
- Ensuring embeddings from different models are never mixed
"""

import pytest
from ragcore import (
    Chunk,
    DeterministicLocalEmbeddingProvider,
    KnowledgeBase,
    cosine_similarity,
)


def test_embedding_provider_and_similarity():
    provider = DeterministicLocalEmbeddingProvider("model-alpha", dim=32)
    vecs = provider.embed_texts(["quantum computing qubit superposition", "classical binary transistor"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 32

    # Query similarity
    q_vec = provider.embed_query("quantum superposition")
    sim1 = cosine_similarity(q_vec, vecs[0])
    sim2 = cosine_similarity(q_vec, vecs[1])
    assert sim1 > sim2


def test_hybrid_rrf_and_model_change_reindexing():
    kb = KnowledgeBase(name="Hybrid KB")
    kb.add_document("doc1", "quantum.txt", [(1, "Superconducting qubits maintain coherence in dilution refrigerators.")])
    kb.add_document("doc2", "relativity.txt", [(1, "General relativity describes spacetime curvature around massive stars.")])

    # 1. Attach Provider A
    provider_a = DeterministicLocalEmbeddingProvider("provider-v1", dim=32)
    kb.set_embedding_provider(provider_a)

    for c in kb.chunks:
        assert c.embedding_model == "provider-v1"
        assert len(c.embedding) == 32

    # Query with Hybrid RRF
    res = kb.retrieve("dilution refrigerators qubits", top_k=2)
    assert len(res) >= 1
    top_chunk, score = res[0]
    assert top_chunk.doc_id == "doc1"
    assert score > 0.0

    # 2. Change Provider to Provider B with different dimension
    provider_b = DeterministicLocalEmbeddingProvider("provider-v2-upgraded", dim=64)
    kb.set_embedding_provider(provider_b)

    # Chunks must all be re-indexed to provider-v2-upgraded (never mixed!)
    for c in kb.chunks:
        assert c.embedding_model == "provider-v2-upgraded"
        assert len(c.embedding) == 64

    # Query again with new model
    res_b = kb.retrieve("dilution refrigerators qubits", top_k=2)
    assert len(res_b) >= 1
    assert res_b[0][0].doc_id == "doc1"
