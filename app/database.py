"""
Database and persistence layer using stdlib sqlite3.
Handles schema initialization, sanitized upload storage, transactions, and state reloading.
"""

from __future__ import annotations
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragcore import Chunk, IngestError, KnowledgeBase


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and unsafe character exploitation.
    """
    # Extract only the base name (no slashes, no directory structure)
    base = os.path.basename(filename)
    # Remove null bytes
    base = base.replace("\0", "")
    # Normalize unicode / strip path traversal sequences
    base = re.sub(r"\.\.+[/\\Release]*", "", base)
    # Allow alphanumeric, dot, underscore, hyphen
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    # Trim leading dots to prevent hidden files
    cleaned = cleaned.lstrip(".")
    if not cleaned:
        cleaned = f"upload_{int(time.time())}.dat"
    return cleaned


class Database:
    def __init__(self, db_path: Union[str, Path] = "data/rag.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_ms INTEGER NOT NULL DEFAULT 0,
                    file_path TEXT,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    kb_id TEXT NOT NULL,
                    doc_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    embedding_model TEXT,
                    embedding_json TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    debug_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(kb_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_conv_kb ON conversations(kb_id);
                CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            """)

    # ------------------ Knowledge Bases ------------------

    def create_kb(self, kb_id: str, name: str, description: str = "") -> Dict[str, Any]:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (id, name, description) VALUES (?, ?, ?)",
                (kb_id, name, description),
            )
        return self.get_kb(kb_id)  # type: ignore

    def list_kbs(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_kb(self, kb_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
            return dict(row) if row else None

    def update_kb(self, kb_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            updates = []
            params = []
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if not updates:
                return self.get_kb(kb_id)
            params.append(kb_id)
            conn.execute(f"UPDATE knowledge_bases SET {', '.join(updates)} WHERE id = ?", params)
        return self.get_kb(kb_id)

    def delete_kb(self, kb_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
            return cur.rowcount > 0

    # ------------------ Documents ------------------

    def insert_document_initial(
        self,
        doc_id: str,
        kb_id: str,
        name: str,
        sha256: str,
        size: int,
        file_path: str,
    ) -> Dict[str, Any]:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, kb_id, name, sha256, size, status, file_path)
                VALUES (?, ?, ?, ?, ?, 'processing', ?)
                """,
                (doc_id, kb_id, name, sha256, size, file_path),
            )
        return self.get_document(doc_id)  # type: ignore

    def commit_document_chunks(
        self,
        doc_id: str,
        kb_id: str,
        chunks: List[Chunk],
        page_count: int,
        processing_ms: int,
    ) -> None:
        """
        Transactional commit: inserts all chunks and transitions document status to 'indexed'.
        Document only becomes 'indexed' after chunks are safely committed.
        """
        with self._get_connection() as conn:
            # Delete any existing chunks for this doc
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            
            chunk_rows = [
                (
                    c.id,
                    c.doc_id,
                    kb_id,
                    c.doc_name,
                    c.page_number,
                    c.chunk_index,
                    c.text,
                    c.token_count,
                    c.embedding_model,
                    json.dumps(c.embedding) if c.embedding is not None else None,
                )
                for c in chunks
            ]
            conn.executemany(
                """
                INSERT INTO chunks (
                    id, doc_id, kb_id, doc_name, page_number, chunk_index,
                    text, token_count, embedding_model, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_rows,
            )

            conn.execute(
                """
                UPDATE documents
                SET status = 'indexed',
                    chunk_count = ?,
                    page_count = ?,
                    processing_ms = ?,
                    error = NULL
                WHERE id = ?
                """,
                (len(chunks), page_count, processing_ms, doc_id),
            )

    def mark_document_failed(self, doc_id: str, error_msg: str, processing_ms: int = 0) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = 'failed',
                    error = ?,
                    processing_ms = ?
                WHERE id = ?
                """,
                (error_msg, processing_ms, doc_id),
            )

    def list_documents(self, kb_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at DESC",
                (kb_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def find_document_by_hash(self, kb_id: str, sha256: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? AND sha256 = ?",
                (kb_id, sha256),
            ).fetchone()
            return dict(row) if row else None

    def rename_document(self, doc_id: str, new_name: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.execute("UPDATE documents SET name = ? WHERE id = ?", (new_name, doc_id))
            conn.execute("UPDATE chunks SET doc_name = ? WHERE doc_id = ?", (new_name, doc_id))
        return self.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        with self._get_connection() as conn:
            row = conn.execute("SELECT file_path FROM documents WHERE id = ?", (doc_id,)).fetchone()
            file_path = row["file_path"] if row else None
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            deleted = cur.rowcount > 0

        if deleted and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        return deleted

    # ------------------ Chunks ------------------

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            if data["embedding_json"]:
                data["embedding"] = json.loads(data["embedding_json"])
            else:
                data["embedding"] = None
            return data

    def get_chunks_for_kb(self, kb_id: str) -> List[Chunk]:
        """Load all committed chunks for a Knowledge Base."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.doc_id, c.doc_name, c.page_number, c.chunk_index,
                       c.text, c.token_count, c.embedding_model, c.embedding_json
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.kb_id = ? AND d.status = 'indexed'
                ORDER BY c.doc_id, c.chunk_index ASC
                """,
                (kb_id,),
            ).fetchall()

            chunks: List[Chunk] = []
            for r in rows:
                emb = json.loads(r["embedding_json"]) if r["embedding_json"] else None
                chunks.append(
                    Chunk(
                        id=r["id"],
                        doc_id=r["doc_id"],
                        doc_name=r["doc_name"],
                        page_number=r["page_number"],
                        chunk_index=r["chunk_index"],
                        text=r["text"],
                        token_count=r["token_count"],
                        embedding=emb,
                        embedding_model=r["embedding_model"],
                    )
                )
            return chunks

    # ------------------ Conversations & Messages ------------------

    def create_conversation(self, conv_id: str, kb_id: str, title: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, kb_id, title) VALUES (?, ?, ?)",
                (conv_id, kb_id, title),
            )
        return self.get_conversation(conv_id)  # type: ignore

    def list_conversations(self, kb_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE kb_id = ? ORDER BY created_at DESC",
                (kb_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            return dict(row) if row else None

    def rename_conversation(self, conv_id: str, new_title: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conv_id))
        return self.get_conversation(conv_id)

    def delete_conversation(self, conv_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            return cur.rowcount > 0

    def add_message(
        self,
        msg_id: str,
        conv_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cit_json = json.dumps(citations) if citations is not None else None
        dbg_json = json.dumps(debug) if debug is not None else None
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, citations_json, debug_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conv_id, role, content, cit_json, dbg_json),
            )
        return {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": role,
            "content": content,
            "citations": citations,
            "debug": debug,
        }

    def list_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conv_id,),
            ).fetchall()
            messages = []
            for r in rows:
                m = dict(r)
                m["citations"] = json.loads(m["citations_json"]) if m["citations_json"] else []
                m["debug"] = json.loads(m["debug_json"]) if m["debug_json"] else {}
                messages.append(m)
            return messages

    def search_conversations(self, kb_id: str, query: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            pattern = f"%{query}%"
            rows = conn.execute(
                """
                SELECT DISTINCT c.id, c.kb_id, c.title, c.created_at
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.kb_id = ? AND (c.title LIKE ? OR m.content LIKE ?)
                ORDER BY c.created_at DESC
                """,
                (kb_id, pattern, pattern),
            ).fetchall()
            return [dict(r) for r in rows]
