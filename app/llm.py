"""
LLM Provider integrations: Extractive, Ollama, OpenAI-compatible.
Includes timeout handling, retries, and clean 502/503 error mapping.
"""

from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx

from app.config import settings
from ragcore import Chunk

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an external LLM provider fails."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model = model or settings.LLM_MODEL
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[str, str, str]:
        """
        Generate answer based on retrieved context chunks.
        Returns (answer_text, provider_name, model_name).
        Raises LLMError on external provider outage or timeout.
        """
        # If no relevant context retrieved, answer from general knowledge if LLM is active
        if not retrieved_chunks or all(s <= 0.0 for _, s in retrieved_chunks):
            if self.provider == "ollama":
                return self._call_ollama_general_knowledge(query)
            elif self.provider in ("openai_compatible", "openai"):
                return self._call_openai_general_knowledge(query)
            else:
                return (
                    "No relevant context found in the selected documents for this question.",
                    self.provider,
                    "extractive",
                )

        if self.provider == "ollama":
            return self._call_ollama(query, retrieved_chunks)
        elif self.provider in ("openai_compatible", "openai"):
            return self._call_openai_compatible(query, retrieved_chunks)
        else:
            # Default: extractive
            return self._generate_extractive(query, retrieved_chunks)

    def _generate_extractive(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[str, str, str]:
        """Grounded extractive answer without hallucination."""
        import re
        from ragcore import _tokenize, STOP_WORDS

        best_chunk, top_score = retrieved_chunks[0]

        q_tokens = set(_tokenize(query)) - STOP_WORDS
        if not q_tokens:
            q_tokens = set(_tokenize(query))

        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", best_chunk.text) if s.strip()]
        scored_sentences = []
        for sent in raw_sentences:
            s_tokens = set(_tokenize(sent))
            overlap = len(q_tokens.intersection(s_tokens))
            if overlap > 0:
                scored_sentences.append((sent, overlap))

        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentence = scored_sentences[0][0]
            answer = (
                f"**[Extractive Mode]** Grounded answer from **{best_chunk.doc_name}** (Page {best_chunk.page_number}):\n\n"
                f"> \"{best_sentence}\"\n\n"
                f"*(Full excerpt available in citations below)*"
            )
        else:
            answer = (
                f"**[Extractive Mode]** Grounded answer from **{best_chunk.doc_name}** (Page {best_chunk.page_number}):\n\n"
                f"> \"{best_chunk.text.strip()}\""
            )
        return answer, "extractive", "extractive"

    def _execute_ollama_prompt(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        last_error = None
        for attempt in range(2):
            try:
                with httpx.Client(timeout=45.0) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("response", "").strip()
                    else:
                        raise LLMError(
                            f"Ollama returned HTTP {resp.status_code}: {resp.text}",
                            status_code=502,
                        )
            except httpx.ConnectError:
                last_error = LLMError(
                    f"Unable to connect to Ollama at {self.base_url}. Ensure Ollama is running ('ollama serve').",
                    status_code=503,
                )
            except httpx.TimeoutException:
                last_error = LLMError(
                    f"Ollama request timed out after 45 seconds for model '{self.model}'.",
                    status_code=504,
                )
            except LLMError as e:
                last_error = e
            except Exception as e:
                last_error = LLMError(f"Unexpected error communicating with Ollama: {e}", status_code=502)

            time.sleep(0.5)

        if last_error:
            raise last_error
        raise LLMError("Failed to reach Ollama provider.", status_code=503)

    def _call_ollama(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[str, str, str]:
        relevant = [(c, s) for c, s in retrieved_chunks if s > 0.0]
        if not relevant:
            return self._call_ollama_general_knowledge(query)

        context_str = "\n\n---\n\n".join(
            f"Source: {c.doc_name} (Page {c.page_number})\nExcerpt: {c.text}"
            for c, _ in relevant
        )

        prompt = (
            "You are an intelligent, helpful AI assistant like ChatGPT or Claude, equipped with knowledge from uploaded documents.\n\n"
            f"Document Context:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            "Instructions:\n"
            "- If the question relates to the provided documents, answer thoroughly, accurately, and politely, grounding your response in the document context.\n"
            "- If the question asks for a summary or overview of the documents, provide a well-structured summary.\n"
            "- If the question is partially covered or asks something related, combine the excerpts with your general knowledge to provide the most helpful, complete answer.\n\n"
            "Answer:"
        )
        answer = self._execute_ollama_prompt(prompt)
        return answer, "ollama", self.model

    def _call_ollama_general_knowledge(self, query: str) -> Tuple[str, str, str]:
        prompt = (
            "You are a helpful, intelligent, knowledgeable, and polite AI assistant like ChatGPT or Claude.\n"
            "Answer the user's question clearly, comprehensively, and accurately.\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )
        answer = self._execute_ollama_prompt(prompt)
        return answer, "ollama", self.model

    def _call_openai_compatible(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[str, str, str]:
        context_str = "\n\n---\n\n".join(
            f"Source: {c.doc_name} (Page {c.page_number})\nExcerpt: {c.text}"
            for c, _ in retrieved_chunks
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a local RAG assistant. Answer the user question strictly using the provided context excerpts. "
                    "Always mention the source document name and page number."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {query}",
            },
        ]

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }

        last_error = None
        for attempt in range(2):
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer_text = data["choices"][0]["message"]["content"].strip()
                        return answer_text, "openai_compatible", self.model
                    else:
                        raise LLMError(
                            f"OpenAI-compatible endpoint returned HTTP {resp.status_code}: {resp.text}",
                            status_code=502,
                        )
            except httpx.ConnectError:
                last_error = LLMError(
                    f"Unable to connect to OpenAI-compatible endpoint at {self.base_url}.",
                    status_code=503,
                )
            except httpx.TimeoutException:
                last_error = LLMError(
                    f"OpenAI-compatible request timed out after 30 seconds for model '{self.model}'.",
                    status_code=504,
                )
            except LLMError as e:
                last_error = e
            except Exception as e:
                last_error = LLMError(f"Unexpected error with LLM provider: {e}", status_code=502)

            time.sleep(0.5)

        if last_error:
            raise last_error
        raise LLMError("Failed to reach LLM provider.", status_code=503)

    def _call_openai_general_knowledge(self, query: str) -> Tuple[str, str, str]:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful, intelligent, knowledgeable AI assistant like ChatGPT or Claude.",
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    answer_text = data["choices"][0]["message"]["content"].strip()
                    return answer_text, "openai_compatible", self.model
                else:
                    raise LLMError(f"OpenAI error: {resp.text}", status_code=502)
        except Exception as e:
            raise LLMError(f"OpenAI error: {e}", status_code=502)
