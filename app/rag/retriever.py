"""
Retriever: applies top-k retrieval plus a similarity threshold on top of
the raw VectorStore, and exposes the source documents used so the LLM
service can ground its response in them and so the API can return
"which sources were used" to the caller.
"""
from __future__ import annotations

from app.config import get_settings
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.settings.retrieval_top_k
        results = self.vector_store.query(query, top_k=top_k)
        return [r for r in results if r["score"] >= self.settings.retrieval_score_threshold]
