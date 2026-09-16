"""
Loads the synthetic knowledge base JSON and ingests it into a VectorStore.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.rag.vector_store import VectorStore


def load_documents(knowledge_base_dir: str) -> list[dict]:
    kb_path = Path(knowledge_base_dir) / "kb_documents.json"
    data = json.loads(kb_path.read_text())
    return data["documents"]


def build_vector_store(knowledge_base_dir: str, persist_dir: str) -> tuple[VectorStore, int]:
    documents = load_documents(knowledge_base_dir)
    store = VectorStore(persist_dir=persist_dir)
    count = store.ingest(documents)
    return store, count
