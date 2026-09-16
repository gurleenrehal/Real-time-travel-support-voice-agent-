#!/usr/bin/env python3
"""CLI: (re)build the vector store from the synthetic knowledge base.

Usage:
    python scripts/ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag.ingest import build_vector_store


def main() -> None:
    settings = get_settings()
    store, count = build_vector_store(settings.knowledge_base_dir, settings.chroma_persist_dir)
    print(f"Ingested {count} documents into '{store.COLLECTION_NAME}' at {settings.chroma_persist_dir}")
    print(f"Collection ready: {store.is_ready}")


if __name__ == "__main__":
    main()
