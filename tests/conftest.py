"""Shared pytest fixtures. Forces mock mode (no API keys needed) and uses
an isolated temp Chroma directory per test session so tests never depend
on or mutate the developer's real data/chroma_store."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("ELEVENLABS_API_KEY", "")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def test_vector_store(tmp_path_factory, project_root):
    from app.rag.ingest import build_vector_store

    persist_dir = tmp_path_factory.mktemp("chroma_test_store")
    store, count = build_vector_store(
        knowledge_base_dir=str(project_root / "data" / "knowledge_base"),
        persist_dir=str(persist_dir),
    )
    assert count == 20
    return store


@pytest.fixture()
def retriever(test_vector_store):
    from app.rag.retriever import Retriever

    return Retriever(test_vector_store)


@pytest.fixture()
def llm_service():
    from app.services.llm import LLMService

    return LLMService()


@pytest.fixture()
def settings():
    from app.config import get_settings

    return get_settings()


@pytest.fixture()
def graph(retriever, llm_service, settings):
    from app.graph.workflow import build_graph

    return build_graph(retriever, llm_service, settings)


@pytest.fixture()
def client(test_vector_store, monkeypatch):
    """A TestClient wired to the same test vector store, so REST/WS tests
    don't touch the real persisted data/chroma_store."""
    import app.main as main_module

    monkeypatch.setattr(main_module, "vector_store", test_vector_store)
    monkeypatch.setattr(main_module, "retriever", main_module.Retriever(test_vector_store))
    monkeypatch.setattr(
        main_module, "graph",
        main_module.build_graph(main_module.retriever, main_module.llm_service, main_module.settings),
    )

    from fastapi.testclient import TestClient

    return TestClient(main_module.app)
