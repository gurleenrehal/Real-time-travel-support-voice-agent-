"""
Vector store backed by ChromaDB, using a locally-fit TF-IDF embedding
function.

Why TF-IDF instead of an OpenAI/HF embedding API: this project must run
and be tested with zero API keys and no network access to model hubs.
TF-IDF over the (small, domain-specific) travel-support corpus gives a
genuinely functional, inspectable retrieval signal without depending on
an external service. Swapping in a real embedding API is a one-line
change (implement another class with the same `__call__` signature and
pass it to `VectorStore`) -- see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma-compatible embedding function backed by a fitted TF-IDF model."""

    def __init__(self, vectorizer: TfidfVectorizer | None = None) -> None:
        self.vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=2048
        )
        self._fitted = vectorizer is not None

    def fit(self, corpus: list[str]) -> None:
        self.vectorizer.fit(corpus)
        self._fitted = True

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        if not self._fitted:
            # Degrade gracefully rather than crash: fit on-the-fly on
            # whatever text we've been given (only happens if a query
            # comes in before ingestion, which tests exercise directly).
            self.fit(input)
        matrix = self.vectorizer.transform(input)
        return matrix.toarray().tolist()

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self.vectorizer))

    @classmethod
    def load(cls, path: Path) -> "TfidfEmbeddingFunction":
        vectorizer = pickle.loads(path.read_bytes())
        return cls(vectorizer=vectorizer)

    # -- chromadb's EmbeddingFunction protocol requires these three ------
    @staticmethod
    def name() -> str:
        return "tfidf-local"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "TfidfEmbeddingFunction":
        return TfidfEmbeddingFunction()


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    COLLECTION_NAME = "travel_support_kb"
    VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"

    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        vectorizer_path = self.persist_dir / self.VECTORIZER_FILENAME
        if vectorizer_path.exists():
            self.embedder = TfidfEmbeddingFunction.load(vectorizer_path)
        else:
            self.embedder = TfidfEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def is_ready(self) -> bool:
        try:
            return self._collection.count() > 0
        except Exception:
            return False

    def ingest(self, documents: list[dict[str, Any]]) -> int:
        """(Re)build the collection from a list of {doc_id, text, category} dicts."""
        texts = [d["text"] for d in documents]
        self.embedder.fit(texts)
        self.embedder.save(self.persist_dir / self.VECTORIZER_FILENAME)

        # Recreate the collection so re-ingestion is idempotent.
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            ids=[d["doc_id"] for d in documents],
            documents=texts,
            metadatas=[{"category": d.get("category", "unknown")} for d in documents],
        )
        return len(documents)

    def query(self, text: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.is_ready:
            return []
        result = self._collection.query(query_texts=[text], n_results=top_k)
        out = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc_id, doc_text, meta, distance in zip(ids, docs, metas, distances):
            # Chroma cosine "distance" is 1 - cosine_similarity for normalized
            # vectors; clamp defensively since TF-IDF vectors aren't unit-normalized.
            similarity = max(0.0, 1.0 - distance)
            out.append({
                "doc_id": doc_id,
                "text": doc_text,
                "score": round(similarity, 4),
                "metadata": meta,
            })
        return out
