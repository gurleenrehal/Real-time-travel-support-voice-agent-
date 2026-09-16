def test_retrieval_returns_relevant_doc(retriever):
    results = retriever.retrieve("What happens if my flight is cancelled by the airline?")
    assert len(results) > 0
    assert any(r["doc_id"] == "kb-001" for r in results)


def test_retrieval_scores_are_sorted_desc(retriever):
    results = retriever.retrieve("baggage allowance checked bag weight limit")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_respects_top_k(retriever):
    results = retriever.retrieve("flight delay compensation", top_k=1)
    assert len(results) <= 1


def test_retrieval_returns_empty_for_empty_store(tmp_path):
    from app.rag.vector_store import VectorStore
    from app.rag.retriever import Retriever

    empty_store = VectorStore(persist_dir=str(tmp_path / "empty_store"))
    empty_retriever = Retriever(empty_store)
    assert empty_retriever.retrieve("anything at all") == []
