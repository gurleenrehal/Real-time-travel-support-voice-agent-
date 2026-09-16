# Architecture

## Pipeline

```
User (mic or text)
      |
WebSocket /ws/voice (FastAPI)
      |
  STT (real faster-whisper if available, else offline mock)
      |
LangGraph workflow:
  receive_turn -> retrieve_context -> evaluate_retrieval
      -> tool_call_if_required -> generate_response
      -> evaluate_confidence -> decide_handoff
      -> (conditional) finalize_handoff -> END
      |
  TTS (real ElevenLabs if API key set, else offline mock WAV)
      |
Streaming events back to client (transcript, agent_response, audio_response, metrics, handoff)
```

## Why LangGraph actually controls the workflow (not decorative)

Two things earlier in the graph change what happens later, for real:

1. `evaluate_retrieval` computes `retrieval_ok` (average-floor check on
   retrieval scores). `generate_response` reads this and does **not**
   pass retrieved documents to the responder when `retrieval_ok` is
   `False` — even if the retriever's top-k technically cleared its own
   threshold. This exists because short/vague queries against a
   TF-IDF index can retrieve spuriously-matching low-relevance docs;
   without this gate the agent would confidently "quote policy" that
   isn't actually relevant.
2. `decide_handoff`'s output (`handoff_required`) drives a genuine
   **conditional edge**: `finalize_handoff` only runs, and only then
   appends the "connecting you to a human agent" ticket message, when
   handoff was actually decided. `tests/test_graph_workflow.py` asserts
   both branches.

## Why TF-IDF instead of an embeddings API

Every acceptance requirement for this project (tests must run without
API keys; the repo must not depend on unreachable external services)
ruled out calling OpenAI/Cohere embeddings or downloading a
sentence-transformers model from Hugging Face at ingest time. Instead,
`app/rag/vector_store.py` fits a `scikit-learn` `TfidfVectorizer` on the
20-document synthetic knowledge base and stores those vectors in a real,
persistent ChromaDB collection via a custom `EmbeddingFunction`. This is
a genuinely functional retrieval signal for a small, lexically distinct
domain like this one — not a placeholder — but it is not semantic
retrieval, and it will not generalize to paraphrases the way a real
embedding model would (documented further in docs/EVALUATION.md).
Swapping in a real embedding API is a ~10-line change: implement another
class with the same `__call__(input: list[str]) -> list[list[float]]`
signature and pass it to `VectorStore.__init__`.

## What is real vs. mocked, precisely

| Component | Real implementation | Mock fallback | When mock is used |
|---|---|---|---|
| Retrieval | ChromaDB + TF-IDF vectors | — (always real; TF-IDF *is* the real embedding here) | never |
| LangGraph orchestration | Full StateGraph, real conditional routing | — | never |
| Tools | Function calls, but backed by an in-memory synthetic dataset (not a live airline API) | — | always (by design — see project brief) |
| LLM | OpenAI-compatible chat completion + function calling | Deterministic extractive responder (grounds in top doc / tool result, or admits no answer) | `OPENAI_API_KEY` unset (default in this sandboxed build) |
| STT | faster-whisper | Decodes UTF-8 text sent as "audio" bytes (protocol stand-in) | faster-whisper not installed/model unavailable (default here) |
| TTS | ElevenLabs REST API | Locally-generated silent WAV, duration scaled to text length | `ELEVENLABS_API_KEY` unset (default here) |
| Barge-in | `PlaybackController.interrupt()` — real state tracking, wired into the WebSocket handler | Cannot cancel mid-stream audio because mock TTS returns a complete file, not a stream | always, until real streaming TTS is integrated |
| Observability | Real structured JSON logs + real per-stage latency timers | — | never |

## What a production version would add

- Real embeddings (or a hybrid TF-IDF + dense reranker) for genuine
  semantic retrieval.
- Real streaming STT/TTS with actual mid-stream barge-in cancellation.
- Prometheus/Grafana export instead of the in-memory `/metrics` summary.
- A calibrated confidence model (the current one is an explicit,
  documented heuristic — see docs/EVALUATION.md) trained against labeled
  human-agent-escalation outcomes.
- Persistent conversation memory across turns (this project is
  single-turn stateless per WebSocket message; there is no multi-turn
  dialogue state beyond what a session's `PlaybackController` tracks).
