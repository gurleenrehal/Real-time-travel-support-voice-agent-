# Evaluation Methodology & Results

These numbers come from an actual run of `python scripts/evaluate.py`
against `data/evaluation/eval_set.json` (30 hand-labeled synthetic
conversations across all 10 required categories), on the mock-mode
pipeline (no API keys). They are **not** hand-tuned to look good — see
"Known weaknesses" below for the honest read.

## Latest run

```json
{
  "total_cases": 30,
  "retrieval_hit_rate": 0.70,
  "retrieval_precision_avg": 0.4667,
  "retrieval_relevance_avg_score": 0.2653,
  "tool_call_success_rate": 0.6333,
  "handoff_accuracy": 0.6667,
  "response_groundedness_rate_on_answerable": 0.8333,
  "unsupported_claim_rate_overall": 0.20,
  "correct_unsupported_rate_on_unanswerable": 0.3333,
  "stt_accuracy": null,
  "latency_ms_summary": {
    "retrieval_ms_avg": 1.52,
    "tool_ms_avg": 0.01,
    "llm_ms_avg": 0.005,
    "total_ms_avg": 4.30
  }
}
```

Regenerate with: `python scripts/ingest.py && python scripts/evaluate.py`
(the full per-case breakdown is written to `data/evaluation/report.json`).

## What each metric means here

- **retrieval_hit_rate**: fraction of cases where at least one
  hand-labeled relevant doc was retrieved (for cases with no relevant
  doc expected, a "hit" means correctly retrieving nothing).
- **retrieval_precision_avg**: of the docs retrieved, what fraction were
  actually relevant per the hand-labeled ground truth.
- **tool_call_success_rate**: whether the expected tool (or no tool, when
  none was expected) was actually invoked.
- **handoff_accuracy**: predicted `handoff_required` vs. the hand-labeled
  expectation.
- **response_groundedness_rate_on_answerable**: of the cases that should
  have a grounded answer, how many actually got one (vs. falling back to
  "I don't know").
- **unsupported_claim_rate_overall / correct_unsupported_rate_on_unanswerable**:
  together, a hallucination-risk proxy. Because the mock LLM is
  extractive (it can only ever quote a retrieved doc or a tool result,
  never free-generate), it cannot literally invent policy text — so this
  isn't measuring "did it lie", it's measuring "did it correctly know
  when it didn't have an answer."
- **stt_accuracy**: intentionally `null`. There is no real
  audio + reference-transcript pair available in this sandboxed project,
  and fabricating one would violate the project's own authenticity
  requirement. A real deployment would compute word-error-rate against
  recorded audio here.

## Known weaknesses (found by this evaluation, not hidden)

- **TF-IDF retrieval on short/vague queries** (`retrieval_precision_avg`
  0.47) sometimes pulls in lexically-overlapping but semantically
  irrelevant documents — e.g. a query with just the word "flight" can
  weakly match several unrelated docs that all mention flights. This is
  the direct motivation for the `evaluate_retrieval` confidence-floor
  gate in the LangGraph workflow (see docs/ARCHITECTURE.md), which
  suppresses low-confidence retrieval from being used for grounding —
  but it doesn't fully close the gap, as `correct_unsupported_rate_on_unanswerable`
  (0.33) shows: two-thirds of genuinely unanswerable queries in this run
  still got a (confidently low-scored, handed-off) response rather than
  a clean "I don't know."
- **tool_call_success_rate** (0.63) is held down mostly by cases where a
  question implies a tool (e.g. rebooking) but the keyword-matching
  tool-selector in the mock LLM doesn't fire on the exact phrasing used.
  A real LLM with function-calling would do semantic tool selection
  instead of keyword matching and should do meaningfully better here.
- These numbers reflect the **mock-mode** pipeline specifically, since
  that's what runs in CI and in this sandboxed environment. A deployment
  with a real LLM (semantic tool selection, better paraphrase handling)
  and real embeddings would be expected to score higher on retrieval
  precision and tool-call success — but that is a prediction, not a
  measured claim.

## Confidence heuristic (explicit, not calibrated)

`confidence = 0.6 * retrieval_component + 0.4 * tool_component`, where
`retrieval_component` is the top retrieved document's similarity score
(or a low default if retrieval was gated out / absent), and
`tool_component` is 1.0 for a successful tool call, 0.3 for a failed
lookup, or 0.5 (neutral) when no tool was needed. This is a documented
heuristic, not a calibrated probability — see
docs/ARCHITECTURE.md for what calibrating it properly would require.
