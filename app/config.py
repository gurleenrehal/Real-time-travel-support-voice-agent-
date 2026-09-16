"""
Central configuration for the Real-Time Travel Support Voice Agent.

Everything that varies between environments (API keys, thresholds, model
names) is read from environment variables so no secret is ever hard-coded
in source. When a key is absent, the corresponding service falls back to
a deterministic mock implementation instead of failing -- this is what
lets the whole system run and be tested with zero API keys.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- External providers (all optional; absence => mock mode) ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    whisper_model_size: str = "base"

    # --- RAG ---
    chroma_persist_dir: str = "data/chroma_store"
    knowledge_base_dir: str = "data/knowledge_base"
    retrieval_top_k: int = 3
    retrieval_score_threshold: float = 0.15  # cosine similarity, TF-IDF space

    # --- Confidence / handoff thresholds ---
    confidence_handoff_threshold: float = 0.45
    retrieval_confidence_floor: float = 0.20

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- Evaluation ---
    evaluation_set_path: str = "data/evaluation/eval_set.json"
    evaluation_report_path: str = "data/evaluation/report.json"

    @property
    def llm_mock_mode(self) -> bool:
        return not bool(self.openai_api_key)

    @property
    def tts_mock_mode(self) -> bool:
        return not bool(self.elevenlabs_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
