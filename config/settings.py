"""
EarningsEdge configuration module.

Centralises all environment variables, API keys, model identifiers, and
runtime constants using pydantic-settings.  Every other module imports from
here — no magic strings or hardcoded values anywhere else.

Usage:
    from config.settings import settings
    print(settings.GROQ_MODEL)
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Project root (two levels up from this file: config/ → earningsedge/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Typed, validated settings loaded from the .env file at project root.

    All fields with no default MUST be provided via the .env file before
    running any pipeline component.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM — Primary (Groq) + Fallback (Gemini)
    # ------------------------------------------------------------------
    GROQ_API_KEY: str = Field(..., description="Groq API key — get at console.groq.com")
    GOOGLE_API_KEY: str = Field(..., description="Google AI Studio key — aistudio.google.com")

    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier for primary LLM calls",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model identifier used as fallback",
    )

    # Max tokens for LLM responses
    LLM_MAX_TOKENS: int = Field(default=4096)
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)

    # ------------------------------------------------------------------
    # Embeddings + NLP models (HuggingFace Hub IDs)
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Primary dense embedding model",
    )
    FINBERT_MODEL: str = Field(
        default="ProsusAI/finbert",
        description="Financial-domain sentiment model",
    )
    NLI_MODEL: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="NLI cross-encoder for contradiction detection",
    )
    SPACY_MODEL: str = Field(
        default="en_core_web_trf",
        description="spaCy pipeline for NER",
    )

    # ------------------------------------------------------------------
    # Storage paths (relative to PROJECT_ROOT; created at runtime)
    # ------------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = Field(default="data/chroma_db")
    RAW_DATA_DIR: str = Field(default="data/raw")
    PROCESSED_DATA_DIR: str = Field(default="data/processed")
    MLFLOW_TRACKING_URI: str = Field(default="data/mlflow_runs")

    # ------------------------------------------------------------------
    # SEC / EDGAR
    # ------------------------------------------------------------------
    SEC_USER_AGENT: str = Field(
        default="EarningsEdge contact@earningsedge.dev",
        description="Required by SEC EDGAR — use real name/email in production",
    )
    SEC_BASE_URL: str = Field(default="https://data.sec.gov")
    EDGAR_FULL_TEXT_URL: str = Field(
        default="https://efts.sec.gov/LATEST/search-index"
    )
    # EDGAR politely asks ≤10 req/s
    SEC_RATE_LIMIT_RPS: float = Field(default=8.0, description="Requests per second")

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = Field(default=512, ge=64, le=2048)
    CHUNK_OVERLAP: int = Field(default=64, ge=0)
    MAX_CHUNKS_PER_DOC: int = Field(default=500, ge=10)
    # Cosine similarity threshold below which semantic chunker splits
    SEMANTIC_SPLIT_THRESHOLD: float = Field(default=0.75, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    TOP_K_RETRIEVAL: int = Field(default=10, ge=1)
    METADATA_FILTER_ENABLED: bool = Field(default=True)
    # Cosine similarity threshold for pairing chunks in contradiction check
    CONTRADICTION_SIMILARITY_THRESHOLD: float = Field(default=0.75)
    CONTRADICTION_SCORE_THRESHOLD: float = Field(default=0.70)

    # ------------------------------------------------------------------
    # Tone drift detection thresholds
    # ------------------------------------------------------------------
    DRIFT_STABLE_THRESHOLD: float = Field(
        default=0.10,
        description="Abs sentiment change below this = stable",
    )
    # Hedging words used to compute the hedging language score
    HEDGING_WORDS: list[str] = Field(
        default=[
            "approximately", "may", "could", "might", "subject to",
            "uncertain", "uncertainty", "challenging", "potential",
            "expect", "believe", "anticipate", "assume", "estimate",
            "if", "should", "would", "possibly", "likely", "unlikely",
        ]
    )

    # ------------------------------------------------------------------
    # Earnings quality scorer weights (must sum to 1.0)
    # ------------------------------------------------------------------
    SCORE_WEIGHT_SENTIMENT_DRIFT: float = Field(default=0.30)
    SCORE_WEIGHT_GUIDANCE_ACCURACY: float = Field(default=0.25)
    SCORE_WEIGHT_ACCRUALS: float = Field(default=0.25)
    SCORE_WEIGHT_ANALYST_REVISION: float = Field(default=0.20)

    @field_validator("SCORE_WEIGHT_ANALYST_REVISION")
    @classmethod
    def weights_sum_to_one(cls, v: float, info: object) -> float:
        """Validate that composite score weights sum to 1.0."""
        values = getattr(info, "data", {})
        total = (
            values.get("SCORE_WEIGHT_SENTIMENT_DRIFT", 0.30)
            + values.get("SCORE_WEIGHT_GUIDANCE_ACCURACY", 0.25)
            + values.get("SCORE_WEIGHT_ACCRUALS", 0.25)
            + v
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Composite score weights must sum to 1.0, got {total:.4f}"
            )
        return v

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------
    BACKTEST_START_DATE: str = Field(default="2019-01-01")
    BACKTEST_END_DATE: str = Field(default="2024-12-31")
    INITIAL_CAPITAL: float = Field(default=100_000.0)
    SIGNAL_LONG_THRESHOLD: float = Field(
        default=0.30,
        description="Score above this → LONG signal",
    )
    SIGNAL_SHORT_THRESHOLD: float = Field(
        default=-0.30,
        description="Score below this → SHORT signal",
    )
    HOLDING_PERIOD_DAYS: int = Field(
        default=20,
        description="Trading days to hold position after earnings",
    )
    RISK_FREE_RATE: float = Field(
        default=0.045,
        description="Annual risk-free rate for Sharpe computation",
    )

    # ------------------------------------------------------------------
    # RAG graph
    # ------------------------------------------------------------------
    MAX_RAG_HOPS: int = Field(default=3, description="Max retrieval loops in LangGraph")
    QUALITY_GATE_THRESHOLD: float = Field(
        default=0.70,
        description="Min quality score before answer is accepted",
    )

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    CACHE_DIR: str = Field(default="data/.cache")
    CACHE_TTL_SECONDS: int = Field(default=86_400, description="24 hours")

    # ------------------------------------------------------------------
    # Convenience properties (derived paths as Path objects)
    # ------------------------------------------------------------------
    @property
    def chroma_path(self) -> Path:
        return PROJECT_ROOT / self.CHROMA_PERSIST_DIR

    @property
    def raw_data_path(self) -> Path:
        return PROJECT_ROOT / self.RAW_DATA_DIR

    @property
    def processed_data_path(self) -> Path:
        return PROJECT_ROOT / self.PROCESSED_DATA_DIR

    @property
    def mlflow_path(self) -> Path:
        return PROJECT_ROOT / self.MLFLOW_TRACKING_URI

    @property
    def cache_path(self) -> Path:
        return PROJECT_ROOT / self.CACHE_DIR

    def ensure_dirs(self) -> None:
        """Create all required data directories if they do not exist."""
        dirs = [
            self.chroma_path,
            self.raw_data_path / "sec_filings",
            self.raw_data_path / "transcripts",
            self.raw_data_path / "analyst_data",
            self.processed_data_path,
            self.mlflow_path,
            self.cache_path,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()


if __name__ == "__main__":
    # Quick smoke-test: print resolved settings (masks secret values)
    import json

    settings.ensure_dirs()
    safe = {
        k: ("***" if "KEY" in k or "SECRET" in k else v)
        for k, v in settings.model_dump().items()
    }
    print(json.dumps(safe, indent=2, default=str))
    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"ChromaDB path: {settings.chroma_path}")
    print("All directories ensured ✓")
