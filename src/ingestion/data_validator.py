"""
EarningsEdge data models (Pydantic v2).

Defines the canonical data contracts for every object that flows through the
pipeline:

    SECFiling          — raw EDGAR filing (10-K / 10-Q / 8-K)
    EarningsTranscript — parsed earnings call transcript
    AnalystData        — consensus estimates + price targets
    DocumentChunk      — post-chunking unit stored in ChromaDB
    GuidanceEntity     — forward-looking numerical guidance extracted by NER
    SentimentResult    — FinBERT output per chunk
    QuarterSentiment   — aggregated sentiment for a single quarter
    ToneDriftReport    — quarter-over-quarter drift analysis
    Contradiction      — NLI-detected conflict between two chunks
    GuidanceRecord     — one extracted guidance item with actuals matched
    EarningsQualityScore — composite signal with component breakdown
    BacktestSignal     — single trade generated from the quality score
    TradeResult        — realised PnL for one backtest position
    RAGEvaluation      — RAGAS metric bundle for one query

All models use strict validation where it adds safety without friction.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared Enums
# ---------------------------------------------------------------------------


class FilingType(str, Enum):
    FORM_10K = "10-K"
    FORM_10K_A = "10-K/A"
    FORM_10Q = "10-Q"
    FORM_10Q_A = "10-Q/A"
    FORM_8K = "8-K"
    TRANSCRIPT = "transcript"
    ANALYST = "analyst"


class SectionType(str, Enum):
    RISK_FACTORS = "risk_factors"
    MDA = "mda"               # Management Discussion & Analysis
    GUIDANCE = "guidance"
    FINANCIAL_STATEMENTS = "financial_statements"
    QA = "qa"
    PREPARED_REMARKS = "prepared_remarks"
    PRESS_RELEASE = "press_release"
    COVER = "cover"
    UNKNOWN = "unknown"


class Quarter(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    ANNUAL = "Annual"


class DriftDirection(str, Enum):
    IMPROVING = "improving"
    DETERIORATING = "deteriorating"
    STABLE = "stable"


class AlertLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class ChunkingStrategy(str, Enum):
    SECTION_AWARE = "section_aware"
    SPEAKER_TURN = "speaker_turn"
    SEMANTIC = "semantic"
    FIXED_OVERLAP = "fixed_overlap"


class GuidanceMetric(str, Enum):
    EPS = "EPS"
    REVENUE = "revenue"
    CAPEX = "capex"
    MARGIN = "margin"
    GROWTH = "growth"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Ingestion layer models
# ---------------------------------------------------------------------------


class XBRLData(BaseModel):
    """Structured financial figures extracted from XBRL-tagged filings."""

    eps_basic: float | None = None
    eps_diluted: float | None = None
    revenue: float | None = None          # in USD
    net_income: float | None = None
    operating_cash_flow: float | None = None
    total_assets: float | None = None
    capex: float | None = None
    gross_margin: float | None = None     # as decimal, e.g. 0.42
    operating_margin: float | None = None

    class Config:
        extra = "allow"               # additional XBRL tags stored dynamically


class SECFiling(BaseModel):
    """Canonical representation of one SEC filing downloaded from EDGAR."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker")
    cik: str = Field(..., description="SEC Central Index Key (zero-padded 10-digit str)")
    filing_type: FilingType
    period_of_report: date = Field(..., description="The reporting period end date")
    filed_date: date = Field(..., description="Date the filing was submitted to SEC")
    filing_url: str = Field(..., description="Primary document URL on EDGAR")
    accession_number: str = Field(..., description="EDGAR accession number")
    raw_text: str = Field(default="", description="Full plain-text content of filing")
    xbrl_data: XBRLData = Field(default_factory=XBRLData)
    quarter: Quarter | None = None
    year: int | None = None
    is_amendment: bool = False
    word_count: int = Field(default=0, ge=0)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("cik")
    @classmethod
    def pad_cik(cls, v: str) -> str:
        return v.strip().zfill(10)

    @model_validator(mode="after")
    def set_year_from_period(self) -> "SECFiling":
        if self.year is None and self.period_of_report:
            self.year = self.period_of_report.year
        return self


class SpeakerTurn(BaseModel):
    """Single speaker turn within an earnings call transcript."""

    speaker_name: str
    speaker_role: str = Field(
        default="unknown",
        description="CEO / CFO / Analyst / Operator / Other",
    )
    text: str
    turn_index: int = Field(ge=0)
    section: str = Field(
        default="prepared_remarks",
        description="'prepared_remarks' or 'qa'",
    )


class EarningsTranscript(BaseModel):
    """Parsed earnings call transcript with structured speaker turns."""

    ticker: str
    quarter: Quarter
    year: int = Field(ge=1990)
    call_date: date
    prepared_remarks: list[SpeakerTurn] = Field(default_factory=list)
    qa_section: list[SpeakerTurn] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    raw_text: str = Field(default="")
    source_url: str = Field(default="")
    word_count: int = Field(default=0, ge=0)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()

    @model_validator(mode="after")
    def collect_speakers(self) -> "EarningsTranscript":
        if not self.speakers:
            seen: set[str] = set()
            for turn in self.prepared_remarks + self.qa_section:
                seen.add(turn.speaker_name)
            self.speakers = sorted(seen)
        return self


class AnalystEstimate(BaseModel):
    """Single analyst estimate data point."""

    period: str = Field(description="e.g. '0q' current quarter, '+1y' next year")
    eps_mean: float | None = None
    eps_low: float | None = None
    eps_high: float | None = None
    revenue_mean: float | None = None
    num_analysts: int | None = None


class AnalystData(BaseModel):
    """Analyst consensus data for a ticker pulled via yfinance."""

    ticker: str
    fetch_date: date
    price_target_mean: float | None = None
    price_target_high: float | None = None
    price_target_low: float | None = None
    num_price_target_analysts: int | None = None
    current_rating: str | None = None    # "Buy" / "Hold" / "Sell" / etc.
    estimates: list[AnalystEstimate] = Field(default_factory=list)
    # Revision momentum: positive = net upgrades, negative = net downgrades
    revision_direction: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalised [-1,+1] revision momentum score",
    )

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


# ---------------------------------------------------------------------------
# Processing layer models
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """
    One chunk stored in ChromaDB.

    The `metadata` dict is what ChromaDB indexes for filtered retrieval.
    Keep every field here flat and JSON-serialisable.
    """

    chunk_id: str = Field(..., description="Unique UUID for this chunk")
    text: str = Field(..., min_length=1)

    # Core identifiers
    ticker: str
    filing_type: FilingType
    section_type: SectionType
    quarter: Quarter
    year: int
    filed_date: str   # ISO date string for ChromaDB compatibility

    # Chunk position
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=1)

    # Content signals
    word_count: int = Field(ge=1)
    contains_numbers: bool = False
    speaker: str = Field(default="")   # CEO / CFO / Analyst / ""
    chunking_strategy: ChunkingStrategy

    # Source traceability
    source_url: str = Field(default="")
    accession_number: str = Field(default="")

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Return a flat dict suitable for ChromaDB metadata storage."""
        return {
            "ticker": self.ticker,
            "filing_type": self.filing_type.value,
            "section_type": self.section_type.value,
            "quarter": self.quarter.value,
            "year": self.year,
            "filed_date": self.filed_date,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "word_count": self.word_count,
            "contains_numbers": self.contains_numbers,
            "speaker": self.speaker,
            "chunking_strategy": self.chunking_strategy.value,
            "source_url": self.source_url,
            "accession_number": self.accession_number,
        }


class GuidanceEntity(BaseModel):
    """A single forward-looking numerical guidance statement extracted by NER."""

    metric_type: GuidanceMetric
    value_low: float | None = None
    value_high: float | None = None
    unit: str = Field(default="USD", description="USD / % / bps / shares")
    fiscal_period: str = Field(default="", description="e.g. 'Q2 2024' or 'FY2024'")
    raw_text: str = Field(default="", description="Original sentence containing guidance")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_explicit: bool = Field(
        default=True,
        description="True = explicitly stated number; False = implied range",
    )


# ---------------------------------------------------------------------------
# Analysis layer models
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    """FinBERT output for a single chunk."""

    chunk_id: str
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)

    @property
    def net_score(self) -> float:
        """Net sentiment in [-1, +1]: positive - negative."""
        return self.positive - self.negative

    @property
    def dominant_label(self) -> str:
        scores = {"positive": self.positive, "negative": self.negative, "neutral": self.neutral}
        return max(scores, key=lambda k: scores[k])


class QuarterSentiment(BaseModel):
    """Aggregated FinBERT sentiment for a ticker/quarter combination."""

    ticker: str
    quarter: Quarter
    year: int
    positive_pct: float = Field(ge=0.0, le=1.0)
    negative_pct: float = Field(ge=0.0, le=1.0)
    neutral_pct: float = Field(ge=0.0, le=1.0)
    net_sentiment_score: float = Field(ge=-1.0, le=1.0)
    hedging_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Normalised frequency of hedging words",
    )
    specificity_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of concrete numbers to vague language",
    )
    chunk_count: int = Field(ge=0)
    most_negative_chunks: list[str] = Field(default_factory=list)


class ToneDriftReport(BaseModel):
    """Quarter-over-quarter tone drift analysis for one ticker."""

    ticker: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    quarters_analysed: list[str] = Field(default_factory=list)

    # Time-series of net sentiment score per quarter
    sentiment_trend: list[float] = Field(default_factory=list)
    # Time-series of hedging language score per quarter
    hedging_trend: list[float] = Field(default_factory=list)
    # Time-series of specificity score per quarter
    specificity_trend: list[float] = Field(default_factory=list)

    # Quarter-over-quarter deltas
    drift_magnitudes: list[float] = Field(default_factory=list)
    drift_directions: list[DriftDirection] = Field(default_factory=list)

    alert_level: AlertLevel = AlertLevel.GREEN
    alert_reason: str = Field(default="")

    consecutive_deterioration_count: int = Field(default=0, ge=0)


class Contradiction(BaseModel):
    """NLI-detected contradiction between two document chunks."""

    chunk_a_id: str
    chunk_b_id: str
    chunk_a_text: str
    chunk_b_text: str
    chunk_a_source: str   # e.g. "AAPL 10-Q Q2 2023 — MD&A"
    chunk_b_source: str
    contradiction_score: float = Field(ge=0.0, le=1.0)
    entailment_score: float = Field(ge=0.0, le=1.0)
    neutral_score: float = Field(ge=0.0, le=1.0)
    interpretation: str = Field(
        default="",
        description="LLM-generated plain-English explanation of the conflict",
    )
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class GuidanceRecord(BaseModel):
    """Guidance entity paired with the subsequently reported actual."""

    ticker: str
    quarter: Quarter
    year: int
    metric_type: GuidanceMetric
    guided_low: float | None = None
    guided_high: float | None = None
    guided_mid: float | None = None  # average of low/high
    actual_value: float | None = None
    accuracy_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="1 - abs(guided_mid - actual) / abs(actual)",
    )
    guidance_bias: float | None = Field(
        default=None,
        description="Positive = guided too high (optimistic), negative = sandbagging",
    )
    filing_source: str = Field(default="")


class EarningsQualityScore(BaseModel):
    """Composite earnings quality signal for one ticker/quarter."""

    ticker: str
    quarter: Quarter
    year: int
    composite_score: float = Field(ge=-1.0, le=1.0)

    # Component scores (each normalised to [-1, +1])
    sentiment_drift_component: float = Field(ge=-1.0, le=1.0)
    guidance_accuracy_component: float = Field(ge=-1.0, le=1.0)
    accruals_component: float = Field(ge=-1.0, le=1.0)
    analyst_revision_component: float = Field(ge=-1.0, le=1.0)

    # Weights used (pulled from settings at compute time)
    weight_sentiment_drift: float
    weight_guidance_accuracy: float
    weight_accruals: float
    weight_analyst_revision: float

    accruals_ratio: float | None = Field(
        default=None,
        description="(Net Income - OCF) / Total Assets",
    )
    computed_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def signal(self) -> SignalDirection:
        from config.settings import settings
        if self.composite_score > settings.SIGNAL_LONG_THRESHOLD:
            return SignalDirection.LONG
        if self.composite_score < settings.SIGNAL_SHORT_THRESHOLD:
            return SignalDirection.SHORT
        return SignalDirection.NEUTRAL


# ---------------------------------------------------------------------------
# Backtest layer models
# ---------------------------------------------------------------------------


class BacktestSignal(BaseModel):
    """A trade signal derived from EarningsQualityScore."""

    ticker: str
    signal_date: date          # earnings announcement date
    entry_date: date           # next trading day
    direction: SignalDirection
    quality_score: float = Field(ge=-1.0, le=1.0)
    holding_days: int = Field(ge=1)


class TradeResult(BaseModel):
    """Realised outcome of one backtest position."""

    ticker: str
    direction: SignalDirection
    signal_date: date
    entry_date: date
    exit_date: date
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    quality_score: float
    gross_return: float          # (exit - entry) / entry, sign-adjusted for shorts
    holding_days_actual: int

    @property
    def is_winner(self) -> bool:
        return self.gross_return > 0.0


# ---------------------------------------------------------------------------
# Evaluation layer models
# ---------------------------------------------------------------------------


class RAGEvaluation(BaseModel):
    """RAGAS metric bundle for a single RAG query-response pair."""

    query: str
    answer: str
    contexts: list[str]
    ground_truth: str = Field(default="")

    # RAGAS metrics (0-1, higher is better)
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    context_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    context_recall: float | None = Field(default=None, ge=0.0, le=1.0)

    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(default="")
    latency_ms: float = Field(default=0.0, ge=0.0)

    @property
    def mean_score(self) -> float:
        scores = [
            s for s in [
                self.faithfulness,
                self.answer_relevance,
                self.context_precision,
                self.context_recall,
            ]
            if s is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from datetime import date as d

    filing = SECFiling(
        ticker="aapl",
        cik="320193",
        filing_type=FilingType.FORM_10Q,
        period_of_report=d(2024, 6, 30),
        filed_date=d(2024, 8, 2),
        filing_url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm",
        accession_number="0000320193-24-000123",
        raw_text="Sample text for testing purposes.",
        xbrl_data=XBRLData(eps_diluted=1.53, revenue=85_777_000_000),
        quarter=Quarter.Q3,
    )

    chunk = DocumentChunk(
        chunk_id="abc-123",
        text="We expect revenue between $85B and $90B for the next quarter.",
        ticker="AAPL",
        filing_type=FilingType.FORM_10Q,
        section_type=SectionType.GUIDANCE,
        quarter=Quarter.Q3,
        year=2024,
        filed_date="2024-08-02",
        chunk_index=0,
        total_chunks=120,
        word_count=14,
        contains_numbers=True,
        chunking_strategy=ChunkingStrategy.SECTION_AWARE,
    )

    score = EarningsQualityScore(
        ticker="AAPL",
        quarter=Quarter.Q3,
        year=2024,
        composite_score=0.45,
        sentiment_drift_component=0.6,
        guidance_accuracy_component=0.5,
        accruals_component=0.3,
        analyst_revision_component=0.4,
        weight_sentiment_drift=0.30,
        weight_guidance_accuracy=0.25,
        weight_accruals=0.25,
        weight_analyst_revision=0.20,
    )

    print("SECFiling  :", filing.ticker, filing.cik, filing.quarter)
    print("Chunk meta :", json.dumps(chunk.to_chroma_metadata(), indent=2))
    print("Signal     :", score.signal)
    print("All models OK ✓")
