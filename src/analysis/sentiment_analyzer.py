"""
FinBERT-based sentiment analyser for financial text.

Loads ProsusAI/finbert (pre-trained on financial news and SEC filings) and
scores every DocumentChunk independently.  Results are then aggregated to
the quarter level, weighting each chunk by its word count so that longer,
more substantive chunks contribute more to the composite signal.

Key outputs:
    SentimentResult     — per-chunk scores (positive/negative/neutral)
    QuarterSentiment    — aggregated scores + hedging/specificity signals
                          for one ticker/quarter combination

The FinBERT model is loaded lazily and cached for the lifetime of the
process to avoid repeated cold-start costs during batch processing.

Usage:
    analyzer = SentimentAnalyzer()
    results  = analyzer.score_chunks(chunks)
    quarters = analyzer.aggregate_to_quarters(results, chunks)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import (
    DocumentChunk,
    Quarter,
    QuarterSentiment,
    SentimentResult,
)

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_finbert_pipeline = None


def _get_pipeline() -> Any:
    """Lazy-load the FinBERT HuggingFace pipeline (cached globally)."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline  # type: ignore
            logger.info(f"Loading FinBERT model: {settings.FINBERT_MODEL}")
            _finbert_pipeline = pipeline(
                "text-classification",
                model=settings.FINBERT_MODEL,
                tokenizer=settings.FINBERT_MODEL,
                top_k=None,                # return all three labels
                truncation=True,
                max_length=512,
                device=-1,                 # CPU; set to 0 for GPU
            )
            logger.info("FinBERT loaded ✓")
        except Exception as exc:
            logger.error(f"Failed to load FinBERT: {exc}")
    return _finbert_pipeline


# ---------------------------------------------------------------------------
# Hedging and specificity scorers (used in quarter aggregation)
# ---------------------------------------------------------------------------

_HEDGING_PATTERN = re.compile(
    "|".join(re.escape(w) for w in settings.HEDGING_WORDS),
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(
    r"(?:\$\s*[\d,]+(?:\.\d+)?(?:\s*[BbMmKk](?:illion)?)?|\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*x\b)",
    re.IGNORECASE,
)

_VAGUE_PATTERN = re.compile(
    r"\b(?:some|many|various|certain|several|numerous|significant(?:ly)?|"
    r"substantial(?:ly)?|approximate(?:ly)?|broad(?:ly)?|general(?:ly)?|"
    r"roughly|around|about|near(?:ly)?|potentially|possibly)\b",
    re.IGNORECASE,
)


def _compute_hedging_score(text: str) -> float:
    """
    Compute normalised hedging language frequency.

    Counts hedging words per 100 words.  Values above ~0.15 indicate
    elevated uncertainty language.

    Args:
        text: Input text string.

    Returns:
        Float in [0, ∞) — hedging words per 100 words.
    """
    word_count = max(1, len(text.split()))
    hedge_count = len(_HEDGING_PATTERN.findall(text))
    return hedge_count / word_count * 100


def _compute_specificity_score(text: str) -> float:
    """
    Compute the ratio of concrete numbers to vague language.

    High specificity (near 1.0) means management is giving hard numbers.
    Declining specificity is a warning signal.

    Args:
        text: Input text string.

    Returns:
        Float in [0, 1].
    """
    numbers = len(_NUMBER_PATTERN.findall(text))
    vague = len(_VAGUE_PATTERN.findall(text))
    total = numbers + vague
    if total == 0:
        return 0.5  # neutral when no signal either way
    return numbers / total


# ---------------------------------------------------------------------------
# Label normaliser
# ---------------------------------------------------------------------------

# FinBERT outputs labels: 'positive', 'negative', 'neutral'
# Some model variants use 'POSITIVE', 'LABEL_0' etc. — normalise all.
_LABEL_MAP: dict[str, str] = {
    "positive": "positive", "pos": "positive", "label_2": "positive",
    "negative": "negative", "neg": "negative", "label_0": "negative",
    "neutral": "neutral",   "neu": "neutral",   "label_1": "neutral",
}


def _parse_finbert_output(raw: list[dict[str, Any]]) -> tuple[float, float, float]:
    """
    Parse FinBERT pipeline output into (positive, negative, neutral) floats.

    The pipeline returns a list like:
        [{'label': 'positive', 'score': 0.93}, {'label': 'negative', ...}, ...]

    Args:
        raw: List of label/score dicts from the HF pipeline.

    Returns:
        Tuple (positive, negative, neutral), each in [0, 1], summing to ~1.
    """
    scores: dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in raw:
        label = _LABEL_MAP.get(item["label"].lower(), "neutral")
        scores[label] = float(item["score"])
    return scores["positive"], scores["negative"], scores["neutral"]


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Scores document chunks with FinBERT and aggregates to quarter level.

    Attributes:
        batch_size: Number of chunks sent to the model in one forward pass.
    """

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Chunk-level scoring
    # ------------------------------------------------------------------

    def score_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[SentimentResult]:
        """
        Run FinBERT on every chunk and return per-chunk sentiment scores.

        Processes in batches to avoid OOM on large document sets.  Chunks
        longer than 512 tokens are truncated (FinBERT's context limit).

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            List of SentimentResult objects in the same order as input.
        """
        if not chunks:
            return []

        pipe = _get_pipeline()
        if pipe is None:
            logger.warning("FinBERT unavailable — returning neutral scores")
            return [
                SentimentResult(chunk_id=c.chunk_id, positive=0.0, negative=0.0, neutral=1.0)
                for c in chunks
            ]

        texts = [c.text for c in chunks]
        results: list[SentimentResult] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start: batch_start + self.batch_size]
            batch_chunks = chunks[batch_start: batch_start + self.batch_size]

            try:
                raw_outputs = pipe(batch_texts)
                for chunk, raw in zip(batch_chunks, raw_outputs):
                    pos, neg, neu = _parse_finbert_output(raw)
                    results.append(SentimentResult(
                        chunk_id=chunk.chunk_id,
                        positive=pos,
                        negative=neg,
                        neutral=neu,
                    ))
            except Exception as exc:
                logger.error(f"FinBERT batch failed (start={batch_start}): {exc}")
                for chunk in batch_chunks:
                    results.append(SentimentResult(
                        chunk_id=chunk.chunk_id,
                        positive=0.0, negative=0.0, neutral=1.0,
                    ))

            logger.debug(
                f"Scored {min(batch_start + self.batch_size, len(texts))}/{len(texts)} chunks"
            )

        return results

    def score_text(self, text: str) -> SentimentResult:
        """
        Score a single arbitrary text string.

        Convenience wrapper around score_chunks for one-off queries.

        Args:
            text: Any financial text.

        Returns:
            SentimentResult with a synthetic chunk_id.
        """
        from src.processing.chunker import _make_id
        dummy = DocumentChunk(
            chunk_id=_make_id(),
            text=text,
            ticker="__single__",
            filing_type="10-Q",  # type: ignore
            section_type="unknown",  # type: ignore
            quarter=Quarter.Q1,
            year=2024,
            filed_date="2024-01-01",
            chunk_index=0,
            total_chunks=1,
            word_count=len(text.split()),
            chunking_strategy="fixed_overlap",  # type: ignore
        )
        results = self.score_chunks([dummy])
        return results[0]

    # ------------------------------------------------------------------
    # Quarter-level aggregation
    # ------------------------------------------------------------------

    def aggregate_to_quarters(
        self,
        sentiment_results: list[SentimentResult],
        chunks: list[DocumentChunk],
        most_negative_k: int = 5,
    ) -> dict[tuple[str, Quarter, int], QuarterSentiment]:
        """
        Aggregate chunk-level FinBERT scores to ticker/quarter/year buckets.

        Weighting: each chunk's contribution is proportional to its word
        count so short, low-information chunks don't dilute signal from
        longer substantive passages.

        Args:
            sentiment_results: Output from score_chunks().
            chunks:            The same chunks passed to score_chunks().
            most_negative_k:   Number of most-negative chunk texts to retain.

        Returns:
            Dict keyed by (ticker, quarter, year) → QuarterSentiment.
        """
        # Build a lookup: chunk_id → (SentimentResult, DocumentChunk)
        id_to_result: dict[str, SentimentResult] = {r.chunk_id: r for r in sentiment_results}
        id_to_chunk: dict[str, DocumentChunk] = {c.chunk_id: c for c in chunks}

        # Group by (ticker, quarter, year)
        groups: dict[
            tuple[str, Quarter, int],
            list[tuple[DocumentChunk, SentimentResult]],
        ] = defaultdict(list)

        for chunk in chunks:
            result = id_to_result.get(chunk.chunk_id)
            if result:
                key = (chunk.ticker, chunk.quarter, chunk.year)
                groups[key].append((chunk, result))

        quarter_sentiments: dict[tuple[str, Quarter, int], QuarterSentiment] = {}

        for (ticker, quarter, year), pairs in groups.items():
            total_words = sum(c.word_count for c, _ in pairs)
            if total_words == 0:
                continue

            # Weighted averages
            w_pos = sum(c.word_count * r.positive for c, r in pairs) / total_words
            w_neg = sum(c.word_count * r.negative for c, r in pairs) / total_words
            w_neu = sum(c.word_count * r.neutral  for c, r in pairs) / total_words

            net_score = float(np.clip(w_pos - w_neg, -1.0, 1.0))

            # Hedging and specificity across all chunk texts
            all_text = " ".join(c.text for c, _ in pairs)
            hedging = _compute_hedging_score(all_text)
            specificity = _compute_specificity_score(all_text)

            # Most negative chunks (by net score)
            sorted_by_neg = sorted(pairs, key=lambda p: p[1].net_score)
            most_neg = [c.text[:200] for c, _ in sorted_by_neg[:most_negative_k]]

            quarter_sentiments[(ticker, quarter, year)] = QuarterSentiment(
                ticker=ticker,
                quarter=quarter,
                year=year,
                positive_pct=float(np.clip(w_pos, 0.0, 1.0)),
                negative_pct=float(np.clip(w_neg, 0.0, 1.0)),
                neutral_pct=float(np.clip(w_neu, 0.0, 1.0)),
                net_sentiment_score=net_score,
                hedging_score=float(hedging),
                specificity_score=float(np.clip(specificity, 0.0, 1.0)),
                chunk_count=len(pairs),
                most_negative_chunks=most_neg,
            )

            logger.debug(
                f"[{ticker}] {quarter.value} {year}: "
                f"net={net_score:+.3f}  pos={w_pos:.2%}  neg={w_neg:.2%}  "
                f"hedge={hedging:.2f}  chunks={len(pairs)}"
            )

        return quarter_sentiments

    # ------------------------------------------------------------------
    # Convenience: full pipeline for one ticker
    # ------------------------------------------------------------------

    def analyse_ticker(
        self,
        ticker: str,
        chunks: list[DocumentChunk],
    ) -> dict[tuple[str, Quarter, int], QuarterSentiment]:
        """
        Run the complete sentiment pipeline for one ticker's chunks.

        Filters to the given ticker, scores, and aggregates.

        Args:
            ticker: Stock ticker (upper case).
            chunks: All chunks — will be filtered to this ticker.

        Returns:
            Dict of QuarterSentiment objects keyed by (ticker, quarter, year).
        """
        ticker_chunks = [c for c in chunks if c.ticker == ticker.upper()]
        if not ticker_chunks:
            logger.warning(f"[{ticker}] No chunks found for sentiment analysis")
            return {}

        logger.info(f"[{ticker}] Scoring {len(ticker_chunks)} chunks with FinBERT")
        results = self.score_chunks(ticker_chunks)
        return self.aggregate_to_quarters(results, ticker_chunks)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

sentiment_analyzer = SentimentAnalyzer()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test aggregation logic with mock sentiment results (no model needed)
    from datetime import date as d
    from src.ingestion.data_validator import ChunkingStrategy, FilingType, SectionType

    def _make_chunk(cid: str, text: str, quarter: Quarter, year: int) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=cid, text=text, ticker="AAPL",
            filing_type=FilingType.FORM_10Q,
            section_type=SectionType.MDA, quarter=quarter, year=year,
            filed_date=f"{year}-08-01", chunk_index=0, total_chunks=5,
            word_count=len(text.split()), contains_numbers=True,
            chunking_strategy=ChunkingStrategy.SECTION_AWARE,
        )

    chunks = [
        _make_chunk("c1", "Revenue grew 8% to record $94.9B. Strong iPhone demand.", Quarter.Q3, 2024),
        _make_chunk("c2", "Services hit all-time high of $24.2B, up 14%.", Quarter.Q3, 2024),
        _make_chunk("c3", "Supply chain challenges may pressure margins in Q4.", Quarter.Q3, 2024),
        _make_chunk("c4", "Revenue declined 3% to $117.2B amid challenging demand.", Quarter.Q1, 2024),
        _make_chunk("c5", "We anticipate growth in the second half, approximately.", Quarter.Q1, 2024),
    ]

    # Mock sentiment results (bypassing FinBERT)
    mock_results = [
        SentimentResult(chunk_id="c1", positive=0.85, negative=0.05, neutral=0.10),
        SentimentResult(chunk_id="c2", positive=0.78, negative=0.08, neutral=0.14),
        SentimentResult(chunk_id="c3", positive=0.10, negative=0.72, neutral=0.18),
        SentimentResult(chunk_id="c4", positive=0.05, negative=0.88, neutral=0.07),
        SentimentResult(chunk_id="c5", positive=0.35, negative=0.20, neutral=0.45),
    ]

    analyzer = SentimentAnalyzer()
    quarters = analyzer.aggregate_to_quarters(mock_results, chunks)

    print("=== Quarter Sentiment Aggregation ===")
    for (ticker, q, yr), qs in sorted(quarters.items(), key=lambda x: x[0][2]):
        print(f"\n{ticker} {q.value} {yr}:")
        print(f"  net_score     = {qs.net_sentiment_score:+.3f}")
        print(f"  positive_pct  = {qs.positive_pct:.2%}")
        print(f"  negative_pct  = {qs.negative_pct:.2%}")
        print(f"  hedging_score = {qs.hedging_score:.3f}")
        print(f"  specificity   = {qs.specificity_score:.3f}")
        print(f"  chunks        = {qs.chunk_count}")

    assert len(quarters) == 2, f"Expected 2 quarters, got {len(quarters)}"
    q3_key = ("AAPL", Quarter.Q3, 2024)
    q1_key = ("AAPL", Quarter.Q1, 2024)
    assert quarters[q3_key].net_sentiment_score > quarters[q1_key].net_sentiment_score, \
        "Q3 should be more positive than Q1"
    print("\nAll sentiment aggregation assertions passed ✓")
    print("sentiment_analyzer smoke test passed ✓")
