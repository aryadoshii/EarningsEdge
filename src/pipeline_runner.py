"""
EarningsEdge end-to-end pipeline runner.

Orchestrates the full ingest → process → embed pipeline for a given ticker:

    1. Fetch SEC filings from EDGAR (10-K, 10-Q, 8-K)
    2. Parse filing text and extract sections
    3. Chunk documents with section-aware strategy
    4. Tag chunks with enriched metadata
    5. Save chunks to data/processed/{ticker}/chunks_{form}_{period}.json
    6. Embed chunks and store in ChromaDB

This is what `make embed TICKER=X` and `make analyze TICKER=X` call.

Usage:
    python -m src.pipeline_runner --ticker AAPL --embed
    python -m src.pipeline_runner --ticker AAPL --analyze
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import (
    AnalystData,
    DocumentChunk,
    EarningsQualityScore,
    Quarter,
    SECFiling,
    ToneDriftReport,
)


# ---------------------------------------------------------------------------
# Ingest + Process
# ---------------------------------------------------------------------------

async def run_ingest_pipeline(ticker: str) -> dict[str, Any]:
    """
    Fetch SEC filings, parse, chunk, tag, and save to disk.

    Args:
        ticker: Stock ticker (upper case).

    Returns:
        Dict with keys: filings, chunks, analyst_data
    """
    ticker = ticker.upper()
    logger.info(f"[{ticker}] Starting ingest pipeline")

    # ── 1. Fetch SEC filings ──────────────────────────────────────────
    from src.ingestion.sec_fetcher import SECFetcher
    filings: list[SECFiling] = []
    async with SECFetcher() as sec:
        filings = await sec.fetch_filings(
            ticker,
            form_types=["10-K", "10-Q"],
            max_filings=12,   # ~3 years of quarterly data
        )
    logger.info(f"[{ticker}] Fetched {len(filings)} filings")

    # ── 2. Fetch analyst data ─────────────────────────────────────────
    from src.ingestion.analyst_fetcher import AnalystFetcher
    analyst_data: AnalystData | None = None
    try:
        af = AnalystFetcher()
        analyst_data = await af.fetch(ticker)
        logger.info(f"[{ticker}] Analyst data fetched")
    except Exception as exc:
        logger.warning(f"[{ticker}] Analyst fetch failed: {exc}")

    # ── 3. Parse + chunk each filing ──────────────────────────────────
    from src.processing.document_parser import DocumentParser
    from src.processing.chunker import DocumentChunker
    from src.processing.metadata_tagger import MetadataTagger

    parser  = DocumentParser()
    chunker = DocumentChunker()
    tagger  = MetadataTagger()

    all_chunks: list[DocumentChunk] = []

    for filing in filings:
        try:
            # Parse raw text into sections
            parsed_text = parser.parse_filing(filing)
            filing.raw_text = parsed_text
            filing.word_count = len(parsed_text.split())

            sections = parser.extract_sections(parsed_text)
            if not sections:
                logger.warning(f"[{ticker}] No sections extracted from {filing.filing_type.value} {filing.period_of_report}")
                continue

            # Chunk the filing
            chunks = chunker.chunk_filing(filing, sections)

            # Enrich metadata
            chunks = tagger.tag_batch(chunks)

            all_chunks.extend(chunks)
            logger.info(
                f"[{ticker}] {filing.filing_type.value} {filing.period_of_report}: "
                f"{len(chunks)} chunks"
            )
        except Exception as exc:
            logger.error(f"[{ticker}] Failed to process filing {filing.period_of_report}: {exc}")

    logger.success(f"[{ticker}] Total chunks produced: {len(all_chunks)}")

    # ── 4. Save chunks to disk ────────────────────────────────────────
    _save_chunks(ticker, all_chunks)
    _save_analyst_data(ticker, analyst_data)
    _save_filings_metadata(ticker, filings)

    return {
        "filings":      filings,
        "chunks":       all_chunks,
        "analyst_data": analyst_data,
    }


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

async def run_embed_pipeline(ticker: str) -> int:
    """
    Load saved chunks and embed them into ChromaDB.

    Args:
        ticker: Stock ticker.

    Returns:
        Number of chunks embedded.
    """
    ticker = ticker.upper()
    logger.info(f"[{ticker}] Starting embed pipeline")

    chunks = _load_chunks(ticker)
    if not chunks:
        logger.warning(
            f"[{ticker}] No chunks found. Run `make ingest TICKER={ticker}` first."
        )
        return 0

    from src.embeddings.embedder import Embedder
    embedder = Embedder()
    count = embedder.embed_and_store(chunks)
    logger.success(f"[{ticker}] Embedded {count} chunks into ChromaDB")
    return count


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

async def run_analysis_pipeline(ticker: str) -> dict[str, Any]:
    """
    Run the full analysis layer: sentiment, drift, contradictions, scoring.

    Requires chunks to already be embedded in ChromaDB.

    Args:
        ticker: Stock ticker.

    Returns:
        Dict with analysis results.
    """
    ticker = ticker.upper()
    logger.info(f"[{ticker}] Starting analysis pipeline")

    # Load chunks + filings from disk
    chunks = _load_chunks(ticker)
    filings = _load_filings_metadata(ticker)
    analyst_data = _load_analyst_data(ticker)

    if not chunks:
        logger.error(f"[{ticker}] No chunks available. Run ingest + embed first.")
        return {}

    # ── Sentiment analysis ────────────────────────────────────────────
    from src.analysis.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    sentiment_results = analyzer.score_chunks(chunks)
    quarter_sentiments = analyzer.aggregate_to_quarters(sentiment_results, chunks)
    logger.info(f"[{ticker}] Sentiment: {len(quarter_sentiments)} quarters analysed")

    # ── Tone drift ───────────────────────────────────────────────────
    from src.analysis.tone_drift_detector import ToneDriftDetector
    drift_detector = ToneDriftDetector()
    drift_report = drift_detector.compute_drift(ticker, quarter_sentiments)
    logger.info(f"[{ticker}] Tone drift alert: {drift_report.alert_level.value}")

    # ── Contradiction detection ───────────────────────────────────────
    from src.analysis.contradiction_detector import ContradictionDetector
    contra_detector = ContradictionDetector()
    contradictions = contra_detector.detect(ticker, chunks)
    logger.info(f"[{ticker}] Contradictions found: {len(contradictions)}")

    # ── Guidance accuracy ─────────────────────────────────────────────
    from src.processing.ner_extractor import NERExtractor
    from src.analysis.guidance_accuracy import GuidanceAccuracyTracker

    ner = NERExtractor()
    guidance_by_period: dict[tuple[Quarter, int], list] = {}
    for chunk in chunks:
        if chunk.contains_numbers:
            entities = ner.extract_guidance(chunk.text)
            if entities:
                key = (chunk.quarter, chunk.year)
                guidance_by_period.setdefault(key, []).extend(entities)

    tracker = GuidanceAccuracyTracker()
    guidance_records = tracker.match_guidance_to_actuals(
        guidance_by_period, filings
    )

    # ── Composite quality score ───────────────────────────────────────
    from src.analysis.earnings_quality_scorer import EarningsQualityScorer
    scorer = EarningsQualityScorer()

    # Get latest quarter available
    quarters_available = sorted(
        {(c.quarter, c.year) for c in chunks},
        key=lambda x: (x[1], {"Q1":1,"Q2":2,"Q3":3,"Q4":4}.get(x[0].value,0)),
    )

    scores: list[EarningsQualityScore] = []
    for q, yr in quarters_available:
        filing = next(
            (f for f in filings if f.quarter == q and f.year == yr), None
        )
        score = scorer.compute(
            ticker=ticker, quarter=q, year=yr,
            drift_report=drift_report,
            guidance_records=guidance_records,
            filing=filing,
            analyst_data=analyst_data,
        )
        scores.append(score)
        logger.info(f"[{ticker}] {q.value} {yr}: score={score.composite_score:+.3f} → {score.signal.value}")

    # ── Save results ──────────────────────────────────────────────────
    _save_analysis_results(ticker, scores, drift_report, contradictions)

    return {
        "quality_scores":   scores,
        "drift_report":     drift_report,
        "contradictions":   contradictions,
        "guidance_records": guidance_records,
        "latest_score":     scores[-1] if scores else None,
    }


# ---------------------------------------------------------------------------
# Disk I/O helpers
# ---------------------------------------------------------------------------

def _ticker_dir(ticker: str) -> Path:
    d = settings.processed_data_path / ticker.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_chunks(ticker: str, chunks: list[DocumentChunk]) -> None:
    """Save all chunks for a ticker as a single JSON file."""
    path = _ticker_dir(ticker) / "chunks.json"
    data = [c.model_dump(mode="json") for c in chunks]
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"[{ticker}] Saved {len(chunks)} chunks to {path}")


def _load_chunks(ticker: str) -> list[DocumentChunk]:
    """Load all chunks for a ticker from disk."""
    path = _ticker_dir(ticker) / "chunks.json"
    if not path.exists():
        # Backward compat: try numbered files
        chunk_files = list(_ticker_dir(ticker).glob("chunks_*.json"))
        if not chunk_files:
            return []
        all_raw: list[dict] = []
        for f in chunk_files:
            try:
                all_raw.extend(json.loads(f.read_text()))
            except Exception:
                pass
    else:
        try:
            all_raw = json.loads(path.read_text())
        except Exception as exc:
            logger.error(f"Failed to load chunks: {exc}")
            return []

    chunks: list[DocumentChunk] = []
    for item in all_raw:
        try:
            chunks.append(DocumentChunk(**item))
        except Exception as exc:
            logger.debug(f"Skipping malformed chunk: {exc}")
    logger.info(f"[{ticker}] Loaded {len(chunks)} chunks from disk")
    return chunks


def _save_filings_metadata(ticker: str, filings: list[SECFiling]) -> None:
    path = _ticker_dir(ticker) / "filings.json"
    data = [f.model_dump(mode="json") for f in filings]
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_filings_metadata(ticker: str) -> list[SECFiling]:
    path = _ticker_dir(ticker) / "filings.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
        return [SECFiling(**item) for item in raw]
    except Exception as exc:
        logger.warning(f"Could not load filings metadata: {exc}")
        return []


def _save_analyst_data(ticker: str, data: AnalystData | None) -> None:
    if data is None:
        return
    path = _ticker_dir(ticker) / "analyst.json"
    path.write_text(json.dumps(data.model_dump(mode="json"), indent=2, default=str))


def _load_analyst_data(ticker: str) -> AnalystData | None:
    path = _ticker_dir(ticker) / "analyst.json"
    if not path.exists():
        return None
    try:
        return AnalystData(**json.loads(path.read_text()))
    except Exception:
        return None


def _save_analysis_results(
    ticker: str,
    scores: list[EarningsQualityScore],
    drift_report: ToneDriftReport,
    contradictions: list,
) -> None:
    path = _ticker_dir(ticker) / "analysis.json"
    data = {
        "quality_scores": [s.model_dump(mode="json") for s in scores],
        "drift_report": drift_report.model_dump(mode="json"),
        "contradictions": [c.model_dump(mode="json") for c in contradictions],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"[{ticker}] Analysis results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EarningsEdge pipeline runner")
    parser.add_argument("--ticker",  required=True, help="Stock ticker e.g. AAPL")
    parser.add_argument("--ingest",  action="store_true", help="Run ingest + process")
    parser.add_argument("--embed",   action="store_true", help="Embed chunks into ChromaDB")
    parser.add_argument("--analyze", action="store_true", help="Run analysis pipeline")
    parser.add_argument("--all",     action="store_true", help="Run full pipeline")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    async def _run() -> None:
        if args.all or args.ingest:
            result = await run_ingest_pipeline(ticker)
            logger.success(
                f"[{ticker}] Ingest complete: "
                f"{len(result['filings'])} filings, "
                f"{len(result['chunks'])} chunks"
            )

        if args.all or args.embed:
            count = await run_embed_pipeline(ticker)
            logger.success(f"[{ticker}] Embed complete: {count} chunks in ChromaDB")

        if args.all or args.analyze:
            results = await run_analysis_pipeline(ticker)
            if results.get("latest_score"):
                score = results["latest_score"]
                logger.success(
                    f"[{ticker}] Analysis complete: "
                    f"score={score.composite_score:+.3f} → {score.signal.value}"
                )
                print(f"\n{'='*50}")
                print(f"  {ticker} Earnings Quality Score")
                print(f"{'='*50}")
                from src.analysis.earnings_quality_scorer import EarningsQualityScorer
                print(EarningsQualityScorer().explain(score))

    asyncio.run(_run())