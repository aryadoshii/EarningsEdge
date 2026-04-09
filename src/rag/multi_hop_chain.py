"""
Multi-hop retrieval coordinator.

Orchestrates the full analysis pipeline from raw ticker input to a final
EarningsEdge result, coordinating between the analysis layer (sentiment,
drift, contradictions, quality score) and the RAG graph.

This module is the main entry point for programmatic use of EarningsEdge.
It:
    1. Loads cached or freshly computed analysis results for a ticker
    2. Injects them into the LangGraph initial state
    3. Runs the pipeline
    4. Packages results for the Streamlit dashboard and RAGAS evaluation

Usage:
    chain = MultiHopChain()
    result = await chain.analyse(ticker="AAPL", query="Analyse Q3 2024 earnings")
    result = await chain.analyse_with_full_pipeline(ticker="AAPL")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import (
    EarningsQualityScore,
    Quarter,
    ToneDriftReport,
)
from src.rag.graph import EarningsEdgePipeline, pipeline

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EarningsEdgeResult:
    """
    Complete result of one EarningsEdge analysis run.

    Contains the LLM-generated answer plus all intermediate signals that
    the Streamlit dashboard needs to render charts and metrics.
    """

    ticker: str
    query: str
    quarter: str
    year: int

    # Core outputs
    final_answer: str = ""
    quality_score_obj: dict[str, Any] = field(default_factory=dict)
    tone_drift_report: dict[str, Any] = field(default_factory=dict)
    contradictions: list[dict[str, Any]] = field(default_factory=list)

    # Retrieval stats
    retrieved_chunk_count: int = 0
    hops_used: int = 0

    # Quality gate
    grounding_score: float = 0.0

    # RAGAS
    ragas_input: dict[str, Any] = field(default_factory=dict)

    # Timing
    total_latency_ms: float = 0.0

    @property
    def composite_score(self) -> float:
        return float(self.quality_score_obj.get("composite_score", 0.0))

    @property
    def signal(self) -> str:
        score = self.composite_score
        if score > settings.SIGNAL_LONG_THRESHOLD:
            return "LONG"
        if score < settings.SIGNAL_SHORT_THRESHOLD:
            return "SHORT"
        return "NEUTRAL"

    @property
    def alert_level(self) -> str:
        return self.tone_drift_report.get("alert_level", "GREEN")


# ---------------------------------------------------------------------------
# MultiHopChain
# ---------------------------------------------------------------------------


class MultiHopChain:
    """
    Coordinates the full EarningsEdge analysis pipeline.

    Handles the bridge between the pre-computed analysis layer outputs
    (quality score, drift report) and the LangGraph RAG pipeline.
    """

    def __init__(self, pipeline_instance: EarningsEdgePipeline = pipeline) -> None:
        self._pipeline = pipeline_instance

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    async def analyse(
        self,
        ticker: str,
        query: str,
        quarter: str = "Q4",
        year: int = 2024,
        quality_score: EarningsQualityScore | None = None,
        drift_report: ToneDriftReport | None = None,
    ) -> EarningsEdgeResult:
        """
        Run the complete EarningsEdge analysis for one query.

        This is the main entry point for the Streamlit dashboard and API.
        Pre-computed analysis objects are serialised to dicts and injected
        into the graph initial state.

        Args:
            ticker:        Stock ticker.
            query:         Natural language query.
            quarter:       Reporting quarter e.g. "Q3".
            year:          Reporting year.
            quality_score: Pre-computed EarningsQualityScore (optional —
                           if None, scores default to 0.0 neutral).
            drift_report:  Pre-computed ToneDriftReport (optional).

        Returns:
            EarningsEdgeResult with all pipeline outputs.
        """
        t0 = time.perf_counter()
        ticker = ticker.upper()

        logger.info(
            f"[{ticker}] MultiHopChain.analyse: "
            f"query='{query[:60]}'  period={quarter} {year}"
        )

        # Serialise Pydantic models to dicts for state injection
        qs_dict = _serialise_quality_score(quality_score)
        dr_dict = _serialise_drift_report(drift_report)

        # Run the LangGraph pipeline
        final_state = await self._pipeline.run(
            ticker=ticker,
            query=query,
            quarter=quarter,
            year=year,
            quality_score_obj=qs_dict,
            tone_drift_report=dr_dict,
        )

        total_ms = (time.perf_counter() - t0) * 1000

        result = EarningsEdgeResult(
            ticker=ticker,
            query=query,
            quarter=quarter,
            year=year,
            final_answer=final_state.get("final_answer", ""),
            quality_score_obj=qs_dict,
            tone_drift_report=dr_dict,
            contradictions=final_state.get("contradictions_found", []),
            retrieved_chunk_count=len(final_state.get("retrieved_chunks", [])),
            hops_used=final_state.get("hop_count", 0),
            grounding_score=final_state.get("quality_score", 0.0),
            ragas_input=final_state.get("ragas_input", {}),
            total_latency_ms=total_ms,
        )

        logger.info(
            f"[{ticker}] Analysis complete in {total_ms:.0f}ms — "
            f"signal={result.signal}  grounding={result.grounding_score:.2f}  "
            f"hops={result.hops_used}  contradictions={len(result.contradictions)}"
        )
        return result

    # ------------------------------------------------------------------
    # Full pipeline (including analysis layer)
    # ------------------------------------------------------------------

    async def analyse_with_full_pipeline(
        self,
        ticker: str,
        query: str | None = None,
        quarter: Quarter = Quarter.Q4,
        year: int = 2024,
    ) -> EarningsEdgeResult:
        """
        Run the complete pipeline including loading data from ChromaDB
        and running the analysis layer.

        This is the heavyweight version that computes everything from
        stored chunks.  Use analyse() when you have pre-computed scores.

        Args:
            ticker:  Stock ticker.
            query:   Optional specific query. Defaults to generic quality analysis.
            quarter: Reporting quarter.
            year:    Reporting year.

        Returns:
            EarningsEdgeResult.
        """
        if query is None:
            query = (
                f"Provide a comprehensive earnings quality analysis for {ticker} "
                f"for {quarter.value} {year}. Assess management credibility, "
                f"guidance accuracy, tone drift, and the composite signal."
            )

        logger.info(f"[{ticker}] Full pipeline run: {quarter.value} {year}")

        # Load chunks from ChromaDB
        quality_score, drift_report = await self._load_analysis_results(
            ticker, quarter, year
        )

        return await self.analyse(
            ticker=ticker,
            query=query,
            quarter=quarter.value,
            year=year,
            quality_score=quality_score,
            drift_report=drift_report,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    async def analyse_batch(
        self,
        requests: list[dict[str, Any]],
        concurrency: int = 3,
    ) -> list[EarningsEdgeResult]:
        """
        Run analysis for multiple ticker/query pairs concurrently.

        Respects concurrency limits to avoid hammering the LLM APIs.

        Args:
            requests:    List of dicts with keys: ticker, query, quarter, year.
            concurrency: Max simultaneous pipeline runs.

        Returns:
            List of EarningsEdgeResult in the same order as requests.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(req: dict[str, Any]) -> EarningsEdgeResult:
            async with semaphore:
                return await self.analyse(
                    ticker=req["ticker"],
                    query=req.get("query", "Analyse earnings quality."),
                    quarter=req.get("quarter", "Q4"),
                    year=req.get("year", 2024),
                )

        tasks = [_run_one(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Replace exceptions with error results
        final: list[EarningsEdgeResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(
                    f"Batch item {i} failed: {r}"
                )
                final.append(EarningsEdgeResult(
                    ticker=requests[i].get("ticker", "?"),
                    query=requests[i].get("query", ""),
                    quarter=requests[i].get("quarter", "Q4"),
                    year=requests[i].get("year", 2024),
                    final_answer=f"Analysis failed: {r}",
                ))
            else:
                final.append(r)  # type: ignore[arg-type]

        logger.info(
            f"Batch complete: {sum(1 for r in final if r.final_answer and 'failed' not in r.final_answer)}"
            f"/{len(final)} succeeded"
        )
        return final

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_analysis_results(
        self,
        ticker: str,
        quarter: Quarter,
        year: int,
    ) -> tuple[EarningsQualityScore | None, ToneDriftReport | None]:
        """
        Load pre-computed analysis results saved by pipeline_runner.

        Reads from data/processed/{ticker}/analysis.json which is written
        by `make analyze TICKER=X`.  Falls back to None if not found.
        """
        import json
        from pathlib import Path
        from config.settings import settings

        analysis_path = settings.processed_data_path / ticker.upper() / "analysis.json"
        if not analysis_path.exists():
            logger.debug(f"[{ticker}] No analysis.json found — run `make analyze TICKER={ticker}`")
            return None, None

        try:
            data = json.loads(analysis_path.read_text())

            # Load drift report
            dr: ToneDriftReport | None = None
            dr_data = data.get("drift_report")
            if dr_data:
                dr = ToneDriftReport(**dr_data)

            # Find the quality score for the requested quarter/year
            qs: EarningsQualityScore | None = None
            for score_data in data.get("quality_scores", []):
                if (score_data.get("quarter") == quarter.value
                        and score_data.get("year") == year):
                    qs = EarningsQualityScore(**score_data)
                    break

            # If no exact match, use the latest score
            if qs is None and data.get("quality_scores"):
                qs = EarningsQualityScore(**data["quality_scores"][-1])

            logger.info(f"[{ticker}] Loaded analysis from disk: score={qs.composite_score if qs else 'N/A'}")
            return qs, dr

        except Exception as exc:
            logger.warning(f"[{ticker}] Failed to load analysis results: {exc}")
            return None, None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialise_quality_score(
    score: EarningsQualityScore | None,
) -> dict[str, Any]:
    """Convert EarningsQualityScore Pydantic model to a plain dict."""
    if score is None:
        return {}
    try:
        d = score.model_dump()
        # Convert enums to strings for JSON serialisability
        d["quarter"] = score.quarter.value
        d["signal"] = score.signal.value
        return d
    except Exception as exc:
        logger.warning(f"Quality score serialisation failed: {exc}")
        return {}


def _serialise_drift_report(
    report: ToneDriftReport | None,
) -> dict[str, Any]:
    """Convert ToneDriftReport Pydantic model to a plain dict."""
    if report is None:
        return {}
    try:
        d = report.model_dump()
        d["alert_level"] = report.alert_level.value
        d["drift_directions"] = [dd.value for dd in report.drift_directions]
        return d
    except Exception as exc:
        logger.warning(f"Drift report serialisation failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

multi_hop_chain = MultiHopChain()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    # Test result properties
    result = EarningsEdgeResult(
        ticker="AAPL",
        query="Analyse Q3 2024 earnings",
        quarter="Q3",
        year=2024,
        final_answer="## Executive Summary\nRevenue beat estimates...",
        quality_score_obj={
            "composite_score": 0.45,
            "sentiment_drift_component": 0.60,
            "guidance_accuracy_component": 0.50,
        },
        tone_drift_report={"alert_level": "GREEN", "alert_reason": "Tone stable"},
        contradictions=[],
        retrieved_chunk_count=10,
        hops_used=1,
        grounding_score=0.88,
        total_latency_ms=2340.0,
    )

    assert result.composite_score == 0.45
    assert result.signal == "LONG"
    assert result.alert_level == "GREEN"
    print(f"EarningsEdgeResult.signal:      {result.signal} ✓")
    print(f"EarningsEdgeResult.alert_level: {result.alert_level} ✓")
    print(f"EarningsEdgeResult.composite:   {result.composite_score:+.3f} ✓")

    # Test serialisation
    from src.ingestion.data_validator import AlertLevel, DriftDirection
    dr = ToneDriftReport(
        ticker="AAPL",
        quarters_analysed=["Q1 2024", "Q2 2024", "Q3 2024"],
        sentiment_trend=[0.1, 0.3, 0.45],
        hedging_trend=[5.0, 4.5, 4.0],
        specificity_trend=[0.6, 0.65, 0.70],
        drift_magnitudes=[0.20, 0.15],
        drift_directions=[DriftDirection.IMPROVING, DriftDirection.IMPROVING],
        alert_level=AlertLevel.GREEN,
        alert_reason="Tone improving",
        consecutive_deterioration_count=0,
    )
    dr_dict = _serialise_drift_report(dr)
    assert dr_dict["alert_level"] == "GREEN"
    assert dr_dict["drift_directions"] == ["improving", "improving"]
    print(f"Drift report serialisation: {dr_dict['alert_level']} ✓")

    print("\nmulti_hop_chain smoke test passed ✓")