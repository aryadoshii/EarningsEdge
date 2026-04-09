"""
MLflow experiment tracker for EarningsEdge pipeline runs.

Logs to a local MLflow tracking server at settings.MLFLOW_TRACKING_URI.
Each pipeline run is a separate MLflow run under the 'earningsedge' experiment.

What gets logged:
    Parameters  — ticker, quarter, year, model, chunking settings
    Metrics     — quality score, RAGAS scores, backtest metrics, latency
    Artifacts   — trade log CSV, equity curve plot, contradictions JSON

Usage:
    tracker = MLflowTracker()
    with tracker.start_run(ticker="AAPL", quarter="Q3", year=2024) as run_id:
        tracker.log_quality_score(score)
        tracker.log_ragas_evaluation(eval_result)
        tracker.log_backtest_metrics(metrics)
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import EarningsQualityScore, RAGEvaluation

# MLflow lazy import
_mlflow_available = False
try:
    import mlflow                       # type: ignore
    import mlflow.tracking              # type: ignore
    _mlflow_available = True
except ImportError:
    logger.warning("mlflow not installed — experiment tracking disabled")

_EXPERIMENT_NAME = "earningsedge"


class MLflowTracker:
    """
    Wraps MLflow calls with graceful no-op fallback when MLflow is absent.

    All public methods are safe to call even if mlflow is not installed.
    """

    def __init__(self) -> None:
        if _mlflow_available:
            mlflow.set_tracking_uri(str(settings.mlflow_path))
            mlflow.set_experiment(_EXPERIMENT_NAME)
            logger.info(
                f"MLflow tracking: {settings.mlflow_path}  "
                f"experiment={_EXPERIMENT_NAME}"
            )
        self._active_run_id: str | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        ticker: str,
        quarter: str,
        year: int,
        tags: dict[str, str] | None = None,
    ) -> Generator[str | None, None, None]:
        """
        Start an MLflow run as a context manager.

        Logs standard pipeline parameters on entry and ends the run on exit.

        Args:
            ticker:  Stock ticker.
            quarter: Reporting quarter e.g. "Q3".
            year:    Reporting year.
            tags:    Optional additional MLflow tags.

        Yields:
            MLflow run ID string, or None if MLflow unavailable.
        """
        if not _mlflow_available:
            yield None
            return

        run_name = f"{ticker}_{quarter}_{year}"
        with mlflow.start_run(run_name=run_name) as run:
            self._active_run_id = run.info.run_id
            # Log standard params
            mlflow.log_params({
                "ticker":            ticker,
                "quarter":           quarter,
                "year":              str(year),
                "groq_model":        settings.GROQ_MODEL,
                "embedding_model":   settings.EMBEDDING_MODEL,
                "chunk_size":        settings.CHUNK_SIZE,
                "top_k_retrieval":   settings.TOP_K_RETRIEVAL,
                "max_rag_hops":      settings.MAX_RAG_HOPS,
            })
            if tags:
                mlflow.set_tags(tags)
            logger.info(f"MLflow run started: {run_name} (id={run.info.run_id[:8]})")
            try:
                yield run.info.run_id
            finally:
                self._active_run_id = None
                logger.info(f"MLflow run ended: {run_name}")

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_quality_score(self, score: EarningsQualityScore) -> None:
        """
        Log composite score and all components to MLflow.

        Args:
            score: Computed EarningsQualityScore.
        """
        if not _mlflow_available:
            return
        try:
            mlflow.log_metrics({
                "quality_score_composite":           score.composite_score,
                "quality_score_sentiment_drift":     score.sentiment_drift_component,
                "quality_score_guidance_accuracy":   score.guidance_accuracy_component,
                "quality_score_accruals":            score.accruals_component,
                "quality_score_analyst_revision":    score.analyst_revision_component,
            })
            mlflow.log_param("signal", score.signal.value)
            if score.accruals_ratio is not None:
                mlflow.log_metric("accruals_ratio", score.accruals_ratio)
        except Exception as exc:
            logger.warning(f"MLflow quality score logging failed: {exc}")

    def log_ragas_evaluation(self, evaluation: RAGEvaluation) -> None:
        """
        Log RAGAS metric scores.

        Args:
            evaluation: Computed RAGEvaluation object.
        """
        if not _mlflow_available:
            return
        try:
            metrics: dict[str, float] = {"ragas_mean_score": evaluation.mean_score}
            if evaluation.faithfulness is not None:
                metrics["ragas_faithfulness"]      = evaluation.faithfulness
            if evaluation.answer_relevance is not None:
                metrics["ragas_answer_relevance"]  = evaluation.answer_relevance
            if evaluation.context_precision is not None:
                metrics["ragas_context_precision"] = evaluation.context_precision
            if evaluation.context_recall is not None:
                metrics["ragas_context_recall"]    = evaluation.context_recall
            metrics["rag_latency_ms"] = evaluation.latency_ms
            mlflow.log_metrics(metrics)
        except Exception as exc:
            logger.warning(f"MLflow RAGAS logging failed: {exc}")

    def log_backtest_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log backtest performance metrics.

        Args:
            metrics: Output of compute_all_metrics() — nested dict.
        """
        if not _mlflow_available:
            return
        try:
            flat: dict[str, float] = {}
            for slice_name, slice_metrics in metrics.items():
                if isinstance(slice_metrics, dict):
                    for k, v in slice_metrics.items():
                        if isinstance(v, (int, float)):
                            flat[f"backtest_{slice_name}_{k}"] = float(v)
            mlflow.log_metrics(flat)
        except Exception as exc:
            logger.warning(f"MLflow backtest metrics logging failed: {exc}")

    def log_tone_drift(self, drift_report: dict[str, Any]) -> None:
        """
        Log tone drift summary metrics.

        Args:
            drift_report: Serialised ToneDriftReport dict.
        """
        if not _mlflow_available:
            return
        try:
            mlflow.log_params({
                "drift_alert_level":   drift_report.get("alert_level", "GREEN"),
                "consecutive_det":     str(drift_report.get("consecutive_deterioration_count", 0)),
            })
            sentiment_trend = drift_report.get("sentiment_trend", [])
            if sentiment_trend:
                mlflow.log_metric("drift_latest_sentiment", sentiment_trend[-1])
                mlflow.log_metric("drift_sentiment_range",
                                  max(sentiment_trend) - min(sentiment_trend))
        except Exception as exc:
            logger.warning(f"MLflow drift logging failed: {exc}")

    def log_contradiction_count(self, count: int) -> None:
        if _mlflow_available:
            try:
                mlflow.log_metric("contradiction_count", float(count))
            except Exception:
                pass

    def log_pipeline_latency(self, latency_ms: float) -> None:
        if _mlflow_available:
            try:
                mlflow.log_metric("pipeline_latency_ms", latency_ms)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Artifact logging
    # ------------------------------------------------------------------

    def log_trade_log(self, df: pd.DataFrame, filename: str = "trade_log.csv") -> None:
        """
        Log the backtest trade log CSV as an MLflow artifact.

        Args:
            df:       Trade log DataFrame from Backtester.to_dataframe().
            filename: Artifact filename.
        """
        if not _mlflow_available or df.empty:
            return
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / filename
                df.to_csv(path, index=False)
                mlflow.log_artifact(str(path))
        except Exception as exc:
            logger.warning(f"MLflow trade log artifact failed: {exc}")

    def log_contradictions(
        self,
        contradictions: list[dict[str, Any]],
        filename: str = "contradictions.json",
    ) -> None:
        """
        Log detected contradictions as a JSON artifact.

        Args:
            contradictions: List of contradiction dicts.
            filename:       Artifact filename.
        """
        if not _mlflow_available or not contradictions:
            return
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / filename
                path.write_text(json.dumps(contradictions, indent=2, default=str))
                mlflow.log_artifact(str(path))
        except Exception as exc:
            logger.warning(f"MLflow contradictions artifact failed: {exc}")

    # ------------------------------------------------------------------
    # Run comparison
    # ------------------------------------------------------------------

    def get_recent_runs(
        self,
        n: int = 10,
        ticker: str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve recent MLflow runs for the dashboard comparison table.

        Args:
            n:      Maximum number of runs to return.
            ticker: Optional filter to one ticker.

        Returns:
            DataFrame with run metadata and key metrics.
        """
        if not _mlflow_available:
            return pd.DataFrame()
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
            if experiment is None:
                return pd.DataFrame()

            filter_str = f"params.ticker = '{ticker}'" if ticker else ""
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_str,
                order_by=["start_time DESC"],
                max_results=n,
            )

            rows = []
            for run in runs:
                row = {
                    "run_id":    run.info.run_id[:8],
                    "ticker":    run.data.params.get("ticker", ""),
                    "quarter":   run.data.params.get("quarter", ""),
                    "year":      run.data.params.get("year", ""),
                    "signal":    run.data.params.get("signal", ""),
                    "composite": run.data.metrics.get("quality_score_composite", None),
                    "ragas_mean": run.data.metrics.get("ragas_mean_score", None),
                    "sharpe":    run.data.metrics.get("backtest_overall_sharpe", None),
                }
                rows.append(row)
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning(f"MLflow run retrieval failed: {exc}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

mlflow_tracker = MLflowTracker()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime

    tracker = MLflowTracker()

    # Test that all methods are no-ops when mlflow is unavailable
    print(f"MLflow available: {_mlflow_available}")

    # Test log_backtest_metrics with nested dict
    metrics = {
        "overall": {"sharpe": 1.25, "hit_rate": 0.62, "annualised_return": 0.18},
        "long":    {"sharpe": 1.40, "hit_rate": 0.68, "annualised_return": 0.22},
    }

    if _mlflow_available:
        with tracker.start_run("TEST", "Q3", 2024) as run_id:
            tracker.log_backtest_metrics(metrics)
            print(f"Logged to MLflow run: {run_id[:8] if run_id else 'N/A'}")
    else:
        # Verify no-op behaviour
        tracker.log_backtest_metrics(metrics)
        tracker.log_pipeline_latency(1234.5)
        tracker.log_contradiction_count(3)
        print("No-op fallback: all methods called without error ✓")

    recent = tracker.get_recent_runs(n=5)
    print(f"get_recent_runs: {len(recent)} rows returned ✓")
    print("\nmlflow_tracker smoke test passed ✓")
