"""
Earnings quality composite scorer.

Combines four independent signals into a single normalised score in [-1, +1]:

    EarningsQualityScore = (
        0.30 * sentiment_drift_component     +   # tone improving = positive
        0.25 * guidance_accuracy_component   +   # accurate guidance = positive
        0.25 * accruals_component            +   # low accruals = higher quality
        0.20 * analyst_revision_component        # upward revisions = positive
    )

All weights are configurable in config/settings.py and must sum to 1.0.
All components are pre-normalised to [-1, +1] before weighting.

Accruals ratio:
    accruals_ratio = (Net Income - Operating Cash Flow) / Total Assets
    Low (negative) accruals = earnings are backed by cash flow = quality.
    High (positive) accruals = earnings are accounting-driven = risk.

    Normalisation: sigmoid-style mapping via:
        accruals_component = -tanh(accruals_ratio * 10)
    so accruals_ratio = 0.0 → component = 0.0
       accruals_ratio = 0.1 → component ≈ -0.76  (high accruals = bad)
       accruals_ratio = -0.1 → component ≈ +0.76 (negative accruals = good)

The output EarningsQualityScore feeds directly into:
    1. BacktestSignal generation (score > 0.3 → LONG, < -0.3 → SHORT)
    2. RAG synthesis node (displayed in the final answer)
    3. Streamlit dashboard gauge

Usage:
    scorer = EarningsQualityScorer()
    score  = scorer.compute(ticker, quarter, year, drift_report,
                            guidance_records, filings, analyst_data)
"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import (
    AnalystData,
    EarningsQualityScore,
    GuidanceRecord,
    Quarter,
    SECFiling,
    ToneDriftReport,
    XBRLData,
)

# ---------------------------------------------------------------------------
# Accruals component
# ---------------------------------------------------------------------------


def _compute_accruals_ratio(xbrl: XBRLData) -> float | None:
    """
    Compute the accruals ratio from XBRL financial data.

    accruals_ratio = (Net Income - Operating Cash Flow) / Total Assets

    Args:
        xbrl: XBRLData populated from an SEC filing.

    Returns:
        Accruals ratio as a float, or None if data is missing.
    """
    ni  = xbrl.net_income
    ocf = xbrl.operating_cash_flow
    ta  = xbrl.total_assets

    if ni is None or ocf is None or ta is None or ta == 0:
        return None

    return (ni - ocf) / ta


def _accruals_to_component(accruals_ratio: float) -> float:
    """
    Map accruals ratio to a [-1, +1] component score via tanh.

    Negative accruals ratio (earnings < cash flow) = positive component.
    Positive accruals ratio (earnings > cash flow) = negative component.

    The factor 10 makes the function responsive in the ±0.15 range
    typical of S&P 500 accruals ratios.

    Args:
        accruals_ratio: Raw accruals ratio (typically -0.15 to +0.15).

    Returns:
        Component score in [-1, +1].
    """
    return float(-math.tanh(accruals_ratio * 10.0))


# ---------------------------------------------------------------------------
# Analyst revision component
# ---------------------------------------------------------------------------


def _analyst_revision_component(analyst_data: AnalystData | None) -> float:
    """
    Extract the analyst revision momentum component from AnalystData.

    The AnalystData.revision_direction field is already normalised to
    [-1, +1] by the analyst_fetcher.  We pass it through with a small
    clamp for safety.

    Args:
        analyst_data: AnalystData object or None if unavailable.

    Returns:
        Component score in [-1, +1].  Returns 0.0 if data is unavailable.
    """
    if analyst_data is None:
        return 0.0
    return float(max(-1.0, min(1.0, analyst_data.revision_direction)))


# ---------------------------------------------------------------------------
# EarningsQualityScorer
# ---------------------------------------------------------------------------


class EarningsQualityScorer:
    """
    Computes the composite EarningsQualityScore for a ticker/quarter pair.

    All weights are drawn from config/settings.py.  The validator in
    Settings ensures they sum to 1.0 at startup.
    """

    def __init__(self) -> None:
        self.w_drift     = settings.SCORE_WEIGHT_SENTIMENT_DRIFT
        self.w_guidance  = settings.SCORE_WEIGHT_GUIDANCE_ACCURACY
        self.w_accruals  = settings.SCORE_WEIGHT_ACCRUALS
        self.w_analyst   = settings.SCORE_WEIGHT_ANALYST_REVISION

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        ticker: str,
        quarter: Quarter,
        year: int,
        drift_report: ToneDriftReport | None,
        guidance_records: list[GuidanceRecord],
        filing: SECFiling | None,
        analyst_data: AnalystData | None,
    ) -> EarningsQualityScore:
        """
        Compute the composite earnings quality score for one quarter.

        Args:
            ticker:           Stock ticker.
            quarter:          Reporting quarter.
            year:             Reporting year.
            drift_report:     ToneDriftReport for this ticker (all quarters).
            guidance_records: GuidanceRecord list for this ticker.
            filing:           SECFiling for this quarter (for XBRL data).
            analyst_data:     Most recent AnalystData for this ticker.

        Returns:
            EarningsQualityScore with composite score + component breakdown.
        """
        # ── Component 1: Sentiment drift ────────────────────────────────
        if drift_report is not None:
            from src.analysis.tone_drift_detector import tone_drift_detector
            drift_comp = tone_drift_detector.get_drift_score(drift_report)
        else:
            drift_comp = 0.0
            logger.debug(f"[{ticker}] No drift report — using neutral drift component")

        # ── Component 2: Guidance accuracy ──────────────────────────────
        from src.analysis.guidance_accuracy import guidance_accuracy_tracker

        # Filter records to this quarter and earlier (no look-ahead)
        period_records = [
            r for r in guidance_records
            if (r.year, _q_ord(r.quarter)) <= (year, _q_ord(quarter))
        ]
        guidance_comp = guidance_accuracy_tracker.compute_component_score(
            period_records, min_records=3
        )

        # ── Component 3: Accruals ────────────────────────────────────────
        accruals_ratio: float | None = None
        if filing is not None:
            accruals_ratio = _compute_accruals_ratio(filing.xbrl_data)

        if accruals_ratio is not None:
            accruals_comp = _accruals_to_component(accruals_ratio)
        else:
            accruals_comp = 0.0
            logger.debug(f"[{ticker}] No XBRL accruals data — using neutral component")

        # ── Component 4: Analyst revision momentum ───────────────────────
        analyst_comp = _analyst_revision_component(analyst_data)

        # ── Composite ────────────────────────────────────────────────────
        composite = (
            self.w_drift    * drift_comp
            + self.w_guidance * guidance_comp
            + self.w_accruals * accruals_comp
            + self.w_analyst  * analyst_comp
        )
        composite = float(max(-1.0, min(1.0, composite)))

        score = EarningsQualityScore(
            ticker=ticker,
            quarter=quarter,
            year=year,
            composite_score=composite,
            sentiment_drift_component=round(drift_comp, 4),
            guidance_accuracy_component=round(guidance_comp, 4),
            accruals_component=round(accruals_comp, 4),
            analyst_revision_component=round(analyst_comp, 4),
            weight_sentiment_drift=self.w_drift,
            weight_guidance_accuracy=self.w_guidance,
            weight_accruals=self.w_accruals,
            weight_analyst_revision=self.w_analyst,
            accruals_ratio=accruals_ratio,
        )

        logger.info(
            f"[{ticker}] {quarter.value} {year} EarningsQualityScore: "
            f"{composite:+.4f}  signal={score.signal.value}  "
            f"[drift={drift_comp:+.3f}  guidance={guidance_comp:+.3f}  "
            f"accruals={accruals_comp:+.3f}  analyst={analyst_comp:+.3f}]"
        )
        return score

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        ticker: str,
        quarters: list[tuple[Quarter, int]],
        drift_report: ToneDriftReport | None,
        guidance_records: list[GuidanceRecord],
        filings_by_period: dict[tuple[Quarter, int], SECFiling],
        analyst_data: AnalystData | None,
    ) -> list[EarningsQualityScore]:
        """
        Compute scores for multiple quarters of the same ticker.

        Args:
            ticker:             Stock ticker.
            quarters:           List of (quarter, year) tuples to score.
            drift_report:       ToneDriftReport for all quarters.
            guidance_records:   All guidance records for the ticker.
            filings_by_period:  {(quarter, year): SECFiling} for XBRL lookup.
            analyst_data:       Most recent analyst consensus.

        Returns:
            List of EarningsQualityScore objects in chronological order.
        """
        scores: list[EarningsQualityScore] = []
        for quarter, year in sorted(quarters, key=lambda x: (x[1], _q_ord(x[0]))):
            filing = filings_by_period.get((quarter, year))
            score = self.compute(
                ticker=ticker,
                quarter=quarter,
                year=year,
                drift_report=drift_report,
                guidance_records=guidance_records,
                filing=filing,
                analyst_data=analyst_data,
            )
            scores.append(score)
        return scores

    # ------------------------------------------------------------------
    # Explainability helper
    # ------------------------------------------------------------------

    def explain(self, score: EarningsQualityScore) -> str:
        """
        Generate a plain-English explanation of the composite score.

        Used by the RAG synthesis node and Streamlit dashboard.

        Args:
            score: Computed EarningsQualityScore.

        Returns:
            Multi-line explanation string.
        """
        lines = [
            f"Earnings Quality Score: {score.composite_score:+.3f} "
            f"→ Signal: {score.signal.value}",
            "",
            "Component Breakdown:",
            f"  Sentiment Drift   ({score.weight_sentiment_drift:.0%} weight): "
            f"{score.sentiment_drift_component:+.3f}  "
            f"{'↑ tone improving' if score.sentiment_drift_component > 0.1 else '↓ tone deteriorating' if score.sentiment_drift_component < -0.1 else '→ stable tone'}",
            f"  Guidance Accuracy ({score.weight_guidance_accuracy:.0%} weight): "
            f"{score.guidance_accuracy_component:+.3f}  "
            f"{'✓ consistently accurate/conservative' if score.guidance_accuracy_component > 0.2 else '✗ optimistic or inaccurate' if score.guidance_accuracy_component < -0.2 else '~ mixed record'}",
            f"  Accruals Quality  ({score.weight_accruals:.0%} weight): "
            f"{score.accruals_component:+.3f}  "
            f"{'✓ cash-backed earnings' if score.accruals_component > 0.2 else '✗ accrual-heavy earnings' if score.accruals_component < -0.2 else '~ neutral accruals'}"
            + (f"  [ratio={score.accruals_ratio:.4f}]" if score.accruals_ratio is not None else ""),
            f"  Analyst Revisions ({score.weight_analyst_revision:.0%} weight): "
            f"{score.analyst_revision_component:+.3f}  "
            f"{'↑ net upgrades' if score.analyst_revision_component > 0.1 else '↓ net downgrades' if score.analyst_revision_component < -0.1 else '→ no revision trend'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Q_ORD: dict[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Annual": 5}


def _q_ord(q: Quarter) -> int:
    return _Q_ORD.get(q.value, 0)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

earnings_quality_scorer = EarningsQualityScorer()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date as d
    from unittest.mock import MagicMock, patch

    scorer = EarningsQualityScorer()

    # ── Accruals component ───────────────────────────────────────────────
    xbrl_good = XBRLData(net_income=10e9, operating_cash_flow=15e9, total_assets=100e9)
    xbrl_bad  = XBRLData(net_income=10e9, operating_cash_flow=3e9,  total_assets=100e9)

    ratio_good = _compute_accruals_ratio(xbrl_good)
    ratio_bad  = _compute_accruals_ratio(xbrl_bad)
    comp_good  = _accruals_to_component(ratio_good)
    comp_bad   = _accruals_to_component(ratio_bad)

    print(f"Good accruals: ratio={ratio_good:.4f}  component={comp_good:+.3f}")
    print(f"Bad  accruals: ratio={ratio_bad:.4f}  component={comp_bad:+.3f}")
    assert comp_good > 0, "OCF > NI should give positive accruals component"
    assert comp_bad  < 0, "NI >> OCF should give negative accruals component"
    assert comp_good > comp_bad
    print("Accruals component ✓")

    # ── tanh normalisation properties ───────────────────────────────────
    assert abs(_accruals_to_component(0.0)) < 1e-9
    assert _accruals_to_component(0.1)  < -0.5
    assert _accruals_to_component(-0.1) >  0.5
    print("tanh normalisation ✓")

    # ── Analyst revision component ───────────────────────────────────────
    analyst = AnalystData(
        ticker="AAPL",
        fetch_date=d(2024, 8, 2),
        revision_direction=0.65,
    )
    assert _analyst_revision_component(analyst) == 0.65
    assert _analyst_revision_component(None) == 0.0
    print("Analyst revision component ✓")

    # ── Full composite score (mock drift and guidance components) ────────
    # Build a minimal mock drift report
    mock_drift = MagicMock()
    mock_drift.sentiment_trend = [0.1, 0.2, 0.35, 0.45]  # improving
    mock_drift.ticker = "AAPL"

    # Patch tone_drift_detector.get_drift_score to return a known value
    with patch(
        "src.analysis.earnings_quality_scorer.tone_drift_detector"
    ) as mock_tdd, patch(
        "src.analysis.earnings_quality_scorer.guidance_accuracy_tracker"
    ) as mock_gat:
        mock_tdd.get_drift_score.return_value = 0.60
        mock_gat.compute_component_score.return_value = 0.50

        filing = SECFiling(
            ticker="AAPL", cik="0000320193",
            filing_type="10-Q",  # type: ignore
            period_of_report=d(2024, 6, 30),
            filed_date=d(2024, 8, 2),
            filing_url="https://sec.gov/test",
            accession_number="test-001",
            xbrl_data=xbrl_good,
            quarter=Quarter.Q3, year=2024,
        )

        score = scorer.compute(
            ticker="AAPL",
            quarter=Quarter.Q3,
            year=2024,
            drift_report=mock_drift,
            guidance_records=[],
            filing=filing,
            analyst_data=analyst,
        )

    print(f"\nComposite score: {score.composite_score:+.4f}  signal={score.signal.value}")
    print(scorer.explain(score))
    assert score.composite_score > 0, "All positive components should yield positive composite"
    assert score.signal.value in ("LONG", "NEUTRAL", "SHORT")

    print("\nearnings_quality_scorer smoke test passed ✓")
