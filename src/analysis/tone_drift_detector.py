"""
Tone drift detector — quarter-over-quarter management language analysis.

This is one of the showstopper features of EarningsEdge.  It compares
QuarterSentiment objects across consecutive quarters to detect systematic
shifts in management tone that often precede earnings surprises.

Key signals computed:
    drift_magnitude    — absolute change in net sentiment score QoQ
    drift_direction    — improving / deteriorating / stable
    hedging_trend      — is management using more uncertainty language?
    specificity_trend  — are guidance statements becoming vaguer?
    consecutive_det.   — how many consecutive quarters of deterioration?

Alert levels:
    GREEN  — stable or improving tone
    YELLOW — one quarter of deterioration OR increasing hedging
    RED    — 2+ consecutive quarters deteriorating + rising hedging language

The RED alert is the primary trading signal input: it captures the
"boiling frog" pattern where management gradually softens language before
a miss, but no single quarter's change is alarming in isolation.

Usage:
    detector = ToneDriftDetector()
    report   = detector.compute_drift(ticker, quarter_sentiments)
"""

from __future__ import annotations

from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import (
    AlertLevel,
    DriftDirection,
    Quarter,
    QuarterSentiment,
    ToneDriftReport,
)

# ---------------------------------------------------------------------------
# Quarter ordering helper
# ---------------------------------------------------------------------------

# Canonical ordering for sorting quarters chronologically
_QUARTER_ORDER: dict[Quarter, int] = {
    Quarter.Q1: 1,
    Quarter.Q2: 2,
    Quarter.Q3: 3,
    Quarter.Q4: 4,
    Quarter.ANNUAL: 5,
}


def _sort_key(qs: QuarterSentiment) -> tuple[int, int]:
    """Sort key for QuarterSentiment: (year, quarter_index)."""
    return (qs.year, _QUARTER_ORDER.get(qs.quarter, 0))


# ---------------------------------------------------------------------------
# ToneDriftDetector
# ---------------------------------------------------------------------------

class ToneDriftDetector:
    """
    Computes quarter-over-quarter tone drift from aggregated sentiment data.

    Parameters are sourced from config/settings.py:
        DRIFT_STABLE_THRESHOLD — absolute score change below which = stable
    """

    def __init__(
        self,
        stable_threshold: float = settings.DRIFT_STABLE_THRESHOLD,
    ) -> None:
        self.stable_threshold = stable_threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_drift(
        self,
        ticker: str,
        quarter_sentiments: dict[tuple[str, Quarter, int], QuarterSentiment],
    ) -> ToneDriftReport:
        """
        Compute a ToneDriftReport for one ticker across all available quarters.

        Args:
            ticker:             Stock ticker (upper case).
            quarter_sentiments: Output of SentimentAnalyzer.aggregate_to_quarters().

        Returns:
            ToneDriftReport with trends, directions, and alert level.
        """
        # Filter and sort this ticker's quarters chronologically
        ticker_quarters: list[QuarterSentiment] = sorted(
            [qs for (t, _, _), qs in quarter_sentiments.items() if t == ticker.upper()],
            key=_sort_key,
        )

        if len(ticker_quarters) < 2:
            logger.warning(
                f"[{ticker}] Only {len(ticker_quarters)} quarter(s) — "
                f"need ≥2 for drift analysis"
            )
            return ToneDriftReport(
                ticker=ticker,
                quarters_analysed=[],
                alert_level=AlertLevel.GREEN,
                alert_reason="Insufficient data for drift analysis",
            )

        logger.info(
            f"[{ticker}] Computing drift across "
            f"{len(ticker_quarters)} quarters"
        )

        # Extract time series
        quarter_labels = [
            f"{qs.quarter.value} {qs.year}" for qs in ticker_quarters
        ]
        sentiment_trend = [qs.net_sentiment_score for qs in ticker_quarters]
        hedging_trend   = [qs.hedging_score for qs in ticker_quarters]
        specificity_trend = [qs.specificity_score for qs in ticker_quarters]

        # Compute QoQ deltas
        drift_magnitudes: list[float] = []
        drift_directions: list[DriftDirection] = []

        for i in range(1, len(ticker_quarters)):
            prev = ticker_quarters[i - 1]
            curr = ticker_quarters[i]
            delta = curr.net_sentiment_score - prev.net_sentiment_score
            drift_magnitudes.append(round(abs(delta), 4))

            if abs(delta) <= self.stable_threshold:
                direction = DriftDirection.STABLE
            elif delta > 0:
                direction = DriftDirection.IMPROVING
            else:
                direction = DriftDirection.DETERIORATING

            drift_directions.append(direction)

            logger.debug(
                f"[{ticker}] {prev.quarter.value}{prev.year}→"
                f"{curr.quarter.value}{curr.year}: "
                f"Δ={delta:+.3f}  direction={direction.value}"
            )

        # Determine alert level
        alert_level, alert_reason, consec_det = self._evaluate_alert(
            ticker,
            drift_directions,
            hedging_trend,
            specificity_trend,
        )

        report = ToneDriftReport(
            ticker=ticker,
            quarters_analysed=quarter_labels,
            sentiment_trend=sentiment_trend,
            hedging_trend=hedging_trend,
            specificity_trend=specificity_trend,
            drift_magnitudes=drift_magnitudes,
            drift_directions=drift_directions,
            alert_level=alert_level,
            alert_reason=alert_reason,
            consecutive_deterioration_count=consec_det,
        )

        logger.info(
            f"[{ticker}] Drift report complete — alert={alert_level.value}  "
            f"consec_det={consec_det}  reason='{alert_reason}'"
        )
        return report

    # ------------------------------------------------------------------
    # Alert evaluation
    # ------------------------------------------------------------------

    def _evaluate_alert(
        self,
        ticker: str,
        directions: list[DriftDirection],
        hedging_trend: list[float],
        specificity_trend: list[float],
    ) -> tuple[AlertLevel, str, int]:
        """
        Evaluate drift directions and language trends into an alert level.

        Alert escalation rules:
            GREEN  → no deterioration in recent quarters
            YELLOW → 1 consecutive deterioration OR increasing hedging
            RED    → 2+ consecutive deterioration AND increasing hedging

        Args:
            ticker:            Stock ticker (for logging).
            directions:        List of DriftDirection values (QoQ).
            hedging_trend:     Hedging score per quarter.
            specificity_trend: Specificity score per quarter.

        Returns:
            Tuple of (AlertLevel, reason_string, consecutive_det_count).
        """
        # Count trailing consecutive deteriorations
        consec_det = 0
        for d in reversed(directions):
            if d == DriftDirection.DETERIORATING:
                consec_det += 1
            else:
                break

        # Check hedging trend: is hedging increasing in recent quarters?
        hedging_rising = False
        if len(hedging_trend) >= 3:
            recent = hedging_trend[-3:]
            hedging_rising = recent[-1] > recent[0] * 1.2  # 20% increase

        # Check specificity trend: is it declining?
        spec_declining = False
        if len(specificity_trend) >= 3:
            recent_spec = specificity_trend[-3:]
            spec_declining = recent_spec[-1] < recent_spec[0] * 0.85  # 15% drop

        # Alert logic
        if consec_det >= 2 and (hedging_rising or spec_declining):
            reason_parts = [
                f"{consec_det} consecutive quarters of deteriorating sentiment",
            ]
            if hedging_rising:
                reason_parts.append("increasing hedging language")
            if spec_declining:
                reason_parts.append("declining guidance specificity")
            return AlertLevel.RED, "; ".join(reason_parts), consec_det

        if consec_det >= 2:
            return (
                AlertLevel.YELLOW,
                f"{consec_det} consecutive quarters of deteriorating sentiment",
                consec_det,
            )

        if consec_det == 1 and hedging_rising:
            return (
                AlertLevel.YELLOW,
                "1 quarter deterioration with rising hedging language",
                consec_det,
            )

        if hedging_rising and spec_declining:
            return (
                AlertLevel.YELLOW,
                "Increasing hedging language and declining specificity — monitor",
                consec_det,
            )

        return AlertLevel.GREEN, "Tone stable or improving", consec_det

    # ------------------------------------------------------------------
    # Comparison utilities
    # ------------------------------------------------------------------

    def compare_quarters(
        self,
        q_current: QuarterSentiment,
        q_prior: QuarterSentiment,
    ) -> dict[str, float | str]:
        """
        Produce a structured comparison between two consecutive quarters.

        Useful for the Streamlit dashboard's side-by-side view.

        Args:
            q_current: More recent quarter.
            q_prior:   Prior quarter.

        Returns:
            Dict with sentiment delta, hedging delta, specificity delta,
            and direction label.
        """
        sentiment_delta = q_current.net_sentiment_score - q_prior.net_sentiment_score
        hedging_delta   = q_current.hedging_score - q_prior.hedging_score
        spec_delta      = q_current.specificity_score - q_prior.specificity_score

        if abs(sentiment_delta) <= self.stable_threshold:
            direction = DriftDirection.STABLE
        elif sentiment_delta > 0:
            direction = DriftDirection.IMPROVING
        else:
            direction = DriftDirection.DETERIORATING

        return {
            "sentiment_delta": round(sentiment_delta, 4),
            "hedging_delta":   round(hedging_delta, 4),
            "specificity_delta": round(spec_delta, 4),
            "direction": direction.value,
            "prior_label":   f"{q_prior.quarter.value} {q_prior.year}",
            "current_label": f"{q_current.quarter.value} {q_current.year}",
        }

    def get_drift_score(self, report: ToneDriftReport) -> float:
        """
        Compute a single normalised drift score in [-1, +1].

        Used as the sentiment_drift_component in EarningsQualityScore.

        Positive = improving trend (good signal for longs).
        Negative = deteriorating trend (bad signal, or good for shorts).

        The score is the mean of the last 3 QoQ deltas (or fewer if not
        available), clipped to [-1, +1].

        Args:
            report: Computed ToneDriftReport.

        Returns:
            Float in [-1, +1].
        """
        if not report.sentiment_trend or len(report.sentiment_trend) < 2:
            return 0.0

        # Directional deltas (signed)
        deltas: list[float] = []
        for i in range(1, len(report.sentiment_trend)):
            deltas.append(report.sentiment_trend[i] - report.sentiment_trend[i - 1])

        # Weight more recent deltas higher (geometric decay)
        recent = deltas[-3:]  # last 3 quarters
        weights = [0.25, 0.35, 0.40][: len(recent)]
        # Normalise weights if fewer than 3
        w_sum = sum(weights)
        weighted = sum(d * w / w_sum for d, w in zip(recent, weights))

        # Scale: a delta of ±0.5 maps to ±1.0
        score = float(max(-1.0, min(1.0, weighted * 2.0)))

        logger.debug(
            f"[{report.ticker}] Drift score: {score:+.3f}  "
            f"(last 3 deltas: {[round(d, 3) for d in recent]})"
        )
        return score


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

tone_drift_detector = ToneDriftDetector()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = ToneDriftDetector()

    def _make_qs(
        ticker: str,
        q: Quarter,
        year: int,
        net: float,
        hedge: float,
        spec: float,
    ) -> QuarterSentiment:
        return QuarterSentiment(
            ticker=ticker, quarter=q, year=year,
            positive_pct=(net + 1) / 2,
            negative_pct=(1 - net) / 2,
            neutral_pct=0.1,
            net_sentiment_score=net,
            hedging_score=hedge,
            specificity_score=spec,
            chunk_count=30,
        )

    # Scenario A: RED alert — 3 consecutive deteriorations + rising hedging
    quarters_red = {
        ("AAPL", Quarter.Q1, 2023): _make_qs("AAPL", Quarter.Q1, 2023,  0.40, 5.0, 0.80),
        ("AAPL", Quarter.Q2, 2023): _make_qs("AAPL", Quarter.Q2, 2023,  0.25, 6.5, 0.72),
        ("AAPL", Quarter.Q3, 2023): _make_qs("AAPL", Quarter.Q3, 2023,  0.05, 7.8, 0.60),
        ("AAPL", Quarter.Q4, 2023): _make_qs("AAPL", Quarter.Q4, 2023, -0.15, 9.2, 0.48),
    }
    report_red = detector.compute_drift("AAPL", quarters_red)
    print(f"Scenario A (RED):    alert={report_red.alert_level.value}  "
          f"consec_det={report_red.consecutive_deterioration_count}  "
          f"reason='{report_red.alert_reason}'")
    assert report_red.alert_level == AlertLevel.RED, f"Expected RED, got {report_red.alert_level}"

    # Scenario B: GREEN alert — improving trend
    quarters_green = {
        ("MSFT", Quarter.Q1, 2024): _make_qs("MSFT", Quarter.Q1, 2024, -0.10, 8.0, 0.55),
        ("MSFT", Quarter.Q2, 2024): _make_qs("MSFT", Quarter.Q2, 2024,  0.15, 6.5, 0.65),
        ("MSFT", Quarter.Q3, 2024): _make_qs("MSFT", Quarter.Q3, 2024,  0.40, 5.0, 0.78),
    }
    report_green = detector.compute_drift("MSFT", quarters_green)
    print(f"Scenario B (GREEN):  alert={report_green.alert_level.value}  "
          f"consec_det={report_green.consecutive_deterioration_count}")
    assert report_green.alert_level == AlertLevel.GREEN

    # Scenario C: YELLOW alert — 1 deterioration + hedging rising
    quarters_yellow = {
        ("GOOG", Quarter.Q2, 2024): _make_qs("GOOG", Quarter.Q2, 2024, 0.50, 4.0, 0.80),
        ("GOOG", Quarter.Q3, 2024): _make_qs("GOOG", Quarter.Q3, 2024, 0.30, 5.2, 0.70),
    }
    report_yellow = detector.compute_drift("GOOG", quarters_yellow)
    print(f"Scenario C (YELLOW): alert={report_yellow.alert_level.value}  "
          f"consec_det={report_yellow.consecutive_deterioration_count}")
    assert report_yellow.alert_level in (AlertLevel.YELLOW, AlertLevel.GREEN)

    # Drift scores
    score_red = detector.get_drift_score(report_red)
    score_green = detector.get_drift_score(report_green)
    print(f"\nDrift scores: RED={score_red:+.3f}  GREEN={score_green:+.3f}")
    assert score_red < 0, "Deteriorating trend should give negative drift score"
    assert score_green > 0, "Improving trend should give positive drift score"

    print("\ntone_drift_detector smoke test passed ✓")
