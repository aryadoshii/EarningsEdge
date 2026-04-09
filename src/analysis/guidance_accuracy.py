"""
Guidance accuracy tracker.

Matches forward-looking guidance extracted by the NER pipeline against
actual reported XBRL financial figures from subsequent filings, then
computes per-metric accuracy and management guidance bias.

Key computations:
    accuracy_score  = 1 - |guided_mid - actual| / |actual|
                      Capped to [0, 1].  1.0 = perfect guidance.

    guidance_bias   = (guided_mid - actual) / |actual|
                      Positive = management guided too high (optimistic).
                      Negative = management guided too low (sandbagging).
                      Sandbagging is a POSITIVE signal — companies that
                      consistently guide low and beat are quality compounders.

    component_score = normalised composite guidance accuracy in [-1, +1]
                      used as the guidance_accuracy_component in
                      EarningsQualityScore.

Usage:
    tracker = GuidanceAccuracyTracker()
    records = tracker.match_guidance_to_actuals(guidance_entities, filings)
    score   = tracker.compute_component_score(records)
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev

from loguru import logger

from src.ingestion.data_validator import (
    GuidanceEntity,
    GuidanceMetric,
    GuidanceRecord,
    Quarter,
    SECFiling,
    XBRLData,
)

# ---------------------------------------------------------------------------
# XBRL field → GuidanceMetric mapping
# ---------------------------------------------------------------------------

# Maps a GuidanceMetric to the XBRLData attribute that holds the actual value
_METRIC_TO_XBRL: dict[GuidanceMetric, str] = {
    GuidanceMetric.EPS:     "eps_diluted",
    GuidanceMetric.REVENUE: "revenue",
    GuidanceMetric.CAPEX:   "capex",
    GuidanceMetric.MARGIN:  "gross_margin",
}

# For MARGIN guidance the guided value is in % (e.g. 46.3) but XBRL stores
# it as a decimal (0.463).  This flag triggers the conversion.
_METRIC_IS_PERCENT: set[GuidanceMetric] = {GuidanceMetric.MARGIN, GuidanceMetric.GROWTH}


def _get_actual(xbrl: XBRLData, metric: GuidanceMetric) -> float | None:
    """
    Extract the actual reported value for a metric from an XBRLData object.

    Args:
        xbrl:   XBRLData populated from a subsequent filing.
        metric: The metric we're looking for.

    Returns:
        Float value in the same units as the guided value, or None.
    """
    field = _METRIC_TO_XBRL.get(metric)
    if field is None:
        return None

    value = getattr(xbrl, field, None)
    if value is None:
        return None

    # Convert decimal margin to percentage to match guided values
    if metric in _METRIC_IS_PERCENT and 0.0 < value <= 1.0:
        value = value * 100.0

    return float(value)


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def _accuracy_score(guided_mid: float, actual: float) -> float:
    """
    Compute guidance accuracy score in [0, 1].

    accuracy = 1 - abs(guided_mid - actual) / abs(actual)
    Clamped to [0, 1] — a miss larger than 100% of actual gives 0.0.

    Args:
        guided_mid: Mid-point of the guided range (or single value).
        actual:     Actual reported value.

    Returns:
        Float in [0, 1].
    """
    if actual == 0:
        return 0.5  # undefined — return neutral

    raw = 1.0 - abs(guided_mid - actual) / abs(actual)
    return max(0.0, min(1.0, raw))


def _guidance_bias(guided_mid: float, actual: float) -> float:
    """
    Compute guidance bias in (-∞, +∞).

    Positive = guided too high (optimistic).
    Negative = guided too low (sandbagging).

    Args:
        guided_mid: Mid-point of the guided range.
        actual:     Actual reported value.

    Returns:
        Bias as a fraction of actual.  Clipped to [-2, +2].
    """
    if actual == 0:
        return 0.0
    raw = (guided_mid - actual) / abs(actual)
    return max(-2.0, min(2.0, raw))


# ---------------------------------------------------------------------------
# Quarter ordering utility
# ---------------------------------------------------------------------------

_Q_ORDER: dict[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Annual": 5}


def _sort_key_filing(f: SECFiling) -> tuple[int, int]:
    q_val = _Q_ORDER.get(f.quarter.value if f.quarter else "Q4", 4)
    return (f.year or 0, q_val)


def _next_period(quarter: Quarter, year: int) -> tuple[Quarter, int]:
    """Return the (quarter, year) that immediately follows the given period."""
    mapping = {
        Quarter.Q1: (Quarter.Q2, year),
        Quarter.Q2: (Quarter.Q3, year),
        Quarter.Q3: (Quarter.Q4, year),
        Quarter.Q4: (Quarter.Q1, year + 1),
        Quarter.ANNUAL: (Quarter.Q1, year + 1),
    }
    return mapping[quarter]


# ---------------------------------------------------------------------------
# GuidanceAccuracyTracker
# ---------------------------------------------------------------------------

class GuidanceAccuracyTracker:
    """
    Matches NER-extracted guidance against subsequently reported actuals.

    The fundamental operation is:
        guidance (Q_t) + actual (Q_{t+1}) → GuidanceRecord

    Records are accumulated over multiple quarters and used to compute:
        - Per-metric accuracy trends
        - Overall guidance bias (optimistic vs conservative)
        - Composite guidance_accuracy_component score for the quality scorer
    """

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_guidance_to_actuals(
        self,
        guidance_by_period: dict[tuple[Quarter, int], list[GuidanceEntity]],
        filings: list[SECFiling],
    ) -> list[GuidanceRecord]:
        """
        Match guidance entities to actual reported XBRL values.

        For each (quarter, year) in guidance_by_period, looks for a filing
        from the next quarter that contains XBRL data.

        Args:
            guidance_by_period: {(quarter, year): [GuidanceEntity]} output
                                 from the NER pipeline.
            filings:             All available SECFiling objects for the ticker.

        Returns:
            List of GuidanceRecord objects (matched and unmatched).
            Unmatched records (no actual) have accuracy_score=None.
        """
        if not filings:
            return []

        ticker = filings[0].ticker

        # Build a lookup: (quarter, year) → SECFiling with XBRL data
        filing_lookup: dict[tuple[str, int], SECFiling] = {}
        for f in filings:
            if f.quarter:
                key = (f.quarter.value, f.year or 0)
                # Prefer later amendments
                if key not in filing_lookup or f.is_amendment:
                    filing_lookup[key] = f

        records: list[GuidanceRecord] = []

        for (guidance_q, guidance_yr), entities in guidance_by_period.items():
            if not entities:
                continue

            # Find the subsequent filing (the one that reports actuals)
            next_q, next_yr = _next_period(guidance_q, guidance_yr)
            actual_filing = filing_lookup.get((next_q.value, next_yr))

            for entity in entities:
                # Compute midpoint of guided range
                if entity.value_low is not None and entity.value_high is not None:
                    guided_mid: float | None = (entity.value_low + entity.value_high) / 2.0
                elif entity.value_low is not None:
                    guided_mid = entity.value_low
                else:
                    guided_mid = None

                actual: float | None = None
                accuracy: float | None = None
                bias: float | None = None

                if guided_mid is not None and actual_filing is not None:
                    actual = _get_actual(actual_filing.xbrl_data, entity.metric_type)
                    if actual is not None:
                        accuracy = _accuracy_score(guided_mid, actual)
                        bias = _guidance_bias(guided_mid, actual)

                records.append(GuidanceRecord(
                    ticker=ticker,
                    quarter=guidance_q,
                    year=guidance_yr,
                    metric_type=entity.metric_type,
                    guided_low=entity.value_low,
                    guided_high=entity.value_high,
                    guided_mid=guided_mid,
                    actual_value=actual,
                    accuracy_score=accuracy,
                    guidance_bias=bias,
                    filing_source=actual_filing.filing_url if actual_filing else "",
                ))

                logger.debug(
                    f"[{ticker}] {guidance_q.value} {guidance_yr} "
                    f"{entity.metric_type.value}: "
                    f"guided_mid={guided_mid}  actual={actual}  "
                    f"accuracy={accuracy:.3f if accuracy else 'N/A'}  "
                    f"bias={bias:.3f if bias else 'N/A'}"
                )

        logger.info(
            f"[{ticker}] Guidance matching: {len(records)} records, "
            f"{sum(1 for r in records if r.accuracy_score is not None)} matched"
        )
        return records

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_component_score(
        self,
        records: list[GuidanceRecord],
        min_records: int = 3,
    ) -> float:
        """
        Compute the guidance_accuracy_component score in [-1, +1].

        Aggregation logic:
            1. Compute mean accuracy across all matched records.
            2. Compute mean guidance bias.
            3. Adjust raw accuracy for sandbagging bonus:
               consistent conservative guidance (mean_bias < -0.05)
               receives a +0.1 bonus to the final score.
            4. Map [0, 1] accuracy to [-1, +1] and clip.

        Args:
            records:     List of GuidanceRecord objects from match_guidance_to_actuals.
            min_records: Minimum matched records needed for a reliable score.
                         Returns 0.0 if fewer records are available.

        Returns:
            Float in [-1, +1]:
                +1.0 = perfect guidance accuracy (or consistent sandbagging)
                 0.0 = neutral (50% accuracy or insufficient data)
                -1.0 = consistently misleading / inaccurate guidance
        """
        matched = [r for r in records if r.accuracy_score is not None]

        if len(matched) < min_records:
            logger.warning(
                f"Only {len(matched)} matched records "
                f"(need {min_records}) — returning neutral score"
            )
            return 0.0

        mean_accuracy = mean(r.accuracy_score for r in matched)  # type: ignore[arg-type]
        biases = [r.guidance_bias for r in matched if r.guidance_bias is not None]
        mean_bias = mean(biases) if biases else 0.0

        # Sandbagging bonus: consistent conservative guidance is positive
        sandbagging_bonus = 0.10 if mean_bias < -0.05 else 0.0

        # Map [0, 1] accuracy → [-1, +1]
        # mean_accuracy = 0.5 → score = 0.0
        # mean_accuracy = 1.0 → score = 1.0
        # mean_accuracy = 0.0 → score = -1.0
        raw_score = (mean_accuracy - 0.5) * 2.0 + sandbagging_bonus

        score = float(max(-1.0, min(1.0, raw_score)))

        logger.info(
            f"Guidance accuracy score: {score:+.3f}  "
            f"(mean_accuracy={mean_accuracy:.3f}  mean_bias={mean_bias:+.3f}  "
            f"sandbagging_bonus={sandbagging_bonus:.2f}  n={len(matched)})"
        )
        return score

    def compute_per_metric_summary(
        self,
        records: list[GuidanceRecord],
    ) -> dict[str, dict[str, float]]:
        """
        Compute per-metric accuracy and bias statistics.

        Useful for the Streamlit dashboard's guidance breakdown table.

        Args:
            records: All GuidanceRecord objects for a ticker.

        Returns:
            Dict {metric_name: {mean_accuracy, mean_bias, std_bias, count}}.
        """
        by_metric: dict[str, list[GuidanceRecord]] = defaultdict(list)
        for r in records:
            if r.accuracy_score is not None:
                by_metric[r.metric_type.value].append(r)

        summary: dict[str, dict[str, float]] = {}
        for metric, recs in by_metric.items():
            accuracies = [r.accuracy_score for r in recs]  # type: ignore
            biases = [r.guidance_bias for r in recs if r.guidance_bias is not None]
            summary[metric] = {
                "mean_accuracy": round(mean(accuracies), 4),
                "mean_bias":     round(mean(biases) if biases else 0.0, 4),
                "std_bias":      round(stdev(biases) if len(biases) > 1 else 0.0, 4),
                "count":         float(len(recs)),
            }
        return summary

    def get_bias_label(self, mean_bias: float) -> str:
        """
        Return a human-readable bias label.

        Args:
            mean_bias: Mean guidance bias (signed fraction).

        Returns:
            One of: 'sandbagging', 'accurate', 'slightly optimistic',
                    'optimistic', 'aggressively optimistic'
        """
        if mean_bias < -0.10:
            return "sandbagging"
        if mean_bias < 0.05:
            return "accurate"
        if mean_bias < 0.10:
            return "slightly optimistic"
        if mean_bias < 0.25:
            return "optimistic"
        return "aggressively optimistic"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

guidance_accuracy_tracker = GuidanceAccuracyTracker()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date as d

    tracker = GuidanceAccuracyTracker()

    # ── Test accuracy score calculation ──────────────────────────────────
    assert abs(_accuracy_score(1.55, 1.60) - (1 - abs(1.55 - 1.60) / 1.60)) < 1e-6
    assert _accuracy_score(90e9, 90e9) == 1.0
    assert _accuracy_score(100e9, 90e9) < 0.9
    assert _accuracy_score(0.0, 90e9) == 0.0  # complete miss — clamped to 0
    print("Accuracy score logic ✓")

    # ── Test guidance bias labels ─────────────────────────────────────────
    assert tracker.get_bias_label(-0.15) == "sandbagging"
    assert tracker.get_bias_label(-0.02) == "accurate"
    assert tracker.get_bias_label(0.07) == "slightly optimistic"
    assert tracker.get_bias_label(0.15) == "optimistic"
    assert tracker.get_bias_label(0.35) == "aggressively optimistic"
    print("Bias labels ✓")

    # ── Test component score ──────────────────────────────────────────────
    # High accuracy + sandbagging bias → positive score
    good_records = [
        GuidanceRecord(
            ticker="AAPL", quarter=Quarter.Q1, year=2024,
            metric_type=GuidanceMetric.EPS,
            guided_low=1.50, guided_high=1.60, guided_mid=1.55,
            actual_value=1.65,  # beat
            accuracy_score=_accuracy_score(1.55, 1.65),
            guidance_bias=_guidance_bias(1.55, 1.65),
        ),
        GuidanceRecord(
            ticker="AAPL", quarter=Quarter.Q2, year=2024,
            metric_type=GuidanceMetric.EPS,
            guided_low=1.55, guided_high=1.65, guided_mid=1.60,
            actual_value=1.70,
            accuracy_score=_accuracy_score(1.60, 1.70),
            guidance_bias=_guidance_bias(1.60, 1.70),
        ),
        GuidanceRecord(
            ticker="AAPL", quarter=Quarter.Q3, year=2024,
            metric_type=GuidanceMetric.EPS,
            guided_low=1.50, guided_high=1.60, guided_mid=1.55,
            actual_value=1.68,
            accuracy_score=_accuracy_score(1.55, 1.68),
            guidance_bias=_guidance_bias(1.55, 1.68),
        ),
    ]
    score = tracker.compute_component_score(good_records, min_records=3)
    print(f"Sandbagging company score: {score:+.3f} (expected positive) ✓")
    assert score > 0, f"Expected positive score for sandbagging company, got {score}"

    # Poor accuracy → negative score
    bad_records = [
        GuidanceRecord(
            ticker="XYZ", quarter=Quarter.Q1, year=2024,
            metric_type=GuidanceMetric.REVENUE,
            guided_low=90e9, guided_high=95e9, guided_mid=92.5e9,
            actual_value=78e9,  # big miss
            accuracy_score=_accuracy_score(92.5e9, 78e9),
            guidance_bias=_guidance_bias(92.5e9, 78e9),
        ),
        GuidanceRecord(
            ticker="XYZ", quarter=Quarter.Q2, year=2024,
            metric_type=GuidanceMetric.REVENUE,
            guided_low=85e9, guided_high=90e9, guided_mid=87.5e9,
            actual_value=72e9,
            accuracy_score=_accuracy_score(87.5e9, 72e9),
            guidance_bias=_guidance_bias(87.5e9, 72e9),
        ),
        GuidanceRecord(
            ticker="XYZ", quarter=Quarter.Q3, year=2024,
            metric_type=GuidanceMetric.REVENUE,
            guided_low=80e9, guided_high=85e9, guided_mid=82.5e9,
            actual_value=70e9,
            accuracy_score=_accuracy_score(82.5e9, 70e9),
            guidance_bias=_guidance_bias(82.5e9, 70e9),
        ),
    ]
    bad_score = tracker.compute_component_score(bad_records, min_records=3)
    print(f"Optimistic-miss company score: {bad_score:+.3f} (expected negative) ✓")
    assert bad_score < 0, f"Expected negative score, got {bad_score}"

    summary = tracker.compute_per_metric_summary(good_records + bad_records)
    print(f"Metric summary: {list(summary.keys())} ✓")

    print("\nguidance_accuracy smoke test passed ✓")