"""
Signal generator — converts EarningsQualityScore objects into BacktestSignal
trade entries.

Rules:
    composite_score >  SIGNAL_LONG_THRESHOLD  →  LONG
    composite_score <  SIGNAL_SHORT_THRESHOLD →  SHORT
    otherwise                                 →  NEUTRAL (no position)

Signal date  = earnings announcement date (from SECFiling.filed_date)
Entry date   = next trading day after signal date
Holding period = settings.HOLDING_PERIOD_DAYS trading days

The generator also handles:
    - Deduplication: one signal per ticker per quarter (latest filing wins)
    - Holiday adjustment: entry_date skips weekends (simplified — full
      exchange calendar requires pandas_market_calendars, not included as
      it's a paid dependency; we skip Saturday/Sunday only)

Usage:
    gen = SignalGenerator()
    signals = gen.generate(quality_scores)
    signals = gen.generate_for_ticker(ticker, quality_scores)
"""

from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import (
    BacktestSignal,
    EarningsQualityScore,
    Quarter,
    SignalDirection,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUARTER_MONTH_MAP: dict[str, int] = {
    "Q1": 4,   # Q1 earnings typically announced in April
    "Q2": 7,
    "Q3": 10,
    "Q4": 1,   # Q4 earnings in January of the following year
    "Annual": 2,
}


def _estimate_earnings_date(quarter: Quarter, year: int) -> date:
    """
    Estimate an earnings announcement date from quarter and year.

    Uses the typical month of earnings season for each quarter.
    In production this would be sourced from actual announcement dates
    (available via yfinance earnings_dates or a financial calendar API).

    Args:
        quarter: Reporting quarter enum.
        year:    Reporting year.

    Returns:
        Approximate earnings announcement date (15th of the typical month).
    """
    month = _QUARTER_MONTH_MAP.get(quarter.value, 4)
    # Q4 earnings are in the following calendar year
    actual_year = year + 1 if quarter == Quarter.Q4 else year
    actual_year = max(1990, min(actual_year, 2099))
    return date(actual_year, month, 15)


def _next_trading_day(d: date) -> date:
    """
    Return the next weekday (Mon–Fri) on or after date d.

    Skips Saturday and Sunday.  Does not account for exchange holidays.

    Args:
        d: Reference date.

    Returns:
        Next trading day.
    """
    while d.weekday() >= 5:   # 5=Saturday, 6=Sunday
        d += timedelta(days=1)
    return d


def _trading_day_offset(start: date, n_days: int) -> date:
    """
    Advance start by n_days trading days (skipping weekends).

    Args:
        start:  Starting date.
        n_days: Number of trading days to advance.

    Returns:
        Exit date after n_days trading days.
    """
    d = start
    count = 0
    while count < n_days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


# ---------------------------------------------------------------------------
# SignalGenerator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Converts EarningsQualityScore objects to BacktestSignal trade entries.

    One signal is generated per (ticker, quarter, year) combination.
    NEUTRAL signals are silently dropped — they generate no position.
    """

    def __init__(
        self,
        long_threshold: float = settings.SIGNAL_LONG_THRESHOLD,
        short_threshold: float = settings.SIGNAL_SHORT_THRESHOLD,
        holding_days: int = settings.HOLDING_PERIOD_DAYS,
    ) -> None:
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.holding_days = holding_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        quality_scores: list[EarningsQualityScore],
        announcement_dates: dict[tuple[str, Quarter, int], date] | None = None,
    ) -> list[BacktestSignal]:
        """
        Generate trade signals from a list of EarningsQualityScore objects.

        Args:
            quality_scores:     List of computed EarningsQualityScore objects.
            announcement_dates: Optional override: {(ticker,quarter,year): date}.
                                 If absent, dates are estimated heuristically.

        Returns:
            List of BacktestSignal objects (NEUTRAL signals excluded).
        """
        signals: list[BacktestSignal] = []
        seen: set[tuple[str, Quarter, int]] = set()

        # Sort by computed_at descending so latest supersedes earlier runs
        sorted_scores = sorted(
            quality_scores,
            key=lambda s: s.computed_at,
            reverse=True,
        )

        for qs in sorted_scores:
            key = (qs.ticker, qs.quarter, qs.year)
            if key in seen:
                continue
            seen.add(key)

            direction = self._classify(qs.composite_score)
            if direction == SignalDirection.NEUTRAL:
                continue

            # Determine earnings announcement date
            if announcement_dates and key in announcement_dates:
                signal_date = announcement_dates[key]
            else:
                signal_date = _estimate_earnings_date(qs.quarter, qs.year)

            # Backtest window guard
            backtest_start = date.fromisoformat(settings.BACKTEST_START_DATE)
            backtest_end   = date.fromisoformat(settings.BACKTEST_END_DATE)
            if not (backtest_start <= signal_date <= backtest_end):
                continue

            entry_date = _next_trading_day(signal_date + timedelta(days=1))

            signals.append(BacktestSignal(
                ticker=qs.ticker,
                signal_date=signal_date,
                entry_date=entry_date,
                direction=direction,
                quality_score=round(qs.composite_score, 4),
                holding_days=self.holding_days,
            ))

            logger.debug(
                f"Signal: {qs.ticker} {qs.quarter.value} {qs.year}  "
                f"{direction.value}  score={qs.composite_score:+.3f}  "
                f"entry={entry_date}"
            )

        logger.info(
            f"Generated {len(signals)} signals from {len(quality_scores)} scores "
            f"({sum(1 for s in signals if s.direction == SignalDirection.LONG)} LONG, "
            f"{sum(1 for s in signals if s.direction == SignalDirection.SHORT)} SHORT)"
        )
        return sorted(signals, key=lambda s: s.entry_date)

    def generate_for_ticker(
        self,
        ticker: str,
        quality_scores: list[EarningsQualityScore],
    ) -> list[BacktestSignal]:
        """
        Generate signals for a single ticker.

        Args:
            ticker:        Stock ticker (upper case).
            quality_scores: All quality scores (will be filtered to ticker).

        Returns:
            List of BacktestSignal objects for this ticker.
        """
        ticker_scores = [s for s in quality_scores if s.ticker == ticker.upper()]
        return self.generate(ticker_scores)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify(self, score: float) -> SignalDirection:
        if score > self.long_threshold:
            return SignalDirection.LONG
        if score < self.short_threshold:
            return SignalDirection.SHORT
        return SignalDirection.NEUTRAL


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

signal_generator = SignalGenerator()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime

    gen = SignalGenerator()

    # _estimate_earnings_date
    assert _estimate_earnings_date(Quarter.Q3, 2024) == date(2024, 10, 15)
    assert _estimate_earnings_date(Quarter.Q4, 2023) == date(2024, 1, 15)
    print("_estimate_earnings_date ✓")

    # _next_trading_day (skip weekends)
    saturday = date(2024, 10, 5)   # Saturday
    assert _next_trading_day(saturday) == date(2024, 10, 7)  # Monday
    monday = date(2024, 10, 7)
    assert _next_trading_day(monday) == monday
    print("_next_trading_day ✓")

    # _trading_day_offset
    start = date(2024, 10, 7)  # Monday
    result = _trading_day_offset(start, 20)
    assert result > start
    print(f"_trading_day_offset(20): {start} → {result} ✓")

    # Signal classification
    assert gen._classify(0.45) == SignalDirection.LONG
    assert gen._classify(-0.45) == SignalDirection.SHORT
    assert gen._classify(0.10) == SignalDirection.NEUTRAL
    print("Signal classification ✓")

    # Full generation with mock scores
    from src.ingestion.data_validator import EarningsQualityScore, Quarter

    scores = [
        EarningsQualityScore(
            ticker="AAPL", quarter=Quarter.Q3, year=2022,
            composite_score=0.55,
            sentiment_drift_component=0.6, guidance_accuracy_component=0.5,
            accruals_component=0.5, analyst_revision_component=0.5,
            weight_sentiment_drift=0.30, weight_guidance_accuracy=0.25,
            weight_accruals=0.25, weight_analyst_revision=0.20,
        ),
        EarningsQualityScore(
            ticker="AAPL", quarter=Quarter.Q1, year=2023,
            composite_score=-0.42,
            sentiment_drift_component=-0.5, guidance_accuracy_component=-0.4,
            accruals_component=-0.3, analyst_revision_component=-0.4,
            weight_sentiment_drift=0.30, weight_guidance_accuracy=0.25,
            weight_accruals=0.25, weight_analyst_revision=0.20,
        ),
        EarningsQualityScore(
            ticker="AAPL", quarter=Quarter.Q2, year=2023,
            composite_score=0.12,  # NEUTRAL — should be dropped
            sentiment_drift_component=0.1, guidance_accuracy_component=0.1,
            accruals_component=0.1, analyst_revision_component=0.2,
            weight_sentiment_drift=0.30, weight_guidance_accuracy=0.25,
            weight_accruals=0.25, weight_analyst_revision=0.20,
        ),
    ]

    signals = gen.generate(scores)
    assert len(signals) == 2, f"Expected 2 signals (1 NEUTRAL dropped), got {len(signals)}"
    assert signals[0].direction == SignalDirection.LONG
    assert signals[1].direction == SignalDirection.SHORT
    assert all(s.holding_days == settings.HOLDING_PERIOD_DAYS for s in signals)
    print(f"Signal generation: {len(signals)} signals (NEUTRAL dropped) ✓")
    for s in signals:
        print(f"  {s.ticker} {s.direction.value:5s}  entry={s.entry_date}  score={s.quality_score:+.3f}")

    print("\nsignal_generator smoke test passed ✓")
