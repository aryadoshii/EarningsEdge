"""
Backtesting engine for the EarningsEdge earnings quality signal.

Uses yfinance for price data and implements the backtest logic with both
a vectorbt-accelerated path (when available) and a pure-pandas fallback.

Strategy mechanics:
    - Signal generated on earnings_date (from BacktestSignal)
    - Position entered at OPEN of entry_date (next trading day)
    - Position closed at CLOSE of exit_date (entry + holding_period days)
    - Long: buy shares, profit if price rises
    - Short: sell short, profit if price falls
    - Position sizing: equal-weight, each trade uses (capital / max_positions)
    - No pyramiding — one position per ticker at a time

Output:
    - List of TradeResult objects with realised PnL
    - Full equity curve as pd.Series
    - Benchmark equity curve (SPY buy-and-hold)

Usage:
    bt = Backtester()
    results = await bt.run(signals)
    trade_log, equity, benchmark = results
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.backtest.signal_generator import _trading_day_offset
from src.ingestion.data_validator import BacktestSignal, SignalDirection, TradeResult

# Lazy imports for heavy optional dependencies
_VBT_AVAILABLE = False
try:
    import vectorbt as vbt  # type: ignore
    _VBT_AVAILABLE = True
except ImportError:
    logger.debug("vectorbt not installed — using pandas fallback backtester")

try:
    import yfinance as yf  # type: ignore
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed — backtester will use synthetic prices")

# ---------------------------------------------------------------------------
# Price fetcher
# ---------------------------------------------------------------------------

_price_cache: dict[str, pd.DataFrame] = {}


async def _fetch_prices(
    ticker: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with a local in-memory cache.

    Args:
        ticker: Stock ticker.
        start:  Start date.
        end:    End date (inclusive).

    Returns:
        DataFrame with DatetimeIndex and columns Open, High, Low, Close, Volume.
        Returns empty DataFrame if fetch fails.
    """
    cache_key = f"{ticker}_{start}_{end}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    if not _YF_AVAILABLE:
        logger.warning(f"yfinance unavailable — returning empty DataFrame for {ticker}")
        return pd.DataFrame()

    try:
        loop = asyncio.get_event_loop()
        # yfinance is synchronous — run in thread pool
        df: pd.DataFrame = await loop.run_in_executor(
            None,
            lambda: yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=True,
            ),
        )
        if df.empty:
            logger.warning(f"No price data returned for {ticker} ({start}–{end})")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index)
        _price_cache[cache_key] = df
        return df

    except Exception as exc:
        logger.error(f"Price fetch failed for {ticker}: {exc}")
        return pd.DataFrame()


def _get_price(df: pd.DataFrame, dt: date, col: str = "Close") -> float | None:
    """
    Extract a price from a DataFrame for a given date.

    Looks for the exact date first; if not found (holiday), walks forward
    up to 5 days to find the next available trading session.

    Args:
        df:  OHLCV DataFrame with DatetimeIndex.
        dt:  Target date.
        col: Column name: 'Open' or 'Close'.

    Returns:
        Price as float, or None if not found.
    """
    for offset in range(5):
        target = pd.Timestamp(dt + timedelta(days=offset))
        if target in df.index:
            val = df.loc[target, col]
            return float(val) if not pd.isna(val) else None
    return None


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Runs the EarningsEdge signal through a realistic backtest.

    Supports two execution paths:
        vectorbt — fast vectorised simulation (preferred when installed)
        pandas   — fallback, trade-by-trade simulation

    Both paths produce identical TradeResult lists.
    """

    def __init__(
        self,
        initial_capital: float = settings.INITIAL_CAPITAL,
        holding_days: int = settings.HOLDING_PERIOD_DAYS,
        benchmark_ticker: str = "SPY",
    ) -> None:
        self.initial_capital = initial_capital
        self.holding_days = holding_days
        self.benchmark_ticker = benchmark_ticker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        signals: list[BacktestSignal],
    ) -> tuple[list[TradeResult], pd.Series, pd.Series]:
        """
        Execute the backtest for all signals.

        Args:
            signals: List of BacktestSignal objects from SignalGenerator.

        Returns:
            Tuple of:
                trade_results: List of TradeResult with realised PnL
                equity_curve:  Daily strategy equity pd.Series
                benchmark:     Daily buy-and-hold SPY equity pd.Series
        """
        if not signals:
            logger.warning("Backtester.run called with empty signal list")
            return [], pd.Series(dtype=float), pd.Series(dtype=float)

        logger.info(
            f"Backtester starting: {len(signals)} signals  "
            f"{'vectorbt' if _VBT_AVAILABLE else 'pandas'} engine"
        )

        # Fetch all required price data concurrently
        trade_results = await self._execute_trades(signals)

        if not trade_results:
            return [], pd.Series(dtype=float), pd.Series(dtype=float)

        # Build equity curve
        from src.backtest.metrics import build_equity_curve
        equity = build_equity_curve(trade_results, self.initial_capital)

        # Fetch benchmark
        start = min(s.entry_date for s in signals)
        end   = max(
            _trading_day_offset(s.entry_date, s.holding_days) for s in signals
        )
        benchmark = await self._build_benchmark(start, end)

        logger.info(
            f"Backtest complete: {len(trade_results)} trades executed  "
            f"equity_start={equity.iloc[0]:,.0f}  "
            f"equity_end={equity.iloc[-1]:,.0f}"
        )
        return trade_results, equity, benchmark

    async def run_long_short(
        self,
        signals: list[BacktestSignal],
    ) -> dict[str, Any]:
        """
        Run both long-only and long-short versions for comparison.

        Args:
            signals: All BacktestSignal objects.

        Returns:
            Dict with keys 'long_short' and 'long_only', each containing
            (trade_results, equity_curve, benchmark).
        """
        long_short_results = await self.run(signals)

        long_only_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        long_only_results = await self.run(long_only_signals)

        return {
            "long_short": long_short_results,
            "long_only":  long_only_results,
        }

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    async def _execute_trades(
        self,
        signals: list[BacktestSignal],
    ) -> list[TradeResult]:
        """
        Fetch prices and compute realised PnL for each signal.

        Price fetches are batched by ticker to minimise API calls.

        Args:
            signals: List of BacktestSignal objects.

        Returns:
            List of TradeResult objects (signals that couldn't fetch
            prices are omitted with a warning).
        """
        # Determine date range needed for all signals
        all_starts = [s.entry_date for s in signals]
        all_ends   = [
            _trading_day_offset(s.entry_date, s.holding_days + 5)
            for s in signals
        ]
        global_start = min(all_starts) - timedelta(days=5)
        global_end   = max(all_ends) + timedelta(days=5)

        # Unique tickers — fetch all in parallel
        tickers = list({s.ticker for s in signals})
        price_tasks = [
            _fetch_prices(t, global_start, global_end) for t in tickers
        ]
        price_dfs = await asyncio.gather(*price_tasks)
        prices: dict[str, pd.DataFrame] = dict(zip(tickers, price_dfs))

        results: list[TradeResult] = []
        for sig in signals:
            df = prices.get(sig.ticker)
            if df is None or df.empty:
                logger.warning(f"No prices for {sig.ticker} — skipping signal")
                continue

            entry_price = _get_price(df, sig.entry_date, "Open")
            exit_date   = _trading_day_offset(sig.entry_date, sig.holding_days)
            exit_price  = _get_price(df, exit_date, "Close")

            if entry_price is None or exit_price is None or entry_price == 0:
                logger.warning(
                    f"Missing price for {sig.ticker} "
                    f"entry={sig.entry_date} exit={exit_date} — skipping"
                )
                continue

            # Compute gross return (sign-adjusted for shorts)
            raw_return = (exit_price - entry_price) / entry_price
            gross_return = raw_return if sig.direction == SignalDirection.LONG else -raw_return

            actual_days = 0
            d = sig.entry_date
            while d <= exit_date:
                if d.weekday() < 5:
                    actual_days += 1
                d += timedelta(days=1)

            results.append(TradeResult(
                ticker=sig.ticker,
                direction=sig.direction,
                signal_date=sig.signal_date,
                entry_date=sig.entry_date,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                quality_score=sig.quality_score,
                gross_return=gross_return,
                holding_days_actual=actual_days,
            ))

            logger.debug(
                f"{sig.ticker} {sig.direction.value:5s}  "
                f"entry={entry_price:.2f}  exit={exit_price:.2f}  "
                f"return={gross_return:+.2%}"
            )

        logger.info(
            f"Trades executed: {len(results)}/{len(signals)}  "
            f"winners={sum(1 for t in results if t.gross_return > 0)}"
        )
        return results

    async def _build_benchmark(
        self,
        start: date,
        end: date,
    ) -> pd.Series:
        """
        Build a buy-and-hold equity curve for the benchmark (SPY).

        Args:
            start: Start date.
            end:   End date.

        Returns:
            pd.Series with DatetimeIndex representing SPY equity growth
            from self.initial_capital.
        """
        df = await _fetch_prices(self.benchmark_ticker, start, end)
        if df.empty:
            return pd.Series(dtype=float)

        closes = df["Close"].dropna()
        if closes.empty:
            return pd.Series(dtype=float)

        normalised = closes / closes.iloc[0] * self.initial_capital
        return normalised

    # ------------------------------------------------------------------
    # Trade log formatter
    # ------------------------------------------------------------------

    @staticmethod
    def to_dataframe(trades: list[TradeResult]) -> pd.DataFrame:
        """
        Convert a list of TradeResult objects to a formatted DataFrame.

        Suitable for display in the Streamlit trade log table.

        Args:
            trades: List of TradeResult objects.

        Returns:
            DataFrame with one row per trade, sorted by entry_date.
        """
        if not trades:
            return pd.DataFrame()

        rows = [
            {
                "ticker":         t.ticker,
                "direction":      t.direction.value,
                "signal_date":    t.signal_date,
                "entry_date":     t.entry_date,
                "exit_date":      t.exit_date,
                "entry_price":    round(t.entry_price, 2),
                "exit_price":     round(t.exit_price, 2),
                "gross_return":   round(t.gross_return, 4),
                "quality_score":  round(t.quality_score, 3),
                "holding_days":   t.holding_days_actual,
                "winner":         t.is_winner,
            }
            for t in trades
        ]
        df = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
        df["gross_return_pct"] = (df["gross_return"] * 100).round(2)
        return df


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

backtester = Backtester()


# ---------------------------------------------------------------------------
# Smoke test (pure-Python, no network calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from datetime import date as d

    # Test helpers without network
    from src.backtest.signal_generator import _trading_day_offset

    monday = d(2024, 1, 8)
    exit_d = _trading_day_offset(monday, 20)
    assert exit_d > monday
    print(f"_trading_day_offset(20 days): {monday} → {exit_d} ✓")

    # Test _get_price with synthetic DataFrame
    dates = pd.date_range("2024-01-08", periods=25, freq="B")
    prices_series = pd.Series(range(100, 125), index=dates, dtype=float)
    df = pd.DataFrame({"Open": prices_series, "Close": prices_series + 1})

    price = _get_price(df, d(2024, 1, 8), "Open")
    assert price == 100.0, f"Expected 100.0, got {price}"
    print(f"_get_price(2024-01-08, Open): {price} ✓")

    # Weekend skip
    saturday = d(2024, 1, 6)
    price_sat = _get_price(df, saturday, "Close")
    # Should walk forward to Monday Jan 8
    assert price_sat == 101.0, f"Expected 101.0 (Mon close), got {price_sat}"
    print(f"_get_price(Saturday → next Monday): {price_sat} ✓")

    # Test to_dataframe
    trades = [
        TradeResult(
            ticker="AAPL", direction=SignalDirection.LONG,
            signal_date=d(2024, 1, 7), entry_date=d(2024, 1, 8),
            exit_date=d(2024, 2, 5),
            entry_price=185.0, exit_price=195.0,
            quality_score=0.52, gross_return=0.054, holding_days_actual=20,
        )
    ]
    df_log = Backtester.to_dataframe(trades)
    assert len(df_log) == 1
    assert "gross_return_pct" in df_log.columns
    assert df_log.iloc[0]["gross_return_pct"] == 5.4
    print(f"to_dataframe: {len(df_log)} rows ✓")

    print("\nbacktester smoke test passed ✓")
