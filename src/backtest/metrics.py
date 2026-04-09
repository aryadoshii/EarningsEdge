"""
Backtest performance metrics.

Computes all quantitative performance metrics for the EarningsEdge signal:

    Annualised Sharpe Ratio      — risk-adjusted return vs risk-free rate
    Hit Rate                     — % of trades that are profitable
    Average winning trade return
    Average losing trade return
    Win / Loss ratio
    Maximum drawdown             — peak-to-trough equity decline
    Annualised return
    Information Coefficient      — Spearman correlation of signal with forward return
    All metrics split: overall / long-leg / short-leg

All functions are pure (no I/O).  Inputs are lists of TradeResult objects
and/or a pandas Series of daily equity values.

Usage:
    from src.backtest.metrics import compute_all_metrics
    stats = compute_all_metrics(trades, equity_curve)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import SignalDirection, TradeResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_DAILY = settings.RISK_FREE_RATE / TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------


def hit_rate(trades: list[TradeResult]) -> float:
    """
    Fraction of trades with positive gross return.

    Args:
        trades: List of completed TradeResult objects.

    Returns:
        Hit rate in [0, 1].  Returns 0.0 for empty input.
    """
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t.gross_return > 0)
    return winners / len(trades)


def avg_return(trades: list[TradeResult]) -> float:
    """Mean gross return across all trades."""
    if not trades:
        return 0.0
    return float(np.mean([t.gross_return for t in trades]))


def avg_win(trades: list[TradeResult]) -> float:
    """Mean return of winning trades."""
    wins = [t.gross_return for t in trades if t.gross_return > 0]
    return float(np.mean(wins)) if wins else 0.0


def avg_loss(trades: list[TradeResult]) -> float:
    """Mean return of losing trades (negative number)."""
    losses = [t.gross_return for t in trades if t.gross_return <= 0]
    return float(np.mean(losses)) if losses else 0.0


def win_loss_ratio(trades: list[TradeResult]) -> float:
    """
    Ratio of average win to absolute average loss.

    > 1.0 means winners are larger than losers on average.

    Returns:
        Win/loss ratio.  Returns 0.0 if no losers.
    """
    aw = avg_win(trades)
    al = abs(avg_loss(trades))
    return aw / al if al > 0 else 0.0


def information_coefficient(trades: list[TradeResult]) -> float:
    """
    Spearman rank correlation between signal score and forward return.

    IC > 0 means the signal positively predicts returns.
    IC in [0.05, 0.10] is considered good for a fundamental signal.

    Args:
        trades: List of TradeResult with quality_score and gross_return.

    Returns:
        Spearman IC in [-1, +1].  Returns 0.0 if < 5 trades.
    """
    if len(trades) < 5:
        return 0.0

    scores = np.array([t.quality_score for t in trades])
    returns = np.array([t.gross_return for t in trades])

    # Rank both series
    def _rank(arr: np.ndarray) -> np.ndarray:
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks

    score_ranks  = _rank(scores)
    return_ranks = _rank(returns)

    n = len(trades)
    d_sq = np.sum((score_ranks - return_ranks) ** 2)
    ic = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))
    return float(np.clip(ic, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Equity-curve metrics
# ---------------------------------------------------------------------------


def build_equity_curve(
    trades: list[TradeResult],
    initial_capital: float = settings.INITIAL_CAPITAL,
) -> pd.Series:
    """
    Construct a daily equity curve from a list of completed trades.

    Each trade's return is applied on its exit date.  Between trades the
    equity value is held flat (uninvested cash earns the risk-free rate in
    a real implementation, but we keep it flat here for simplicity).

    Args:
        trades:          Completed trades sorted by exit_date.
        initial_capital: Starting portfolio value in USD.

    Returns:
        pd.Series with DatetimeIndex and equity values.
    """
    if not trades:
        return pd.Series(dtype=float)

    sorted_trades = sorted(trades, key=lambda t: t.exit_date)
    start = sorted_trades[0].entry_date
    end   = sorted_trades[-1].exit_date

    # Build daily date range
    dates = pd.date_range(start=start, end=end, freq="B")  # business days
    equity = pd.Series(index=dates, data=initial_capital, dtype=float)

    capital = initial_capital
    for trade in sorted_trades:
        exit_ts = pd.Timestamp(trade.exit_date)
        if exit_ts in equity.index:
            capital *= (1.0 + trade.gross_return)
            # Apply return forward from exit date
            equity.loc[exit_ts:] = capital

    return equity


def annualised_return(equity: pd.Series) -> float:
    """
    Compute CAGR from an equity curve.

    Args:
        equity: pd.Series with DatetimeIndex.

    Returns:
        Annualised return as a decimal (e.g. 0.15 = 15%).
    """
    if equity.empty or len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def annualised_sharpe(
    equity: pd.Series,
    risk_free_rate: float = settings.RISK_FREE_RATE,
) -> float:
    """
    Annualised Sharpe Ratio from an equity curve.

    Sharpe = (mean_daily_excess_return * 252) / (std_daily_return * sqrt(252))

    Args:
        equity:         pd.Series with DatetimeIndex.
        risk_free_rate: Annual risk-free rate (default from settings).

    Returns:
        Annualised Sharpe Ratio.  Returns 0.0 for < 20 data points.
    """
    if equity.empty or len(equity) < 20:
        return 0.0

    daily_returns = equity.pct_change().dropna()
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily

    std = excess.std()
    if std == 0 or math.isnan(std):
        return 0.0

    return float((excess.mean() / std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough percentage decline.

    Args:
        equity: pd.Series with DatetimeIndex.

    Returns:
        Max drawdown as a positive decimal (e.g. 0.25 = 25% drawdown).
    """
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(abs(drawdown.min()))


# ---------------------------------------------------------------------------
# Composite metrics bundle
# ---------------------------------------------------------------------------


def compute_all_metrics(
    trades: list[TradeResult],
    benchmark_equity: pd.Series | None = None,
    initial_capital: float = settings.INITIAL_CAPITAL,
) -> dict[str, Any]:
    """
    Compute the complete metrics bundle for the backtest.

    Metrics are computed for overall, long-only, and short-only slices.

    Args:
        trades:           All completed TradeResult objects.
        benchmark_equity: Optional buy-and-hold S&P 500 equity curve for
                          comparison (same DatetimeIndex as strategy).
        initial_capital:  Starting capital.

    Returns:
        Dict with keys: overall, long, short, benchmark (if provided).
        Each sub-dict contains: sharpe, hit_rate, avg_return, avg_win,
        avg_loss, win_loss_ratio, max_drawdown, annualised_return, ic,
        trade_count.
    """
    if not trades:
        logger.warning("compute_all_metrics called with empty trade list")
        return {}

    long_trades  = [t for t in trades if t.direction == SignalDirection.LONG]
    short_trades = [t for t in trades if t.direction == SignalDirection.SHORT]

    equity       = build_equity_curve(trades, initial_capital)
    long_equity  = build_equity_curve(long_trades, initial_capital)
    short_equity = build_equity_curve(short_trades, initial_capital)

    def _slice_metrics(slice_trades: list[TradeResult], eq: pd.Series) -> dict[str, Any]:
        return {
            "trade_count":      len(slice_trades),
            "sharpe":           round(annualised_sharpe(eq), 3),
            "hit_rate":         round(hit_rate(slice_trades), 4),
            "avg_return":       round(avg_return(slice_trades), 5),
            "avg_win":          round(avg_win(slice_trades), 5),
            "avg_loss":         round(avg_loss(slice_trades), 5),
            "win_loss_ratio":   round(win_loss_ratio(slice_trades), 3),
            "max_drawdown":     round(max_drawdown(eq), 4),
            "annualised_return": round(annualised_return(eq), 4),
            "ic":               round(information_coefficient(slice_trades), 4),
        }

    result: dict[str, Any] = {
        "overall": _slice_metrics(trades, equity),
        "long":    _slice_metrics(long_trades, long_equity),
        "short":   _slice_metrics(short_trades, short_equity),
    }

    if benchmark_equity is not None and not benchmark_equity.empty:
        result["benchmark"] = {
            "sharpe":            round(annualised_sharpe(benchmark_equity), 3),
            "max_drawdown":      round(max_drawdown(benchmark_equity), 4),
            "annualised_return": round(annualised_return(benchmark_equity), 4),
        }

    logger.info(
        f"Metrics computed: {len(trades)} trades  "
        f"sharpe={result['overall']['sharpe']}  "
        f"hit_rate={result['overall']['hit_rate']:.1%}  "
        f"ann_return={result['overall']['annualised_return']:.1%}"
    )
    return result


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """
    Format the metrics dict as a readable table string.

    Args:
        metrics: Output of compute_all_metrics().

    Returns:
        Multi-line string suitable for logging or terminal display.
    """
    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║         EarningsEdge Backtest Results            ║",
        "╠══════════════════════════════════════════════════╣",
    ]

    for slice_name in ("overall", "long", "short", "benchmark"):
        if slice_name not in metrics:
            continue
        m = metrics[slice_name]
        label = slice_name.upper()
        lines.append(f"║  {label:8s}                                       ║")
        for key, val in m.items():
            if isinstance(val, float):
                display = f"{val:.1%}" if "rate" in key or "return" in key or "drawdown" in key else f"{val:.3f}"
            else:
                display = str(val)
            lines.append(f"║    {key:25s}: {display:>10s}        ║")
        lines.append("╠══════════════════════════════════════════════════╣")

    lines[-1] = "╚══════════════════════════════════════════════════╝"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date as d

    # Build synthetic trades
    trades = [
        TradeResult(ticker="AAPL", direction=SignalDirection.LONG,
                    signal_date=d(2022,10,15), entry_date=d(2022,10,17), exit_date=d(2022,11,14),
                    entry_price=142.0, exit_price=155.0, quality_score=0.55, gross_return=0.0915, holding_days_actual=20),
        TradeResult(ticker="MSFT", direction=SignalDirection.LONG,
                    signal_date=d(2023,1,15), entry_date=d(2023,1,17), exit_date=d(2023,2,14),
                    entry_price=240.0, exit_price=250.0, quality_score=0.42, gross_return=0.0417, holding_days_actual=20),
        TradeResult(ticker="GOOG", direction=SignalDirection.SHORT,
                    signal_date=d(2023,4,15), entry_date=d(2023,4,17), exit_date=d(2023,5,16),
                    entry_price=108.0, exit_price=101.0, quality_score=-0.38, gross_return=0.0648, holding_days_actual=20),
        TradeResult(ticker="META", direction=SignalDirection.LONG,
                    signal_date=d(2023,7,15), entry_date=d(2023,7,17), exit_date=d(2023,8,14),
                    entry_price=290.0, exit_price=278.0, quality_score=0.35, gross_return=-0.0414, holding_days_actual=20),
        TradeResult(ticker="AMZN", direction=SignalDirection.SHORT,
                    signal_date=d(2023,10,15), entry_date=d(2023,10,17), exit_date=d(2023,11,14),
                    entry_price=130.0, exit_price=138.0, quality_score=-0.31, gross_return=-0.0615, holding_days_actual=20),
    ]

    metrics = compute_all_metrics(trades)

    overall = metrics["overall"]
    print(f"Trade count  : {overall['trade_count']}")
    print(f"Hit rate     : {overall['hit_rate']:.1%}")
    print(f"Avg return   : {overall['avg_return']:.2%}")
    print(f"Sharpe ratio : {overall['sharpe']:.3f}")
    print(f"Max drawdown : {overall['max_drawdown']:.1%}")
    print(f"IC           : {overall['ic']:.3f}")

    assert overall["trade_count"] == 5
    assert 0.0 <= overall["hit_rate"] <= 1.0
    assert -5.0 <= overall["sharpe"] <= 5.0
    assert 0.0 <= overall["max_drawdown"] <= 1.0

    print("\n" + format_metrics_table(metrics))
    print("\nmetrics smoke test passed ✓")
