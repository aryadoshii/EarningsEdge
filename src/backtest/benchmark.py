"""
Benchmark comparison utilities.

Compares the EarningsEdge strategy equity curve against an S&P 500
buy-and-hold benchmark (SPY ETF) over the same time period.

Outputs:
    excess_return   — strategy annualised return minus benchmark
    information_ratio — excess return / tracking error
    beta            — strategy correlation-adjusted sensitivity to market
    alpha           — Jensen's alpha (annualised)
    up_capture      — % of benchmark up-days captured by strategy
    down_capture    — % of benchmark down-days captured (lower = better)

Usage:
    from src.backtest.benchmark import compute_benchmark_comparison
    comparison = compute_benchmark_comparison(strategy_equity, benchmark_equity)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualised_return,
    annualised_sharpe,
    max_drawdown,
)


def _align_series(
    strategy: pd.Series,
    benchmark: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Align two equity series to the same DatetimeIndex.

    Resamples to business-day frequency and forward-fills gaps.

    Args:
        strategy:  Strategy equity curve.
        benchmark: Benchmark equity curve.

    Returns:
        Tuple of (aligned_strategy, aligned_benchmark).
    """
    combined = pd.concat([strategy, benchmark], axis=1, join="inner")
    combined = combined.ffill()
    return combined.iloc[:, 0], combined.iloc[:, 1]


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def compute_benchmark_comparison(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    risk_free_rate: float = settings.RISK_FREE_RATE,
) -> dict[str, Any]:
    """
    Compute a full set of benchmark-relative performance metrics.

    Args:
        strategy_equity:  Daily equity curve of the EarningsEdge strategy.
        benchmark_equity: Daily equity curve of the benchmark (SPY).
        risk_free_rate:   Annual risk-free rate.

    Returns:
        Dict with benchmark-relative stats.
    """
    if strategy_equity.empty or benchmark_equity.empty:
        logger.warning("Empty equity series — returning empty comparison")
        return {}

    strat, bench = _align_series(strategy_equity, benchmark_equity)

    if len(strat) < 20:
        logger.warning("Insufficient data for benchmark comparison")
        return {}

    strat_ret  = _daily_returns(strat)
    bench_ret  = _daily_returns(bench)

    # ── Basic relative metrics ────────────────────────────────────────
    strat_ann_return = annualised_return(strat)
    bench_ann_return = annualised_return(bench)
    excess_return    = strat_ann_return - bench_ann_return

    # ── Tracking error + information ratio ──────────────────────────
    diff_returns  = strat_ret.values - bench_ret.values
    tracking_err  = float(np.std(diff_returns) * math.sqrt(TRADING_DAYS_PER_YEAR))
    info_ratio    = excess_return / tracking_err if tracking_err > 0 else 0.0

    # ── Beta + alpha (CAPM) ──────────────────────────────────────────
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_strat  = strat_ret - rf_daily
    excess_bench  = bench_ret - rf_daily

    cov_matrix = np.cov(excess_strat.values, excess_bench.values)
    bench_var  = cov_matrix[1, 1]
    beta       = float(cov_matrix[0, 1] / bench_var) if bench_var > 0 else 1.0
    alpha_daily = float(np.mean(excess_strat.values) - beta * np.mean(excess_bench.values))
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR

    # ── Up / Down capture ────────────────────────────────────────────
    up_days   = bench_ret > 0
    down_days = bench_ret < 0

    up_capture   = 0.0
    down_capture = 0.0

    if up_days.sum() > 0:
        up_capture = float(
            strat_ret[up_days].mean() / bench_ret[up_days].mean()
        ) if bench_ret[up_days].mean() != 0 else 0.0

    if down_days.sum() > 0:
        down_capture = float(
            strat_ret[down_days].mean() / bench_ret[down_days].mean()
        ) if bench_ret[down_days].mean() != 0 else 0.0

    comparison = {
        # Absolute metrics
        "strategy_ann_return":  round(strat_ann_return, 4),
        "benchmark_ann_return": round(bench_ann_return, 4),
        "strategy_sharpe":      round(annualised_sharpe(strat), 3),
        "benchmark_sharpe":     round(annualised_sharpe(bench), 3),
        "strategy_max_dd":      round(max_drawdown(strat), 4),
        "benchmark_max_dd":     round(max_drawdown(bench), 4),
        # Relative metrics
        "excess_return":        round(excess_return, 4),
        "tracking_error":       round(tracking_err, 4),
        "information_ratio":    round(info_ratio, 3),
        "beta":                 round(beta, 3),
        "alpha_annual":         round(alpha_annual, 4),
        "up_capture":           round(up_capture, 3),
        "down_capture":         round(down_capture, 3),
    }

    logger.info(
        f"Benchmark comparison: excess_return={excess_return:.1%}  "
        f"IR={info_ratio:.3f}  alpha={alpha_annual:.1%}  beta={beta:.3f}"
    )
    return comparison


def format_comparison_table(comparison: dict[str, Any]) -> str:
    """
    Format benchmark comparison as a readable two-column table.

    Args:
        comparison: Output of compute_benchmark_comparison().

    Returns:
        Formatted string.
    """
    pct_keys = {
        "strategy_ann_return", "benchmark_ann_return",
        "strategy_max_dd", "benchmark_max_dd",
        "excess_return", "tracking_error", "alpha_annual",
    }
    lines = ["Benchmark Comparison (vs SPY buy-and-hold)", "-" * 46]
    for k, v in comparison.items():
        if isinstance(v, float):
            display = f"{v:.1%}" if k in pct_keys else f"{v:.3f}"
        else:
            display = str(v)
        lines.append(f"  {k:30s}: {display}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Synthetic equity curves
    np.random.seed(42)
    dates = pd.date_range("2022-01-03", periods=504, freq="B")

    # Strategy: 12% annual with 15% vol
    strat_returns = np.random.normal(0.12 / 252, 0.15 / np.sqrt(252), 504)
    strat_equity  = pd.Series(
        100_000 * np.cumprod(1 + strat_returns),
        index=dates,
    )

    # Benchmark: 10% annual with 18% vol
    bench_returns = np.random.normal(0.10 / 252, 0.18 / np.sqrt(252), 504)
    bench_equity  = pd.Series(
        100_000 * np.cumprod(1 + bench_returns),
        index=dates,
    )

    comparison = compute_benchmark_comparison(strat_equity, bench_equity)
    print(format_comparison_table(comparison))

    assert "excess_return" in comparison
    assert "information_ratio" in comparison
    assert "beta" in comparison
    assert "alpha_annual" in comparison
    assert "up_capture" in comparison
    assert "down_capture" in comparison

    print("\nbenchmark comparison smoke test passed ✓")
