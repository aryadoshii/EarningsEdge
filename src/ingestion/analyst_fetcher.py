"""
Analyst consensus and revision data fetcher.

Pulls analyst estimates, price targets, and revision history for a given
ticker using yfinance (free, no API key required).

Data collected:
    - EPS estimates (current/next quarter and full year)
    - Revenue estimates
    - Consensus price targets (mean, high, low)
    - Number of analysts
    - Revision direction score normalised to [-1, +1]

Usage:
    fetcher = AnalystFetcher()
    data = await fetcher.fetch("AAPL")
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from src.ingestion.data_validator import AnalystData, AnalystEstimate
from src.utils.rate_limiter import RateLimiter


class AnalystFetcher:
    """Async analyst data fetcher using yfinance."""

    def __init__(self) -> None:
        self.rate_limiter = RateLimiter(max_calls=2.0, period=1.0)

    async def fetch(self, ticker: str) -> AnalystData:
        """
        Fetch analyst consensus data for a ticker.

        Args:
            ticker: Equity ticker symbol.

        Returns:
            Populated AnalystData model. Fields are None if unavailable.
        """
        ticker = ticker.upper().strip()
        logger.info(f"[Analyst] Fetching analyst data for {ticker}")

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._fetch_yfinance, ticker)

        return self._build_model(ticker, raw)

    async def fetch_batch(self, tickers: list[str]) -> dict[str, AnalystData]:
        """Fetch analyst data for multiple tickers."""
        results: dict[str, AnalystData] = {}
        for ticker in tickers:
            results[ticker] = await self.fetch(ticker)
            await asyncio.sleep(0.5)
        return results

    # ------------------------------------------------------------------
    # yfinance (synchronous — run in executor)
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, ticker: str) -> dict[str, Any]:
        """Pull raw analyst data from yfinance. Always returns a dict."""
        out: dict[str, Any] = {
            "price_target_mean": None,
            "price_target_high": None,
            "price_target_low":  None,
            "num_analysts":      None,
            "current_rating":    None,
            "eps_estimates":     {},
            "rev_estimates":     {},
            "upgrades":          0,
            "downgrades":        0,
        }

        try:
            tk = yf.Ticker(ticker)

            # ── Price targets ──────────────────────────────────────────
            try:
                pt = tk.analyst_price_targets
                # yfinance may return a dict or DataFrame depending on version
                if isinstance(pt, dict):
                    out["price_target_mean"] = _safe_float(pt.get("mean"))
                    out["price_target_high"] = _safe_float(pt.get("high"))
                    out["price_target_low"]  = _safe_float(pt.get("low"))
                    out["num_analysts"]      = _safe_int(pt.get("numberOfAnalysts"))
                elif isinstance(pt, pd.DataFrame) and not pt.empty:
                    out["price_target_mean"] = _safe_float(pt.get("mean"))
                    out["price_target_high"] = _safe_float(pt.get("high"))
                    out["price_target_low"]  = _safe_float(pt.get("low"))
            except Exception as exc:
                logger.debug(f"[Analyst] Price targets error for {ticker}: {exc}")

            # ── EPS estimates ──────────────────────────────────────────
            try:
                eps_df = tk.earnings_estimate
                if eps_df is not None and not eps_df.empty:
                    out["eps_estimates"] = _parse_estimates_df(eps_df, "avg")
            except Exception as exc:
                logger.debug(f"[Analyst] EPS estimates error for {ticker}: {exc}")

            # ── Revenue estimates ──────────────────────────────────────
            try:
                rev_df = tk.revenue_estimate
                if rev_df is not None and not rev_df.empty:
                    out["rev_estimates"] = _parse_estimates_df(rev_df, "avg")
            except Exception as exc:
                logger.debug(f"[Analyst] Revenue estimates error for {ticker}: {exc}")

            # ── Upgrades / downgrades ──────────────────────────────────
            try:
                updown = tk.upgrades_downgrades
                if updown is not None and not updown.empty:
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
                    recent = updown[updown.index >= cutoff]
                    upgrades, downgrades = _count_revision_moves(
                        recent.to_dict("records")
                    )
                    out["upgrades"]   = upgrades
                    out["downgrades"] = downgrades
            except Exception as exc:
                logger.debug(f"[Analyst] Upgrades/downgrades error for {ticker}: {exc}")

            # ── Consensus rating ───────────────────────────────────────
            try:
                rec = tk.recommendations
                if rec is not None and not rec.empty:
                    latest = rec.iloc[-1]
                    out["current_rating"] = str(
                        latest.get("To Grade") or latest.get("toGrade") or ""
                    ) or None
            except Exception:
                pass

        except Exception as exc:
            logger.warning(f"[Analyst] yfinance fetch failed for {ticker}: {exc}")

        return out

    def _build_model(self, ticker: str, raw: dict[str, Any]) -> AnalystData:
        """
        Convert raw yfinance data into a validated AnalystData model.

        Maps yfinance fields to the correct AnalystData schema.
        """
        # Build AnalystEstimate list from EPS + revenue dicts
        estimates: list[AnalystEstimate] = []
        eps = raw.get("eps_estimates", {})
        rev = raw.get("rev_estimates", {})

        period_map = {
            "current_quarter": "0q",
            "next_quarter":    "+1q",
            "current_year":    "0y",
            "next_year":       "+1y",
        }
        for key, period_label in period_map.items():
            eps_val = eps.get(key)
            rev_val = rev.get(key)
            if eps_val is not None or rev_val is not None:
                estimates.append(AnalystEstimate(
                    period=period_label,
                    eps_mean=eps_val,
                    revenue_mean=rev_val,
                ))

        # Revision direction score: (upgrades - downgrades) / (total + 1)
        upgrades   = int(raw.get("upgrades",   0) or 0)
        downgrades = int(raw.get("downgrades", 0) or 0)
        total = upgrades + downgrades
        revision_direction = float((upgrades - downgrades) / (total + 1)) if total > 0 else 0.0
        # Clamp to [-1, +1]
        revision_direction = max(-1.0, min(1.0, revision_direction))

        return AnalystData(
            ticker=ticker,
            fetch_date=date.today(),                          # ── FIXED: required field
            price_target_mean=raw.get("price_target_mean"),
            price_target_high=raw.get("price_target_high"),
            price_target_low=raw.get("price_target_low"),
            num_price_target_analysts=raw.get("num_analysts"),  # ── FIXED: correct field name
            current_rating=raw.get("current_rating"),
            estimates=estimates,
            revision_direction=revision_direction,              # ── FIXED: always a float
        )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _parse_estimates_df(df: pd.DataFrame, value_col: str) -> dict[str, Any]:
    """Convert yfinance estimates DataFrame to a keyed dict."""
    mapping = {
        "0q":  "current_quarter",
        "+1q": "next_quarter",
        "0y":  "current_year",
        "+1y": "next_year",
    }
    result: dict[str, Any] = {}
    if value_col not in df.columns:
        return result
    for idx_label, key in mapping.items():
        if idx_label in df.index:
            val = df.loc[idx_label, value_col]
            result[key] = _safe_float(val)
    return result


def _count_revision_moves(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count upgrades and downgrades from yfinance records."""
    upgrades = downgrades = 0
    up_words   = {"upgrade", "initiated", "raised", "buy", "outperform", "overweight"}
    down_words = {"downgrade", "lowered", "sell", "underperform", "underweight"}

    for rec in records:
        action   = str(rec.get("Action", "") or "").lower()
        to_grade = str(rec.get("ToGrade", "") or rec.get("toGrade", "") or "").lower()
        text = action + " " + to_grade
        if any(w in text for w in up_words):
            upgrades += 1
        elif any(w in text for w in down_words):
            downgrades += 1

    return upgrades, downgrades


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value).strip().replace(",", "")) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import asyncio as _asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()

    async def _demo() -> None:
        fetcher = AnalystFetcher()
        data = await fetcher.fetch(args.ticker)
        print(f"Ticker           : {data.ticker}")
        print(f"Fetch date       : {data.fetch_date}")
        print(f"Price target mean: ${data.price_target_mean}")
        print(f"Num analysts     : {data.num_price_target_analysts}")
        print(f"Current rating   : {data.current_rating}")
        print(f"Revision dir     : {data.revision_direction:+.3f}")
        print(f"Estimates        : {len(data.estimates)} periods")
        for e in data.estimates:
            print(f"  {e.period:4s}  EPS={e.eps_mean}  Rev={e.revenue_mean}")

    _asyncio.run(_demo())