"""
SEC EDGAR filing fetcher.

Downloads 10-K, 10-Q, and 8-K filings for a given ticker using the public
EDGAR REST API (no key required).

Usage:
    python -m src.ingestion.sec_fetcher --ticker AAPL
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Any

import httpx
from loguru import logger
from pydantic import ValidationError

from config.settings import settings
from src.ingestion.data_validator import FilingType, Quarter, SECFiling, XBRLData
from src.utils.rate_limiter import RateLimiter

_TICKER_TO_CIK_URL  = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL    = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_FACTS_URL  = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# ── FIXED: use correct FilingType enum values ──────────────────────────────
_FORM_MAP: dict[str, FilingType] = {
    "10-K":   FilingType.FORM_10K,
    "10-K/A": FilingType.FORM_10K_A,
    "10-Q":   FilingType.FORM_10Q,
    "10-Q/A": FilingType.FORM_10Q_A,
    "8-K":    FilingType.FORM_8K,
}

_XBRL_CONCEPTS: dict[str, str] = {
    "revenue":             "Revenues",
    "net_income":          "NetIncomeLoss",
    "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
    "total_assets":        "Assets",
    "capex":               "PaymentsToAcquirePropertyPlantAndEquipment",
    "eps_diluted":         "EarningsPerShareDiluted",
    "eps_basic":           "EarningsPerShareBasic",
}


class SECFetcher:
    """Async EDGAR filing fetcher with XBRL data extraction."""

    def __init__(self) -> None:
        self.rate_limiter = RateLimiter(max_calls=settings.SEC_RATE_LIMIT_RPS, period=1.0)
        self._cik_cache: dict[str, str] = {}
        self._ticker_map: dict[str, str] = {}
        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SECFetcher":
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": settings.SEC_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json, text/html, */*",
            },
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.client:
            await self.client.aclose()

    async def fetch_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        max_filings: int = 20,
    ) -> list[SECFiling]:
        """Fetch SEC filings for a ticker. Returns list of SECFiling objects."""
        ticker = ticker.upper().strip()
        form_types = form_types or ["10-K", "10-Q", "8-K"]
        logger.info(f"[SEC] Fetching {form_types} for {ticker}")

        cik = await self._resolve_cik(ticker)
        stubs = await self._get_filing_stubs(cik, form_types, start_date, end_date)
        logger.info(f"[SEC] Found {len(stubs)} filing stubs for {ticker}")

        xbrl_facts = await self._fetch_xbrl_facts(cik)

        results: list[SECFiling] = []
        for stub in stubs[:max_filings]:
            try:
                filing = await self._hydrate_filing(ticker, cik, stub, xbrl_facts)
                if filing:
                    results.append(filing)
            except Exception as exc:
                logger.warning(f"[SEC] Skipping {stub.get('accessionNumber')}: {exc}")

        logger.success(f"[SEC] Fetched {len(results)} filings for {ticker}")
        return results

    async def resolve_cik(self, ticker: str) -> str:
        return await self._resolve_cik(ticker.upper())

    async def _get(self, url: str) -> httpx.Response:
        assert self.client is not None
        for attempt in range(1, 4):
            await self.rate_limiter.acquire()
            try:
                r = await self.client.get(url)
                r.raise_for_status()
                return r
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise
                if exc.response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                elif attempt == 3:
                    raise
                else:
                    await asyncio.sleep(2 ** attempt)
            except httpx.TransportError:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"All attempts failed: {url}")

    async def _resolve_cik(self, ticker: str) -> str:
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        if not self._ticker_map:
            logger.debug("[SEC] Loading ticker→CIK map from EDGAR")
            resp = await self._get(_TICKER_TO_CIK_URL)
            raw: dict[str, dict[str, Any]] = resp.json()
            self._ticker_map = {
                v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                for v in raw.values()
            }
        cik = self._ticker_map.get(ticker)
        if not cik:
            raise ValueError(f"No CIK for '{ticker}'")
        self._cik_cache[ticker] = cik
        logger.debug(f"[SEC] Resolved {ticker} → CIK {cik}")
        return cik

    async def _get_filing_stubs(
        self,
        cik: str,
        form_types: list[str],
        start_date: date | None,
        end_date: date | None,
    ) -> list[dict[str, Any]]:
        resp = await self._get(_SUBMISSIONS_URL.format(cik=cik))
        data: dict[str, Any] = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        keys = list(recent.keys())
        n = len(recent.get("form", []))
        stubs: list[dict[str, Any]] = []

        for i in range(n):
            stub = {k: recent[k][i] for k in keys if i < len(recent[k])}
            if stub.get("form", "") not in form_types:
                continue
            filed_str = stub.get("filingDate", "")
            if filed_str:
                try:
                    filed = date.fromisoformat(filed_str)
                    if start_date and filed < start_date:
                        continue
                    if end_date and filed > end_date:
                        continue
                except ValueError:
                    pass
            stubs.append(stub)

        stubs.sort(key=lambda s: s.get("filingDate", ""), reverse=True)
        return stubs

    async def _fetch_xbrl_facts(self, cik: str) -> dict[str, Any]:
        try:
            resp = await self._get(_COMPANY_FACTS_URL.format(cik=cik))
            facts: dict[str, Any] = resp.json()
            return facts.get("facts", {}).get("us-gaap", {})
        except Exception as exc:
            logger.warning(f"[SEC] XBRL unavailable for CIK {cik}: {exc}")
            return {}

    async def _hydrate_filing(
        self,
        ticker: str,
        cik: str,
        stub: dict[str, Any],
        xbrl_facts: dict[str, Any],
    ) -> SECFiling | None:
        form = stub.get("form", "")
        accession_raw = stub.get("accessionNumber", "")
        primary_doc   = stub.get("primaryDocument", "")
        filing_date_str = stub.get("filingDate", "")
        report_date_str = stub.get("reportDate", "")

        if not accession_raw or not primary_doc:
            return None

        accession_clean = accession_raw.replace("-", "")
        cik_int = str(int(cik))
        doc_url = f"{_EDGAR_ARCHIVES_BASE}/{cik_int}/{accession_clean}/{primary_doc}"

        raw_text = await self._download_filing_text(doc_url)

        try:
            filed_date = date.fromisoformat(filing_date_str)
        except ValueError:
            filed_date = date.today()

        try:
            period_of_report = date.fromisoformat(report_date_str)
        except ValueError:
            period_of_report = filed_date

        # ── FIXED: use correct field names 'quarter' and 'year' ──────────
        quarter = _infer_quarter(period_of_report, form)
        xbrl_data = _extract_xbrl_for_period(xbrl_facts, period_of_report)
        filing_type = _FORM_MAP.get(form, FilingType.FORM_8K)

        try:
            return SECFiling(
                ticker=ticker,
                cik=cik,
                filing_type=filing_type,
                period_of_report=period_of_report,
                filed_date=filed_date,
                quarter=quarter,
                year=period_of_report.year,
                filing_url=doc_url,
                accession_number=accession_raw,
                raw_text=raw_text,
                word_count=len(raw_text.split()) if raw_text else 0,
                xbrl_data=xbrl_data,
                is_amendment="/A" in form,
            )
        except ValidationError as exc:
            logger.warning(f"[SEC] Validation failed {ticker} {form}: {exc}")
            return None

    async def _download_filing_text(self, url: str) -> str:
        try:
            resp = await self._get(url)
            raw = resp.text
            ct = resp.headers.get("content-type", "")
            if "html" in ct.lower() or raw.strip().startswith("<"):
                return _strip_html(raw)
            return raw
        except Exception as exc:
            logger.warning(f"[SEC] Download failed {url}: {exc}")
            return ""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    text = re.sub(r"<ix:[^>]+>.*?</ix:[^>]+>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&nbsp;", " ").replace("&#160;", " ").replace("&quot;", '"'))
    return re.sub(r"\s+", " ", text).strip()


def _infer_quarter(report_date: date, form: str) -> Quarter | None:
    """Map period-end date to fiscal quarter. Returns None for 10-K."""
    if "10-K" in form:
        return None
    m = report_date.month
    if m <= 3:   return Quarter.Q1
    if m <= 6:   return Quarter.Q2
    if m <= 9:   return Quarter.Q3
    return Quarter.Q4


def _extract_xbrl_for_period(us_gaap_facts: dict[str, Any], period: date) -> XBRLData:
    """Extract XBRL financials for the given period end date."""
    period_str = period.isoformat()
    values: dict[str, float | None] = {}

    for field_name, concept in _XBRL_CONCEPTS.items():
        concept_data = us_gaap_facts.get(concept, {})
        units_data: dict[str, list[dict[str, Any]]] = concept_data.get("units", {})

        unit_key = "USD/shares" if "PerShare" in concept else "USD"
        if unit_key not in units_data:
            unit_key = next(iter(units_data), None)  # type: ignore[assignment]

        if unit_key is None:
            values[field_name] = None
            continue

        entries: list[dict[str, Any]] = units_data[unit_key]
        match = next(
            (e for e in entries if e.get("end") == period_str
             and e.get("form", "").startswith("10-")),
            None,
        )
        if match is None:
            match = _find_closest_xbrl_entry(entries, period)

        if match and "val" in match:
            values[field_name] = float(match["val"])
        else:
            values[field_name] = None

    return XBRLData(**values)


def _find_closest_xbrl_entry(
    entries: list[dict[str, Any]],
    target: date,
    max_days: int = 10,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_delta = max_days + 1
    for entry in entries:
        try:
            end = date.fromisoformat(entry.get("end", ""))
            delta = abs((end - target).days)
            if delta < best_delta:
                best_delta = delta
                best = entry
        except ValueError:
            continue
    return best if best_delta <= max_days else None


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
        async with SECFetcher() as fetcher:
            cik = await fetcher.resolve_cik(args.ticker)
            print(f"{args.ticker} CIK: {cik}")
            filings = await fetcher.fetch_filings(args.ticker, form_types=["10-K"], max_filings=2)
            for f in filings:
                ni  = f.xbrl_data.net_income
                ocf = f.xbrl_data.operating_cash_flow
                ta  = f.xbrl_data.total_assets
                accruals = (
                    round((ni - ocf) / ta, 4)
                    if ni is not None and ocf is not None and ta
                    else "N/A"
                )
                print(
                    f"  {f.filing_type.value} | Period: {f.period_of_report} | "
                    f"Words: {f.word_count:,} | NI: {ni} | OCF: {ocf} | "
                    f"Accruals ratio: {accruals}"
                )

    _asyncio.run(_demo())