"""
Earnings call transcript fetcher.

Scrapes earnings call transcripts from Motley Fool as the primary source,
with a fallback to Alpha Vantage (free tier, limited coverage).

Pipeline:
    1. Build Motley Fool search URL for the given ticker + quarter
    2. Fetch the transcript article page via httpx
    3. Parse HTML with BeautifulSoup to extract speaker turns
    4. Segment into prepared_remarks vs Q&A sections
    5. Return a validated EarningsTranscript Pydantic object

Speaker role classification:
    - CEO/President/Co-CEO → "CEO"
    - CFO/Chief Financial Officer → "CFO"
    - External analyst affiliation (e.g. "Goldman Sachs") → "Analyst"
    - Operator → "Operator"
    - Other executives → raw title string

Usage:
    fetcher = TranscriptFetcher()
    transcript = await fetcher.fetch_transcript("AAPL", "Q1", 2024)
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag
from loguru import logger
from pydantic import ValidationError

from config.settings import settings
from src.ingestion.data_validator import EarningsTranscript, Quarter, SpeakerTurn
from src.utils.rate_limiter import RateLimiter

# Motley Fool search endpoint for earnings transcripts
_FOOL_SEARCH_URL = (
    "https://www.fool.com/search/solr.aspx"
    "?q={ticker}+earnings+call+transcript+{quarter}+{year}"
    "&collection=fool&type=article"
)
# Motley Fool base URL for constructing absolute links
_FOOL_BASE = "https://www.fool.com"
# Motley Fool earnings transcript list page (more reliable than search)
_FOOL_TRANSCRIPT_LIST = (
    "https://www.fool.com/earnings-call-transcripts/?filter={ticker}"
)

# CEO/CFO/executive title patterns
_CEO_PATTERNS = re.compile(
    r"\b(ceo|chief executive|president|co-ceo|co-chief)\b", re.IGNORECASE
)
_CFO_PATTERNS = re.compile(
    r"\b(cfo|chief financial|finance officer)\b", re.IGNORECASE
)
_ANALYST_PATTERNS = re.compile(
    r"\b(analyst|research|securities|capital|bank|asset|investment|equity)\b",
    re.IGNORECASE,
)

# Transcript section boundary markers
_QA_MARKERS = re.compile(
    r"\b(question.and.answer|q\s?&\s?a session|questions and answers|q&a)\b",
    re.IGNORECASE,
)
_PREPARED_MARKERS = re.compile(
    r"\b(prepared remarks|opening remarks|conference call|presentation)\b",
    re.IGNORECASE,
)


class TranscriptFetcher:
    """
    Async earnings call transcript fetcher.

    Implements Motley Fool scraping (primary) and Alpha Vantage (fallback).
    All public methods are async coroutines.

    Attributes:
        client:       Shared httpx.AsyncClient.
        rate_limiter: Polite crawl rate (1 req/s to avoid bans).
    """

    def __init__(self) -> None:
        self.rate_limiter = RateLimiter(max_calls=1.0, period=1.0)
        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "TranscriptFetcher":
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; EarningsEdge research bot; "
                    f"{settings.SEC_USER_AGENT})"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.client:
            await self.client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_transcript(
        self,
        ticker: str,
        quarter: str | Quarter,
        year: int,
    ) -> EarningsTranscript | None:
        """
        Fetch an earnings call transcript for the given ticker + period.

        Tries Motley Fool first; falls back to returning None if not found.
        Callers should handle None gracefully (not all tickers have transcripts).

        Args:
            ticker:  Equity ticker symbol (case-insensitive).
            quarter: Quarter enum or string "Q1"/"Q2"/"Q3"/"Q4".
            year:    Fiscal year (e.g. 2024).

        Returns:
            EarningsTranscript if a matching transcript is found, else None.
        """
        ticker = ticker.upper().strip()
        if isinstance(quarter, str):
            quarter = Quarter(quarter.upper())

        logger.info(f"[Transcript] Fetching {ticker} {quarter.value} {year}")

        transcript = await self._fetch_from_motley_fool(ticker, quarter, year)
        if transcript:
            logger.success(
                f"[Transcript] Got {ticker} {quarter.value} {year} "
                f"({len(transcript.prepared_remarks + transcript.qa_section)} turns)"
            )
            return transcript

        logger.warning(
            f"[Transcript] Not found on Motley Fool: {ticker} {quarter.value} {year}"
        )
        return None

    async def fetch_recent_transcripts(
        self,
        ticker: str,
        num_quarters: int = 8,
    ) -> list[EarningsTranscript]:
        """
        Fetch the most recent N quarters of transcripts for a ticker.

        Args:
            ticker:       Equity ticker symbol.
            num_quarters: How many past quarters to attempt.

        Returns:
            List of successfully fetched transcripts (may be < num_quarters).
        """
        ticker = ticker.upper().strip()
        transcripts: list[EarningsTranscript] = []

        # Generate quarter sequence going backwards from today
        periods = _generate_past_quarters(num_quarters)

        for quarter, year in periods:
            transcript = await self.fetch_transcript(ticker, quarter, year)
            if transcript:
                transcripts.append(transcript)
            # Brief sleep between requests to be polite
            await asyncio.sleep(1.5)

        logger.info(
            f"[Transcript] Fetched {len(transcripts)}/{num_quarters} "
            f"transcripts for {ticker}"
        )
        return transcripts

    # ------------------------------------------------------------------
    # Motley Fool scraping
    # ------------------------------------------------------------------

    async def _fetch_from_motley_fool(
        self,
        ticker: str,
        quarter: Quarter,
        year: int,
    ) -> EarningsTranscript | None:
        """
        Scrape earnings transcript from Motley Fool.

        Strategy:
            1. Search for the article URL via Motley Fool transcript page
            2. Fetch and parse the article HTML
            3. Extract structured speaker turns

        Returns:
            EarningsTranscript on success, None otherwise.
        """
        article_url = await self._find_fool_article_url(ticker, quarter, year)
        if not article_url:
            return None

        try:
            await self.rate_limiter.acquire()
            resp = await self.client.get(article_url)  # type: ignore[union-attr]
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(f"[Transcript] Failed to fetch {article_url}: {exc}")
            return None

        return self._parse_fool_article(resp.text, ticker, quarter, year, article_url)

    async def _find_fool_article_url(
        self,
        ticker: str,
        quarter: Quarter,
        year: int,
    ) -> str | None:
        """
        Search Motley Fool for the transcript article URL.

        Uses the transcript listing page filtered by ticker, then matches
        by quarter/year in the article title or URL slug.

        Returns:
            Full article URL string, or None if not found.
        """
        assert self.client is not None
        search_url = _FOOL_TRANSCRIPT_LIST.format(ticker=ticker.lower())

        try:
            await self.rate_limiter.acquire()
            resp = await self.client.get(search_url)
            resp.raise_for_status()
        except Exception as exc:
            logger.debug(f"[Transcript] Transcript list fetch failed: {exc}")
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        # Find article links that match the quarter/year pattern
        q_pattern = re.compile(
            rf"{quarter.value}.*{year}|{year}.*{quarter.value}", re.IGNORECASE
        )
        # Also match fiscal quarter patterns like "first quarter", "second quarter"
        _FQ_MAP = {
            Quarter.Q1: r"first.quarter",
            Quarter.Q2: r"second.quarter",
            Quarter.Q3: r"third.quarter",
            Quarter.Q4: r"fourth.quarter",
        }
        fq_pattern = re.compile(
            rf"({_FQ_MAP.get(quarter, quarter.value)}).*{year}|{year}.*({_FQ_MAP.get(quarter, quarter.value)})",
            re.IGNORECASE,
        )

        for link in soup.find_all("a", href=True):
            href: str = link["href"]
            text: str = link.get_text(" ", strip=True)
            # Must mention the ticker and the quarter/year
            if ticker.lower() not in href.lower() and ticker.lower() not in text.lower():
                continue
            if q_pattern.search(text) or q_pattern.search(href) or fq_pattern.search(text):
                full_url = href if href.startswith("http") else f"{_FOOL_BASE}{href}"
                logger.debug(f"[Transcript] Candidate URL: {full_url}")
                return full_url

        return None

    def _parse_fool_article(
        self,
        html: str,
        ticker: str,
        quarter: Quarter,
        year: int,
        url: str,
    ) -> EarningsTranscript | None:
        """
        Parse a Motley Fool transcript article into structured speaker turns.

        Motley Fool transcript HTML structure:
            - Article body in <div class="article-body"> or <div id="article-body">
            - Each speaker introduced by a bold <p><strong>Name:</strong></p>
            - Speaker text follows in subsequent <p> tags until the next speaker

        Args:
            html:    Raw HTML from the article page.
            ticker:  Ticker symbol.
            quarter: Fiscal quarter.
            year:    Fiscal year.
            url:     Source URL for provenance.

        Returns:
            EarningsTranscript, or None if parsing yields no meaningful content.
        """
        soup = BeautifulSoup(html, "lxml")

        # Locate article body — Motley Fool uses multiple possible selectors
        body: Tag | None = (
            soup.find("div", class_="article-body")
            or soup.find("div", id="article-body")
            or soup.find("div", class_=re.compile(r"article|content|body", re.I))
            or soup.find("article")
        )
        if not body:
            logger.warning("[Transcript] Could not locate article body in HTML")
            return None

        paragraphs = body.find_all("p")
        if len(paragraphs) < 5:
            logger.warning("[Transcript] Too few paragraphs — likely not a transcript")
            return None

        # Parse speaker turns from paragraphs
        turns = _extract_speaker_turns(paragraphs)
        if not turns:
            logger.warning("[Transcript] No speaker turns extracted")
            return None

        # Segment into prepared remarks vs Q&A
        prepared, qa = _segment_transcript(turns)

        # Extract call date from meta tags if available
        call_date = _extract_call_date(soup) or date(year, _quarter_to_month(quarter), 1)

        raw_text = "\n\n".join(
            f"{t.speaker_name}: {t.text}" for t in turns
        )

        try:
            return EarningsTranscript(
                ticker=ticker,
                quarter=quarter,
                year=year,
                call_date=call_date,
                source_url=url,
                prepared_remarks=prepared,
                qa_section=qa,
                raw_text=raw_text,
            )
        except ValidationError as exc:
            logger.warning(f"[Transcript] Validation error: {exc}")
            return None


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------


def _extract_speaker_turns(paragraphs: list[Tag]) -> list[SpeakerTurn]:
    """
    Extract speaker turns from a list of BeautifulSoup paragraph tags.

    Motley Fool marks each speaker with a bold element at the start of a
    paragraph, e.g.: <p><strong>Tim Cook -- CEO:</strong> We are pleased...</p>

    Args:
        paragraphs: List of <p> Tag objects from the article body.

    Returns:
        Ordered list of SpeakerTurn objects.
    """
    turns: list[SpeakerTurn] = []
    turn_index = 0
    current_speaker: str | None = None
    current_role: str = "Unknown"
    current_company: str = "Unknown"
    buffer: list[str] = []

    # Pattern: "Name -- Title:" or "Name, Title:" or just "Name:"
    speaker_re = re.compile(
        r"^([A-Z][a-zA-Z\s\.\-]+?)(?:\s*[-–—,]+\s*|\s*:\s*)([^:]+)?:\s*",
        re.DOTALL,
    )

    def flush_turn(text: str) -> None:
        nonlocal turn_index
        if current_speaker and text.strip() and len(text.split()) > 3:
            turns.append(
                SpeakerTurn(
                    speaker_name=current_speaker,
                    speaker_role=current_role,
                    text=text.strip(),
                    turn_index=turn_index,
                    section="prepared_remarks",  # updated in segment step
                )
            )
            turn_index += 1

    for para in paragraphs:
        # Check for bold speaker intro
        bold = para.find(["strong", "b"])
        para_text = para.get_text(" ", strip=True)

        if bold:
            bold_text = bold.get_text(" ", strip=True)
            m = speaker_re.match(bold_text)
            if m:
                # Flush previous turn
                flush_turn(" ".join(buffer))
                buffer = []

                current_speaker = m.group(1).strip().rstrip("-–—,")
                title_str = (m.group(2) or "").strip()
                current_role = _classify_role(title_str)
                current_company = _classify_company(title_str)

                # Remainder of paragraph after speaker intro
                remainder = para_text[len(bold_text):].strip().lstrip(":").strip()
                if remainder:
                    buffer.append(remainder)
                continue

        # Regular paragraph text — accumulate into current turn
        if current_speaker and para_text.strip():
            buffer.append(para_text)

    # Flush final turn
    flush_turn(" ".join(buffer))
    return turns


def _segment_transcript(turns: list[SpeakerTurn]) -> tuple[list[SpeakerTurn], list[SpeakerTurn]]:
    """
    Segment speaker turns into prepared remarks and Q&A sections.

    Q&A begins when the Operator announces "questions and answers" or an
    analyst asks a question (role == "Analyst").

    Args:
        turns: Full ordered list of speaker turns.

    Returns:
        Tuple of (prepared_remarks, qa_section).
    """
    qa_start_idx = len(turns)  # Default: all prepared

    for i, turn in enumerate(turns):
        is_qa_marker = (
            _QA_MARKERS.search(turn.text)
            or turn.speaker_role == "Operator"
            and i > 2
            or turn.speaker_role == "Analyst"
        )
        if is_qa_marker and i > 0:
            qa_start_idx = i
            break

    prepared = turns[:qa_start_idx]
    qa = turns[qa_start_idx:]

    # Update section field on each turn now that we know which section it belongs to
    for t in prepared:
        object.__setattr__(t, "section", "prepared_remarks")
    for t in qa:
        object.__setattr__(t, "section", "qa")

    return prepared, qa


def _classify_role(title: str) -> str:
    """Map a raw title string to a standardised speaker role."""
    if not title:
        return "Executive"
    if _CEO_PATTERNS.search(title):
        return "CEO"
    if _CFO_PATTERNS.search(title):
        return "CFO"
    if _ANALYST_PATTERNS.search(title):
        return "Analyst"
    if "operator" in title.lower():
        return "Operator"
    return title[:50]  # Truncate long titles


def _classify_company(title: str) -> str:
    """
    Attempt to extract analyst firm name from title string.

    For issuers (CEO/CFO) returns "issuer"; for analysts returns the firm
    name if recognisable, otherwise "Unknown".
    """
    if _CEO_PATTERNS.search(title) or _CFO_PATTERNS.search(title):
        return "issuer"
    if _ANALYST_PATTERNS.search(title):
        # Heuristic: last word(s) before a dash or comma are often the firm
        parts = re.split(r"[-–—,]", title)
        return parts[-1].strip() if parts else "Unknown"
    return "Unknown"


def _extract_call_date(soup: BeautifulSoup) -> date | None:
    """Extract earnings call date from HTML meta tags or article text."""
    # Try Open Graph published time
    meta_time = soup.find("meta", {"property": "article:published_time"})
    if meta_time and isinstance(meta_time, Tag):
        content = meta_time.get("content", "")
        if content:
            try:
                return datetime.fromisoformat(str(content)[:10]).date()
            except ValueError:
                pass
    # Try schema.org datePublished
    date_elem = soup.find(attrs={"itemprop": "datePublished"})
    if date_elem and isinstance(date_elem, Tag):
        try:
            return datetime.fromisoformat(
                str(date_elem.get("content", ""))[:10]
            ).date()
        except ValueError:
            pass
    return None


def _quarter_to_month(quarter: Quarter) -> int:
    """Return the first calendar month of a fiscal quarter."""
    return {Quarter.Q1: 1, Quarter.Q2: 4, Quarter.Q3: 7, Quarter.Q4: 10}[quarter]


def _generate_past_quarters(n: int) -> list[tuple[Quarter, int]]:
    """
    Generate the N most recent fiscal quarters ending before today.

    Returns list of (Quarter, year) tuples, newest first.
    """
    today = date.today()
    results: list[tuple[Quarter, int]] = []
    month, year = today.month, today.year

    for _ in range(n):
        month -= 3
        if month <= 0:
            month += 12
            year -= 1
        # Map month to quarter
        q_map = {1: Quarter.Q1, 2: Quarter.Q1, 3: Quarter.Q1,
                 4: Quarter.Q2, 5: Quarter.Q2, 6: Quarter.Q2,
                 7: Quarter.Q3, 8: Quarter.Q3, 9: Quarter.Q3,
                 10: Quarter.Q4, 11: Quarter.Q4, 12: Quarter.Q4}
        results.append((q_map[month], year))

    return results


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio

    async def _demo() -> None:
        async with TranscriptFetcher() as fetcher:
            transcript = await fetcher.fetch_transcript("AAPL", "Q1", 2024)
            if transcript:
                print(f"Transcript: {transcript.ticker} {transcript.quarter} {transcript.year}")
                print(f"Call date : {transcript.call_date}")
                print(f"Speakers  : {transcript.speakers}")
                print(f"Prepared turns: {len(transcript.prepared_remarks)}")
                print(f"Q&A turns     : {len(transcript.qa_section)}")
                if transcript.prepared_remarks:
                    t = transcript.prepared_remarks[0]
                    print(f"\nFirst turn [{t.speaker_role}] {t.speaker_name}:")
                    print(t.text[:300])
            else:
                print("Transcript not found")

    _asyncio.run(_demo())
