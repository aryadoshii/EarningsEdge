"""
Document parser for SEC filings and earnings transcripts.

Handles two input formats:
    - HTML / plain text from EDGAR (most 10-K, 10-Q, 8-K filings post-1996)
    - PDF documents (older filings, some analyst reports)

Extraction pipeline:
    HTML  → BeautifulSoup → section-aware text blocks
    PDF   → pdfplumber   → page text → merged document

The parser does NOT chunk — it returns one clean string per filing.
Chunking is handled by chunker.py.

Usage:
    parser = DocumentParser()
    text = parser.parse_html(html_string)
    text = parser.parse_pdf(pdf_bytes)
    text = parser.parse_filing(filing)   # auto-detects format
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

from loguru import logger

from src.ingestion.data_validator import (
    EarningsTranscript,
    SECFiling,
    SectionType,
    SpeakerTurn,
)
from src.processing.text_cleaner import (
    clean_sec_text,
    clean_transcript_text,
    get_sec_section_pattern,
)

# Optional heavy imports — guarded so the module is importable even if
# pdfplumber is not installed in a lightweight test environment.
try:
    import pdfplumber  # type: ignore[import-untyped]
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False
    logger.warning("pdfplumber not installed — PDF parsing disabled")

try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed — HTML parsing degraded")


# ---------------------------------------------------------------------------
# Section detection patterns
# ---------------------------------------------------------------------------

# Maps Item numbers to our canonical SectionType enum values (10-K / 10-Q)
_SECTION_ORDER_10K: list[tuple[str, SectionType]] = [
    ("business",               SectionType.COVER),
    ("risk_factors",           SectionType.RISK_FACTORS),
    ("mda",                    SectionType.MDA),
    ("financial_statements",   SectionType.FINANCIAL_STATEMENTS),
    ("quantitative_disclosures", SectionType.MDA),   # grouped with MDA
    ("controls",               SectionType.MDA),
]

# Transcript speaker role classifier
_SPEAKER_ROLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:CEO|Chief Executive)\b",         re.I), "CEO"),
    (re.compile(r"\b(?:CFO|Chief Financial)\b",         re.I), "CFO"),
    (re.compile(r"\b(?:COO|Chief Operating)\b",         re.I), "COO"),
    (re.compile(r"\b(?:Analyst|Research)\b",            re.I), "Analyst"),
    (re.compile(r"\bOperator\b",                        re.I), "Operator"),
    (re.compile(r"\b(?:Investor Relations|IR)\b",       re.I), "IR"),
]

# Transcript Q&A section marker
_QA_START = re.compile(
    r"question.and.answer|q\s*[&and]+\s*a\s+session|q\s*[&and]+\s*a\s+portion",
    re.I,
)

# Speaker line: "Tim Cook -- CEO" or "JOHN SMITH, Goldman Sachs -- Analyst:"
_SPEAKER_LINE = re.compile(
    r"^([A-Z][A-Za-z\s\.\-\']{2,50})"   # name
    r"(?:\s*[-–—,]\s*[A-Za-z\s&,\.]+)?" # optional title/firm
    r"\s*[:\-–—]",                       # delimiter
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# DocumentParser
# ---------------------------------------------------------------------------


class DocumentParser:
    """
    Parses raw HTML, plain text, and PDF documents into clean strings.

    The class is stateless — all parse methods accept inputs and return
    outputs without modifying instance state.  Instantiate once and reuse.
    """

    # ------------------------------------------------------------------
    # HTML parsing
    # ------------------------------------------------------------------

    def parse_html(self, html: str) -> str:
        """
        Extract plain text from an HTML string.

        Uses BeautifulSoup if available; falls back to regex-based stripping.

        Args:
            html: Raw HTML string from EDGAR or a transcript site.

        Returns:
            Clean plain text.
        """
        if _BS4_AVAILABLE:
            soup = BeautifulSoup(html, "lxml")
            # Remove scripts, styles, and hidden elements
            for tag in soup(["script", "style", "head", "meta", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        else:
            # Regex fallback
            text = re.sub(r"<[^>]+>", " ", html)

        return clean_sec_text(text)

    # ------------------------------------------------------------------
    # PDF parsing
    # ------------------------------------------------------------------

    def parse_pdf(self, pdf_data: Union[bytes, Path]) -> str:
        """
        Extract text from a PDF file or bytes object.

        Args:
            pdf_data: Either raw PDF bytes or a Path to a PDF file.

        Returns:
            Concatenated plain text of all pages.

        Raises:
            RuntimeError: If pdfplumber is not installed.
        """
        if not _PDF_AVAILABLE:
            raise RuntimeError(
                "pdfplumber is required for PDF parsing. "
                "Run: uv add pdfplumber"
            )

        pages: list[str] = []

        if isinstance(pdf_data, Path):
            context = pdfplumber.open(pdf_data)
        else:
            context = pdfplumber.open(io.BytesIO(pdf_data))

        with context as pdf:
            logger.debug(f"Parsing PDF: {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if page_text:
                        pages.append(page_text)
                except Exception as exc:
                    logger.warning(f"Failed to extract page {i}: {exc}")

        raw = "\n\n".join(pages)
        return clean_sec_text(raw)

    # ------------------------------------------------------------------
    # Section-aware extraction for 10-K / 10-Q
    # ------------------------------------------------------------------

    def extract_sections(self, text: str) -> dict[SectionType, str]:
        """
        Split a 10-K or 10-Q text into named sections.

        Uses the Item-number patterns from text_cleaner to find section
        boundaries.  Sections run from their header to the next detected
        header.

        Args:
            text: Clean plain text of a 10-K or 10-Q filing.

        Returns:
            Dict mapping SectionType → section text.  Sections not found
            are omitted from the dict.
        """
        # Find all section header positions
        matches: list[tuple[int, SectionType]] = []
        for section_key, section_type in _SECTION_ORDER_10K:
            pattern = get_sec_section_pattern(section_key)
            for m in pattern.finditer(text):
                matches.append((m.start(), section_type))

        if not matches:
            logger.warning("No section headers detected — returning full text as MDA")
            return {SectionType.MDA: text}

        # Sort by position
        matches.sort(key=lambda x: x[0])

        sections: dict[SectionType, str] = {}
        for idx, (start_pos, sec_type) in enumerate(matches):
            end_pos = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
            section_text = text[start_pos:end_pos].strip()
            if len(section_text) > 200:  # skip near-empty sections
                # Merge if section type seen before (e.g. duplicate MDA entries)
                if sec_type in sections:
                    sections[sec_type] += "\n\n" + section_text
                else:
                    sections[sec_type] = section_text

        logger.debug(
            f"Sections extracted: {[s.value for s in sections]}"
        )
        return sections

    # ------------------------------------------------------------------
    # Transcript parsing
    # ------------------------------------------------------------------

    def parse_transcript(
        self,
        raw_text: str,
        ticker: str = "",
    ) -> tuple[list[SpeakerTurn], list[SpeakerTurn]]:
        """
        Parse a raw earnings call transcript into structured speaker turns.

        Splits the document at the Q&A marker into two sections:
        prepared_remarks and qa_section.  Within each section, identifies
        individual speaker turns using the SPEAKER_LINE pattern.

        Args:
            raw_text: Cleaned transcript text.
            ticker:   Ticker symbol (used only for logging).

        Returns:
            Tuple of (prepared_remarks, qa_section) — each is a list of
            SpeakerTurn objects.
        """
        text = clean_transcript_text(raw_text)

        # Split at Q&A start
        qa_match = _QA_START.search(text)
        if qa_match:
            prepared_raw = text[: qa_match.start()]
            qa_raw = text[qa_match.end() :]
            logger.debug(f"[{ticker}] Q&A section found at char {qa_match.start()}")
        else:
            prepared_raw = text
            qa_raw = ""
            logger.debug(f"[{ticker}] No explicit Q&A marker found")

        prepared_turns = self._extract_speaker_turns(prepared_raw, "prepared_remarks")
        qa_turns = self._extract_speaker_turns(qa_raw, "qa")

        return prepared_turns, qa_turns

    def _extract_speaker_turns(
        self,
        text: str,
        section: str,
    ) -> list[SpeakerTurn]:
        """
        Extract individual speaker turns from a transcript section.

        Args:
            text:    Text of one transcript section.
            section: 'prepared_remarks' or 'qa'.

        Returns:
            List of SpeakerTurn objects in document order.
        """
        turns: list[SpeakerTurn] = []
        if not text.strip():
            return turns

        # Find all speaker boundaries
        boundaries: list[tuple[int, str]] = []
        for m in _SPEAKER_LINE.finditer(text):
            raw_name = m.group(1).strip()
            # Filter out false positives (very short matches, numbers)
            if len(raw_name) >= 3 and not raw_name.isdigit():
                boundaries.append((m.start(), raw_name))

        if not boundaries:
            # No speaker turns found — treat whole section as one unknown turn
            turns.append(
                SpeakerTurn(
                    speaker_name="Unknown",
                    speaker_role="unknown",
                    text=text.strip(),
                    turn_index=0,
                    section=section,
                )
            )
            return turns

        for idx, (start, speaker_name) in enumerate(boundaries):
            end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
            turn_text = text[start:end].strip()
            # Remove the speaker header from the turn text
            turn_text = re.sub(r"^[^\n]+\n", "", turn_text, count=1).strip()

            if not turn_text or len(turn_text) < 10:
                continue

            role = self._classify_speaker_role(speaker_name)
            turns.append(
                SpeakerTurn(
                    speaker_name=speaker_name,
                    speaker_role=role,
                    text=turn_text,
                    turn_index=len(turns),
                    section=section,
                )
            )

        return turns

    @staticmethod
    def _classify_speaker_role(name_and_title: str) -> str:
        """
        Infer speaker role from name/title string.

        Args:
            name_and_title: Raw speaker label from transcript.

        Returns:
            Role string: CEO / CFO / COO / Analyst / Operator / IR / unknown.
        """
        for pattern, role in _SPEAKER_ROLE_PATTERNS:
            if pattern.search(name_and_title):
                return role
        return "unknown"

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def parse_filing(self, filing: SECFiling) -> str:
        """
        Auto-detect format and parse a SECFiling object.

        If raw_text is already populated (e.g. downloaded as HTML), cleans
        and returns it.  Handles both HTML and plain text gracefully.

        Args:
            filing: Populated SECFiling object with raw_text set.

        Returns:
            Clean plain text of the filing.
        """
        raw = filing.raw_text
        if not raw:
            logger.warning(
                f"[{filing.ticker}] Empty raw_text in "
                f"{filing.filing_type.value} {filing.period_of_report}"
            )
            return ""

        # Heuristic: if text contains multiple HTML tags it's HTML
        if raw.count("<") > 20 and raw.count(">") > 20:
            return self.parse_html(raw)
        return clean_sec_text(raw)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

document_parser = DocumentParser()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = DocumentParser()

    # -- HTML test
    sample_html = """
    <html><body>
    <script>alert('x')</script>
    <h2>Item 1A. Risk Factors</h2>
    <p>We operate in highly competitive markets. Competition may reduce
    our market share. We expect revenue of <b>$90 billion</b> next quarter.</p>
    <h2>Item 7. Management&#8217;s Discussion and Analysis</h2>
    <p>Net revenue increased 6% year over year to $94.9 billion, driven
    by strong iPhone and Services performance.</p>
    </body></html>
    """
    text = parser.parse_html(sample_html)
    print("=== Parsed HTML ===")
    print(text[:400])

    # -- Section extraction test
    sections = parser.extract_sections(text)
    print(f"\nSections found: {list(sections.keys())}")

    # -- Transcript test
    sample_transcript = """
    Tim Cook -- CEO: Good morning, everyone. We are very pleased to report
    record revenue of $94.9 billion for the quarter, up 6% year over year.

    Luca Maestri -- CFO: Thank you, Tim. Our gross margin was 46.2%, and
    we generated operating cash flow of $29.9 billion.

    Question-and-Answer Session

    Mike Ng -- Goldman Sachs -- Analyst: Congratulations on the results.
    Can you comment on iPhone demand trends heading into Q2?

    Tim Cook -- CEO: Thanks, Mike. iPhone demand remains very robust.
    """
    prepared, qa = parser.parse_transcript(sample_transcript, ticker="AAPL")
    print(f"\nPrepared remarks: {len(prepared)} turns")
    for turn in prepared:
        print(f"  [{turn.speaker_role:8s}] {turn.speaker_name}: {turn.text[:60]}...")
    print(f"Q&A section: {len(qa)} turns")
    print("\ndocument_parser smoke test passed ✓")
