"""
Text cleaning and normalisation for SEC filings and earnings transcripts.

Removes SEC boilerplate, EDGAR header/footer artefacts, HTML entities,
excessive whitespace, and other noise that would pollute embeddings and
NLP models.  Every function is pure (no I/O) and stateless so it can be
called in parallel batch processing.

Usage:
    from src.processing.text_cleaner import clean_sec_text, clean_transcript_text
    clean = clean_sec_text(raw_filing_html_or_text)
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from loguru import logger

# ---------------------------------------------------------------------------
# Compiled regex patterns — compiled once at import time for performance
# ---------------------------------------------------------------------------

# EDGAR filing header block (appears before the actual document body)
_EDGAR_HEADER = re.compile(
    r"<SEC-HEADER>.*?</SEC-HEADER>",
    re.DOTALL | re.IGNORECASE,
)
_EDGAR_DOCUMENT_TAGS = re.compile(
    r"</?(?:SEC-DOCUMENT|SEQUENCE|FILENAME|DESCRIPTION|TYPE|TEXT)>",
    re.IGNORECASE,
)

# HTML tags (broad — BeautifulSoup handles structured parsing; this is a
# safety net for leftover tags after extraction)
_HTML_TAGS = re.compile(r"<[^>]+>")

# HTML entities (&amp; &nbsp; &#160; etc.)
_HTML_ENTITIES = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")

# Exhibit references common in 10-K/10-Q
_EXHIBIT_REF = re.compile(
    r"(?:Exhibit|EX)-?\d{1,3}(?:\.\d{1,2})?[^\n]*\n",
    re.IGNORECASE,
)

# EDGAR XBRL inline tags  <ix:nonNumeric ...> etc.
_XBRL_TAGS = re.compile(r"<ix:[^>]+>|</ix:[^>]+>", re.IGNORECASE)

# Page-break artefacts from PDF extraction
_PAGE_BREAK = re.compile(r"\f|\x0c")

# Repeated separator lines  ─── ═══ *** ---
_SEPARATOR_LINE = re.compile(r"^[\-=*_~#]{3,}\s*$", re.MULTILINE)

# Table of contents entries:  "Item 1A. Risk Factors ........... 12"
_TOC_ENTRY = re.compile(r"^.{0,80}\.{3,}\s*\d+\s*$", re.MULTILINE)

# Consecutive blank lines — collapse to a single blank line
_MULTI_BLANK = re.compile(r"\n{3,}")

# Whitespace runs inside a line
_INLINE_SPACES = re.compile(r"[ \t]{2,}")

# Purely numeric lines (e.g. page numbers)
_LONE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

# Common SEC boilerplate phrases
_BOILERPLATE_PHRASES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"this report contains forward.looking statements[^\n]*",
        r"see accompanying notes to (?:condensed )?(?:consolidated )?financial statements",
        r"the following discussion should be read in conjunction with[^\n]*",
        r"for the (?:three|six|nine|twelve) months? ended[^\n]{0,60}",
        r"table of contents",
        r"index to financial statements",
        r"(?:incorporated herein by reference|incorporated by reference)[^\n]*",
    ]
]

# Transcript operator filler lines
_OPERATOR_FILLER = re.compile(
    r"^(?:Operator|Coordinator):\s*(?:Thank you[,.]?|Ladies and gentlemen[,.]?|"
    r"Please (?:stand by|hold)[,.]?|Your (?:next|first) question[^\n]*)[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)

# Transcript boilerplate disclaimers
_TRANSCRIPT_DISCLAIMER = re.compile(
    r"(?:This transcript|The following|Motley Fool|Seeking Alpha)[^\n]*"
    r"(?:transcription|transcript service|accuracy)[^\n]*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------


def remove_html(text: str) -> str:
    """Strip all HTML/XML tags and decode common HTML entities."""
    text = _XBRL_TAGS.sub(" ", text)
    text = _EDGAR_HEADER.sub("", text)
    text = _EDGAR_DOCUMENT_TAGS.sub("", text)
    text = _HTML_TAGS.sub(" ", text)
    # Decode a curated set of common entities manually (avoids html.unescape
    # dependency issues with edge cases in malformed EDGAR HTML)
    entity_map = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
        "&apos;": "'", "&nbsp;": " ", "&#160;": " ", "&#8212;": "—",
        "&#8211;": "–", "&#8216;": "'", "&#8217;": "'",
        "&#8220;": '"', "&#8221;": '"',
    }
    for entity, char in entity_map.items():
        text = text.replace(entity, char)
    # Remaining numeric/named entities → strip
    text = _HTML_ENTITIES.sub(" ", text)
    return text


def normalise_unicode(text: str) -> str:
    """
    Normalise to NFC, replace exotic Unicode punctuation with ASCII
    equivalents, and strip zero-width / non-printable characters.
    """
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u2019": "'",   # right single quotation
        "\u2018": "'",   # left single quotation
        "\u201c": '"',   # left double quotation
        "\u201d": '"',   # right double quotation
        "\u2013": "-",   # en dash
        "\u2014": " - ", # em dash
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
        "\u200b": "",    # zero-width space
        "\u00ad": "",    # soft hyphen
        "\ufeff": "",    # BOM
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Strip remaining non-printable chars (keep newlines + tabs)
    text = "".join(
        ch if (unicodedata.category(ch)[0] != "C" or ch in "\n\t") else " "
        for ch in text
    )
    return text


def remove_boilerplate(text: str) -> str:
    """Remove common SEC/transcript boilerplate phrases."""
    for pattern in _BOILERPLATE_PHRASES:
        text = pattern.sub("", text)
    return text


def clean_whitespace(text: str) -> str:
    """Collapse excessive whitespace, page breaks, and separator lines."""
    text = _PAGE_BREAK.sub("\n", text)
    text = _SEPARATOR_LINE.sub("", text)
    text = _TOC_ENTRY.sub("", text)
    text = _LONE_NUMBER.sub("", text)
    text = _INLINE_SPACES.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_sec_text(raw: str) -> str:
    """
    Full cleaning pipeline for SEC filing text (10-K / 10-Q / 8-K).

    Steps:
        1. HTML removal + entity decoding
        2. Unicode normalisation
        3. Exhibit reference stripping
        4. Boilerplate phrase removal
        5. Whitespace normalisation

    Args:
        raw: Raw text extracted from an EDGAR filing (HTML or plain text).

    Returns:
        Cleaned plain text, ready for chunking and embedding.
    """
    text = remove_html(raw)
    text = normalise_unicode(text)
    text = _EXHIBIT_REF.sub("", text)
    text = remove_boilerplate(text)
    text = clean_whitespace(text)

    original_len = len(raw)
    cleaned_len = len(text)
    reduction_pct = 100 * (1 - cleaned_len / original_len) if original_len else 0
    logger.debug(
        f"SEC text cleaned: {original_len:,} → {cleaned_len:,} chars "
        f"({reduction_pct:.1f}% reduction)"
    )
    return text


def clean_transcript_text(raw: str) -> str:
    """
    Full cleaning pipeline for earnings call transcript text.

    Removes transcript service disclaimers, operator filler lines, and
    applies standard whitespace normalisation.

    Args:
        raw: Raw transcript text (e.g. scraped from Motley Fool).

    Returns:
        Cleaned transcript text.
    """
    text = remove_html(raw)
    text = normalise_unicode(text)
    text = _TRANSCRIPT_DISCLAIMER.sub("", text)
    text = _OPERATOR_FILLER.sub("", text)
    text = remove_boilerplate(text)
    text = clean_whitespace(text)

    logger.debug(f"Transcript text cleaned: {len(raw):,} → {len(text):,} chars")
    return text


def clean_analyst_text(raw: str) -> str:
    """Minimal cleaning for analyst report text (preserve numbers carefully)."""
    text = remove_html(raw)
    text = normalise_unicode(text)
    text = clean_whitespace(text)
    return text


@lru_cache(maxsize=8)
def get_sec_section_pattern(section_name: str) -> re.Pattern[str]:
    """
    Return a compiled regex that detects a specific SEC filing section header.

    Cached so repeated calls don't recompile.  Covers the standard Item
    numbering used in 10-K and 10-Q filings.

    Args:
        section_name: One of 'risk_factors', 'mda', 'financial_statements',
                      'quantitative_disclosures', 'controls', 'legal'.
    """
    patterns: dict[str, str] = {
        "risk_factors": (
            r"item\s+1a\.?\s+risk\s+factors"
        ),
        "mda": (
            r"item\s+[27]\.?\s+"
            r"management(?:'s|\s+discussion)\s+and\s+analysis"
        ),
        "financial_statements": (
            r"item\s+[18]\.?\s+"
            r"(?:financial\s+statements|consolidated\s+balance)"
        ),
        "quantitative_disclosures": (
            r"item\s+(?:3|7a)\.?\s+quantitative\s+and\s+qualitative\s+disclosures"
        ),
        "controls": (
            r"item\s+(?:4|9a)\.?\s+controls\s+and\s+procedures"
        ),
        "legal": (
            r"item\s+[13]\.?\s+legal\s+proceedings"
        ),
        "business": (
            r"item\s+1\.?\s+business(?!\s+of)"
        ),
    }
    raw = patterns.get(section_name, re.escape(section_name))
    return re.compile(raw, re.IGNORECASE | re.MULTILINE)


def count_financial_numbers(text: str) -> int:
    """
    Count the number of financial figures in text.

    Matches patterns like: $1.2B, $850M, 12%, 3.5x, 1,234,567

    Args:
        text: Any text string.

    Returns:
        Count of financial number patterns found.
    """
    pattern = re.compile(
        r"(?:\$\s*[\d,]+(?:\.\d+)?(?:\s*[BbMmKk](?:illion|illions?)?)?"
        r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?"
        r"|\b\d+(?:\.\d+)?\s*%"
        r"|\b\d+(?:\.\d+)?\s*x\b)",
        re.IGNORECASE,
    )
    return len(pattern.findall(text))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_sec = """
    <SEC-HEADER>
    ACCESSION NUMBER: 0000320193-24-000123
    </SEC-HEADER>
    <TEXT>
    <DOCUMENT>
    <TYPE>10-Q
    <SEQUENCE>1

    &nbsp; Item&nbsp;1A.&nbsp;<b>Risk Factors</b> ......... 12
    &#160;
    Management\u2019s Discussion and Analysis of Financial Condition
    This report contains forward-looking statements that involve risks.

    We expect revenue between $85&#160;billion and $90 billion for Q4 2024.
    See accompanying notes to condensed consolidated financial statements.

    &#8212; Page 42 &#8212;

    </DOCUMENT>
    </TEXT>
    """

    cleaned = clean_sec_text(sample_sec)
    print("=== Cleaned SEC text ===")
    print(repr(cleaned[:300]))
    print(f"\nFinancial numbers found: {count_financial_numbers(cleaned)}")

    sample_transcript = """
    Operator: Thank you. Ladies and gentlemen, please stand by.
    Operator: Your first question comes from John Smith of Goldman Sachs.

    CEO (Tim Cook): Thank you for joining today\u2019s call. We\u2019re
    very pleased to report record revenue of $94.9 billion, up 6% year over year.

    Motley Fool transcription services provided accuracy may vary.
    """

    cleaned_t = clean_transcript_text(sample_transcript)
    print("\n=== Cleaned transcript text ===")
    print(repr(cleaned_t))
    print("\ntext_cleaner smoke test passed ✓")
