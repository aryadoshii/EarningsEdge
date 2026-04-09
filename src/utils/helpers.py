"""
Shared utility functions used across EarningsEdge modules.

Contains stateless pure helpers for: text normalisation, date manipulation,
financial number parsing, quarter arithmetic, and misc formatting.

Nothing in this file should import from other src/ modules (prevents circular
imports).  All functions are independently unit-testable.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Regex to detect financial figures: $3.2B, 394.3 million, -12.5%, etc.
_FINANCIAL_NUMBER_RE = re.compile(
    r"""
    (?:                              # Optional currency symbol
        [\$£€¥]
    )?
    [-−]?                            # Optional negative sign
    \d{1,3}                          # Leading digits
    (?:[,\.\d]*\d)?                  # Thousands separators / decimals
    (?:\s*(?:billion|million|trillion|thousand|B|M|T|K))?  # Magnitude suffix
    (?:\s*%)?                        # Optional percentage
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Common SEC boilerplate phrases to strip from filings
_BOILERPLATE_PHRASES = [
    r"this\s+(?:annual|quarterly)\s+report\s+(?:on\s+form\s+)?10-[kq]",
    r"forward[\s-]looking\s+statements?",
    r"safe\s+harbor\s+statement",
    r"cautionary\s+note\s+regarding",
    r"item\s+\d+[a-z]?\s*\.",
    r"table\s+of\s+contents",
    r"signature[s]?\s+pursuant\s+to\s+the\s+requirements",
]
_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PHRASES), re.IGNORECASE | re.DOTALL
)


def contains_financial_numbers(text: str) -> bool:
    """
    Return True if the text contains at least one financial figure.

    Detects: dollar amounts, percentages, magnitudes (B/M/T), negative values.

    Args:
        text: Plain text string to search.

    Returns:
        True if any financial number pattern is found.
    """
    return bool(_FINANCIAL_NUMBER_RE.search(text))


def count_financial_numbers(text: str) -> int:
    """
    Count the number of financial figures in the text.

    Args:
        text: Plain text string.

    Returns:
        Integer count of financial number matches.
    """
    return len(_FINANCIAL_NUMBER_RE.findall(text))


def normalise_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into single spaces.

    Also strips leading/trailing whitespace and removes null bytes.

    Args:
        text: Raw string (may contain tabs, newlines, carriage returns).

    Returns:
        Cleaned string with normalised whitespace.
    """
    text = text.replace("\x00", "").replace("\r\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_words: int = 150) -> str:
    """
    Truncate text to at most `max_words` words, appending "…" if cut.

    Args:
        text:      Input text.
        max_words: Maximum number of words to keep.

    Returns:
        Truncated string (with ellipsis if truncated).
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def clean_sec_text(text: str) -> str:
    """
    Remove common SEC filing boilerplate from extracted text.

    Args:
        text: Raw extracted text from an SEC filing.

    Returns:
        Text with boilerplate phrases collapsed to single spaces.
    """
    text = _BOILERPLATE_RE.sub(" ", text)
    return normalise_whitespace(text)


# ---------------------------------------------------------------------------
# Financial number parsing
# ---------------------------------------------------------------------------

_MAGNITUDE_MAP: dict[str, float] = {
    "trillion": 1e12,
    "t": 1e12,
    "billion": 1e9,
    "b": 1e9,
    "million": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}

_NUMBER_RE = re.compile(
    r"([-−]?)\$?\s*(\d[\d,\.]*)\s*(trillion|billion|million|thousand|[tTbBmMkK])?",
    re.IGNORECASE,
)


def parse_financial_value(text: str) -> float | None:
    """
    Extract and parse the first financial number from a text string.

    Handles: "$3.2 billion", "394,300 million", "−$1.2B", "12.5%"

    Args:
        text: Text containing a financial number.

    Returns:
        Parsed float value (in base units, e.g. dollars not billions),
        or None if no parseable number found.
    """
    text = text.strip().replace("−", "-").replace("–", "-")
    m = _NUMBER_RE.search(text)
    if not m:
        return None

    sign_str, num_str, suffix = m.group(1), m.group(2), m.group(3) or ""
    try:
        value = float(num_str.replace(",", ""))
    except ValueError:
        return None

    magnitude = _MAGNITUDE_MAP.get(suffix.lower(), 1.0)
    value *= magnitude
    if sign_str in ("-", "−"):
        value = -value

    return value


def normalise_to_millions(value: float, unit: str) -> float:
    """
    Convert a value with a stated unit to millions.

    Args:
        value: Numeric value.
        unit:  Unit string, e.g. "billion", "million", "thousand", "USD".

    Returns:
        Value expressed in millions.
    """
    multiplier_to_base = _MAGNITUDE_MAP.get(unit.lower().strip(), 1.0)
    return value * multiplier_to_base / 1e6


# ---------------------------------------------------------------------------
# Quarter / date helpers
# ---------------------------------------------------------------------------

def quarter_to_period_string(quarter: str, year: int) -> str:
    """
    Format quarter and year into a period string.

    Args:
        quarter: "Q1", "Q2", "Q3", or "Q4".
        year:    Four-digit year.

    Returns:
        String like "Q1 2024".
    """
    return f"{quarter} {year}"


def period_to_date(quarter: str, year: int) -> date:
    """
    Convert a fiscal quarter + year to an approximate period-end date.

    Assumes calendar-quarter fiscal years (Q1 = March 31, etc.)

    Args:
        quarter: "Q1", "Q2", "Q3", or "Q4".
        year:    Four-digit year.

    Returns:
        End date of the fiscal quarter.
    """
    quarter_end_months = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}
    quarter_end_days = {"Q1": 31, "Q2": 30, "Q3": 30, "Q4": 31}
    month = quarter_end_months.get(quarter.upper(), 12)
    day = quarter_end_days.get(quarter.upper(), 31)
    return date(year, month, day)


def previous_quarter(quarter: str, year: int) -> tuple[str, int]:
    """
    Return the quarter immediately before the given one.

    Args:
        quarter: "Q1"–"Q4".
        year:    Four-digit year.

    Returns:
        Tuple of (previous_quarter_str, previous_year).
    """
    q_num = int(quarter[1])
    if q_num == 1:
        return "Q4", year - 1
    return f"Q{q_num - 1}", year


def next_quarter(quarter: str, year: int) -> tuple[str, int]:
    """
    Return the quarter immediately after the given one.

    Args:
        quarter: "Q1"–"Q4".
        year:    Four-digit year.

    Returns:
        Tuple of (next_quarter_str, next_year).
    """
    q_num = int(quarter[1])
    if q_num == 4:
        return "Q1", year + 1
    return f"Q{q_num + 1}", year


def quarters_between(
    start_quarter: str,
    start_year: int,
    end_quarter: str,
    end_year: int,
) -> list[tuple[str, int]]:
    """
    Generate all quarters from start to end (inclusive).

    Args:
        start_quarter: "Q1"–"Q4".
        start_year:    Four-digit year.
        end_quarter:   "Q1"–"Q4".
        end_year:      Four-digit year.

    Returns:
        Ordered list of (quarter_str, year) tuples.
    """
    result: list[tuple[str, int]] = []
    q, y = start_quarter, start_year

    # Safety cap to prevent infinite loops
    max_iterations = 40

    for _ in range(max_iterations):
        result.append((q, y))
        if q == end_quarter and y == end_year:
            break
        q, y = next_quarter(q, y)

    return result


# ---------------------------------------------------------------------------
# Misc formatting
# ---------------------------------------------------------------------------

def format_large_number(value: float, decimals: int = 1) -> str:
    """
    Format a large number with magnitude suffix for display.

    Args:
        value:    Numeric value (in base units, e.g. dollars).
        decimals: Decimal places to show.

    Returns:
        Human-readable string, e.g. "$394.3B" or "$1.2M".
    """
    if abs(value) >= 1e12:
        return f"${value / 1e12:.{decimals}f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.{decimals}f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.{decimals}f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.{decimals}f}K"
    return f"${value:.{decimals}f}"


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Clamp value to [low, high] range."""
    return max(low, min(high, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide numerator by denominator, returning default if denominator is zero.

    Args:
        numerator:   Dividend.
        denominator: Divisor.
        default:     Value to return when denominator is zero.

    Returns:
        Division result, or default on ZeroDivisionError.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Recursively flatten a nested dict into a single-level dict.

    Args:
        d:          Input dict (may be nested).
        parent_key: Prefix key for recursive calls.
        sep:        Separator between nested keys.

    Returns:
        Flat dict with dotted keys.

    Example:
        {"a": {"b": 1, "c": 2}} → {"a.b": 1, "a.c": 2}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Text utilities
    sample = "Revenue grew to $394.3 billion, up 12.5% year-over-year."
    print(f"Has numbers  : {contains_financial_numbers(sample)}")
    print(f"Number count : {count_financial_numbers(sample)}")

    # Number parsing
    for s in ["$3.2 billion", "−$1.5B", "394,300 million", "12.5%", "no number here"]:
        parsed = parse_financial_value(s)
        print(f"  '{s}' → {parsed}")

    # Quarter helpers
    print(f"Q1 2024 prev : {previous_quarter('Q1', 2024)}")
    print(f"Q4 2023 next : {next_quarter('Q4', 2023)}")
    print(f"Q2-Q1 range  : {quarters_between('Q2', 2023, 'Q1', 2024)}")

    # Formatting
    for v in [394_300_000_000, 1_200_000_000, 850_000_000, 12_500]:
        print(f"  {v:,} → {format_large_number(v)}")

    print("Helpers smoke test passed ✓")
