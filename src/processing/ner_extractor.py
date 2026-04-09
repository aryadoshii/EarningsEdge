"""
Named-entity recognition and forward guidance extraction.

Implements a two-layer hybrid NER system:

    Layer 1 — spaCy en_core_web_trf:
        Standard entities: ORG, MONEY, PERCENT, DATE, CARDINAL

    Layer 2 — Rule-based regex matchers:
        EPS guidance     → "expect EPS of $X.XX to $Y.YY"
        Revenue guidance → "revenue between $XB and $YB"
        Growth rates     → "expect growth of X% to Y%"
        Capex            → "capital expenditure of approximately $X"
        Margin           → "operating margin of approximately X%"

    GuidanceEntity objects are emitted for each match with:
        metric_type, value_low, value_high, unit, fiscal_period,
        raw_text, confidence_score, is_explicit

spaCy model is loaded lazily to avoid the cold-start penalty when this
module is imported but NER is not immediately needed.

Usage:
    extractor = NERExtractor()
    entities = extractor.extract(chunk_text)
    guidance  = extractor.extract_guidance(text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import GuidanceEntity, GuidanceMetric

# ---------------------------------------------------------------------------
# Lazy spaCy load
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp() -> Any:  # returns spacy.Language
    """Lazy-load the spaCy transformer pipeline."""
    global _nlp
    if _nlp is None:
        try:
            import spacy  # type: ignore
            logger.info(f"Loading spaCy model: {settings.SPACY_MODEL}")
            _nlp = spacy.load(settings.SPACY_MODEL)
        except OSError:
            logger.warning(
                f"spaCy model '{settings.SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {settings.SPACY_MODEL}"
            )
            try:
                import spacy
                logger.info("Falling back to en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("No spaCy model available — NER disabled")
        except ImportError:
            logger.error("spaCy not installed — NER disabled")
    return _nlp


# ---------------------------------------------------------------------------
# Regex patterns for financial guidance extraction
# ---------------------------------------------------------------------------

# Multiplier suffixes → numeric factor
_MULTIPLIERS: dict[str, float] = {
    "b": 1e9, "bn": 1e9, "bil": 1e9, "billion": 1e9,
    "m": 1e6, "mm": 1e6, "mil": 1e6, "million": 1e6,
    "k": 1e3, "thousand": 1e3,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
}

# Common fiscal period patterns  "Q2 2024", "fiscal 2025", "FY2024"
_FISCAL_PERIOD_PAT = re.compile(
    r"(?:"
    r"(?:Q[1-4]\s+\d{4})"
    r"|(?:fiscal\s+(?:year\s+)?\d{4})"
    r"|(?:FY\s*\d{2,4})"
    r"|(?:(?:first|second|third|fourth)\s+quarter(?:\s+of\s+\d{4})?)"
    r"|(?:full.year\s+\d{4})"
    r"|(?:next\s+(?:quarter|year|fiscal\s+year))"
    r")",
    re.IGNORECASE,
)

# Numeric value with optional $ prefix and multiplier suffix
_VALUE_PAT = (
    r"\$?\s*(\d+(?:\.\d+)?)\s*"         # number
    r"(billion|million|trillion|thousand|bn?|mm?|t|k)?"  # multiplier
)

# Range connector words
_RANGE_CONN = r"\s*(?:to|-|and|–|—)\s*"


def _parse_value(num_str: str, mult_str: str | None) -> float | None:
    """
    Convert a numeric string + multiplier suffix to a float in USD/base units.

    Args:
        num_str:  The numeric portion, e.g. "94.9"
        mult_str: Optional multiplier suffix, e.g. "billion"

    Returns:
        Float value in base units, or None if parsing fails.
    """
    try:
        value = float(num_str.replace(",", ""))
        if mult_str:
            factor = _MULTIPLIERS.get(mult_str.lower(), 1.0)
            value *= factor
        return value
    except ValueError:
        return None


@dataclass
class _RawGuidanceMatch:
    """Intermediate container before constructing a GuidanceEntity."""
    metric: GuidanceMetric
    value_low: float | None
    value_high: float | None
    unit: str
    fiscal_period: str
    raw_text: str
    confidence: float
    is_explicit: bool


# ---------------------------------------------------------------------------
# Guidance patterns
# ---------------------------------------------------------------------------

# Each pattern is (GuidanceMetric, unit, compiled_regex, confidence)
# The regex MUST have named groups: value1, mult1 (and optionally value2, mult2)
# for range patterns, or value1/mult1 for single-value patterns.
_GUIDANCE_PATTERNS: list[tuple[GuidanceMetric, str, re.Pattern[str], float]] = []


def _add_pattern(
    metric: GuidanceMetric,
    unit: str,
    pattern: str,
    confidence: float = 0.85,
) -> None:
    _GUIDANCE_PATTERNS.append((
        metric, unit,
        re.compile(pattern, re.IGNORECASE | re.DOTALL),
        confidence,
    ))


# EPS guidance patterns
_add_pattern(GuidanceMetric.EPS, "USD/share",
    r"(?:expect|guide|guidance|project|anticipate)\w*\s+"
    r"(?:earnings\s+per\s+share|EPS|diluted\s+EPS)\s+"
    r"(?:of\s+|to\s+be\s+|between\s+|in\s+(?:the\s+)?range\s+of\s+)?"
    r"\$?(?P<value1>\d+\.\d{1,4})"
    r"(?:" + _RANGE_CONN + r"\$?(?P<value2>\d+\.\d{1,4}))?",
)
_add_pattern(GuidanceMetric.EPS, "USD/share",
    r"EPS\s+guidance\s+of\s+\$?(?P<value1>\d+\.\d{1,4})"
    r"(?:" + _RANGE_CONN + r"\$?(?P<value2>\d+\.\d{1,4}))?",
)

# Revenue guidance patterns
_add_pattern(GuidanceMetric.REVENUE, "USD",
    r"(?:expect|guide|project|anticipate|forecast)\w*\s+"
    r"(?:(?:total\s+)?(?:net\s+)?revenue[s]?|(?:net\s+)?sales)\s+"
    r"(?:of\s+|to\s+(?:be\s+)?|between\s+|in\s+(?:the\s+)?range\s+of\s+)?"
    r"\$?(?P<value1>\d+(?:\.\d+)?)\s*(?P<mult1>billion|million|B|M|bn|mm)?"
    r"(?:" + _RANGE_CONN +
    r"\$?(?P<value2>\d+(?:\.\d+)?)\s*(?P<mult2>billion|million|B|M|bn|mm))?",
)
_add_pattern(GuidanceMetric.REVENUE, "USD",
    r"revenue\s+(?:guidance|outlook)\s+of\s+"
    r"\$?(?P<value1>\d+(?:\.\d+)?)\s*(?P<mult1>billion|million|B|M|bn|mm)?"
    r"(?:" + _RANGE_CONN +
    r"\$?(?P<value2>\d+(?:\.\d+)?)\s*(?P<mult2>billion|million|B|M|bn|mm))?",
)

# Capex guidance
_add_pattern(GuidanceMetric.CAPEX, "USD",
    r"(?:capital\s+expenditure[s]?|capex)\s+"
    r"(?:of\s+|approximately\s+|between\s+)?"
    r"\$?(?P<value1>\d+(?:\.\d+)?)\s*(?P<mult1>billion|million|B|M|bn|mm)?"
    r"(?:" + _RANGE_CONN +
    r"\$?(?P<value2>\d+(?:\.\d+)?)\s*(?P<mult2>billion|million|B|M|bn|mm))?",
    confidence=0.80,
)

# Margin guidance
_add_pattern(GuidanceMetric.MARGIN, "%",
    r"(?:gross|operating|net|EBITDA)\s+margin\s+"
    r"(?:of\s+|approximately\s+|between\s+|expected\s+(?:at|to\s+be)\s+)?"
    r"(?P<value1>\d+(?:\.\d+)?)\s*%?"
    r"(?:" + _RANGE_CONN + r"(?P<value2>\d+(?:\.\d+)?)\s*%)?",
    confidence=0.80,
)

# Growth rate guidance
_add_pattern(GuidanceMetric.GROWTH, "%",
    r"(?:expect|anticipate|project)\w*\s+(?:revenue\s+)?growth\s+"
    r"(?:of\s+|approximately\s+|between\s+)?"
    r"(?P<value1>\d+(?:\.\d+)?)\s*%?"
    r"(?:" + _RANGE_CONN + r"(?P<value2>\d+(?:\.\d+)?)\s*%)?",
    confidence=0.75,
)


def _extract_fiscal_period(text: str, match_start: int, window: int = 100) -> str:
    """
    Look for a fiscal period reference near a guidance match.

    Searches in a window of characters before and after the match start.

    Args:
        text:        Full text being searched.
        match_start: Character offset of the guidance match.
        window:      Characters to search before/after match.

    Returns:
        Fiscal period string (e.g. "Q4 2024"), or "" if none found.
    """
    start = max(0, match_start - window)
    end = min(len(text), match_start + window)
    context = text[start:end]
    m = _FISCAL_PERIOD_PAT.search(context)
    return m.group(0).strip() if m else ""


# ---------------------------------------------------------------------------
# NERExtractor
# ---------------------------------------------------------------------------

class NERExtractor:
    """
    Hybrid NER extractor for SEC filings and earnings transcripts.

    Combines spaCy en_core_web_trf for standard entities with rule-based
    regex patterns for financial guidance extraction.
    """

    # ------------------------------------------------------------------
    # Standard NER (spaCy)
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        """
        Run spaCy NER on text and return standard financial entities.

        Args:
            text: Input text (one chunk or full section).

        Returns:
            List of dicts: [{text, label, start, end}]
            Filtered to: ORG, MONEY, PERCENT, DATE, CARDINAL, GPE
        """
        nlp = _get_nlp()
        if nlp is None:
            return []

        keep_labels = {"ORG", "MONEY", "PERCENT", "DATE", "CARDINAL", "GPE"}
        try:
            doc = nlp(text[:100_000])  # cap to avoid OOM on very long texts
            return [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
                if ent.label_ in keep_labels
            ]
        except Exception as exc:
            logger.error(f"spaCy NER failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Guidance extraction (regex)
    # ------------------------------------------------------------------

    def extract_guidance(self, text: str) -> list[GuidanceEntity]:
        """
        Extract forward-looking numerical guidance from text using regex.

        Applies all guidance patterns and deduplicates overlapping matches.

        Args:
            text: Text of one chunk or a full filing section.

        Returns:
            List of GuidanceEntity objects (may be empty).
        """
        raw_matches: list[_RawGuidanceMatch] = []

        for metric, unit, pattern, confidence in _GUIDANCE_PATTERNS:
            for m in pattern.finditer(text):
                try:
                    gd = self._parse_guidance_match(m, metric, unit, confidence, text)
                    if gd:
                        raw_matches.append(gd)
                except Exception as exc:
                    logger.debug(f"Guidance parse error: {exc}")

        # Deduplicate by (metric, value_low, value_high) — keep highest confidence
        seen: dict[tuple[str, float | None, float | None], _RawGuidanceMatch] = {}
        for match in raw_matches:
            key = (match.metric.value, _round_val(match.value_low), _round_val(match.value_high))
            if key not in seen or match.confidence > seen[key].confidence:
                seen[key] = match

        entities = [
            GuidanceEntity(
                metric_type=rm.metric,
                value_low=rm.value_low,
                value_high=rm.value_high,
                unit=rm.unit,
                fiscal_period=rm.fiscal_period,
                raw_text=rm.raw_text,
                confidence_score=rm.confidence,
                is_explicit=rm.is_explicit,
            )
            for rm in seen.values()
        ]

        if entities:
            logger.debug(f"Extracted {len(entities)} guidance entities")
        return entities

    def _parse_guidance_match(
        self,
        m: re.Match[str],
        metric: GuidanceMetric,
        unit: str,
        confidence: float,
        full_text: str,
    ) -> _RawGuidanceMatch | None:
        """
        Convert a regex match object into a _RawGuidanceMatch.

        Handles both single-value and range patterns.

        Args:
            m:         The re.Match object.
            metric:    Guidance metric type.
            unit:      Unit of measure.
            confidence: Base confidence score.
            full_text: Original text (for fiscal period context extraction).

        Returns:
            _RawGuidanceMatch or None if the match can't be parsed.
        """
        gd = m.groupdict()

        v1_str = gd.get("value1")
        v2_str = gd.get("value2")
        m1_str = gd.get("mult1")
        m2_str = gd.get("mult2")

        if not v1_str:
            return None

        value1 = _parse_value(v1_str, m1_str)
        value2 = _parse_value(v2_str, m2_str or m1_str) if v2_str else None

        if value1 is None:
            return None

        # Sanity check: reject implausibly small or large values per metric
        if not _sanity_check(metric, value1):
            return None

        fiscal_period = _extract_fiscal_period(full_text, m.start())
        raw_text = full_text[max(0, m.start() - 20): m.end() + 20].strip()

        return _RawGuidanceMatch(
            metric=metric,
            value_low=min(value1, value2) if value2 else value1,
            value_high=max(value1, value2) if value2 else None,
            unit=unit,
            fiscal_period=fiscal_period,
            raw_text=raw_text,
            confidence=confidence,
            is_explicit=(value2 is not None),
        )

    # ------------------------------------------------------------------
    # Combined extraction
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict[str, Any]:
        """
        Run both spaCy NER and guidance extraction on text.

        Args:
            text: Input text chunk or section.

        Returns:
            Dict with keys 'entities' (spaCy) and 'guidance' (regex).
        """
        return {
            "entities": self.extract_entities(text),
            "guidance": self.extract_guidance(text),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_val(v: float | None) -> float | None:
    return round(v, 4) if v is not None else None


# Sanity bounds per metric — reject clearly wrong values
_SANITY_BOUNDS: dict[GuidanceMetric, tuple[float, float]] = {
    GuidanceMetric.EPS: (0.01, 1000.0),              # cents to thousands USD
    GuidanceMetric.REVENUE: (1e4, 5e12),              # $10K to $5T
    GuidanceMetric.CAPEX: (1e4, 1e12),                # $10K to $1T
    GuidanceMetric.MARGIN: (0.01, 100.0),             # 0.01% to 100%
    GuidanceMetric.GROWTH: (-100.0, 10_000.0),        # -100% to 10000%
    GuidanceMetric.OTHER: (0.0, float("inf")),
}


def _sanity_check(metric: GuidanceMetric, value: float) -> bool:
    lo, hi = _SANITY_BOUNDS.get(metric, (0.0, float("inf")))
    return lo <= value <= hi


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

ner_extractor = NERExtractor()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = NERExtractor()

    test_texts = [
        # EPS guidance
        "For Q4 2024, we expect diluted EPS of $1.55 to $1.65.",
        # Revenue guidance range
        "We anticipate total net revenues between $89 billion and $93 billion for the fourth quarter.",
        # Revenue single value
        "Full-year 2024 revenue guidance of $381 billion.",
        # Capex
        "Capital expenditures of approximately $11 billion for fiscal 2024.",
        # Margin
        "We expect gross margin of 45.5% to 46.5% for Q4.",
        # Growth
        "We anticipate revenue growth of 8% to 10% for the next fiscal year.",
        # No guidance (should return empty)
        "The company was founded in 1976 in Cupertino, California.",
    ]

    print("=== Guidance Extraction Tests ===\n")
    for text in test_texts:
        results = extractor.extract_guidance(text)
        print(f"Text: {text[:70]}...")
        if results:
            for g in results:
                print(
                    f"  [{g.metric_type.value:8s}] "
                    f"low={g.value_low:>15,.2f}  "
                    f"high={str(g.value_high or 'N/A'):>15}  "
                    f"unit={g.unit}  "
                    f"period='{g.fiscal_period}'  "
                    f"conf={g.confidence_score:.2f}"
                )
        else:
            print("  (no guidance found)")
        print()

    print("ner_extractor smoke test passed ✓")
