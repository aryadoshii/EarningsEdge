"""
Metadata tagger for document chunks.

Takes raw DocumentChunk objects (which may have incomplete or inferred
metadata) and enriches them with:

    - Confirmed quarter / year derived from filing date or explicit text cues
    - Section type reclassification based on content signals
    - Speaker role normalisation for transcript chunks
    - Guidance flag — marks chunks containing forward-looking numbers
    - Sentiment pre-label — quick lexicon-based polarity tag (positive /
      negative / neutral) as a cheap first-pass filter before FinBERT

No model inference is performed here — this is purely rule-based for speed.
FinBERT sentiment scores are computed separately in analysis/sentiment_analyzer.py.

Usage:
    tagger = MetadataTagger()
    enriched_chunks = tagger.tag_batch(chunks)
    enriched_chunk  = tagger.tag(chunk)
"""

from __future__ import annotations

import re
from datetime import date

from loguru import logger

from src.ingestion.data_validator import (
    DocumentChunk,
    FilingType,
    Quarter,
    SectionType,
)

# ---------------------------------------------------------------------------
# Content-based reclassification patterns
# ---------------------------------------------------------------------------

# These patterns override the section_type when the chunk text clearly
# signals a different section than the structural parser detected.

_GUIDANCE_SIGNALS = re.compile(
    r"\b(?:expect(?:ation)?s?|guidance|outlook|forecast|project(?:ion)?|"
    r"anticipate|target|full.year|fiscal \d{4}|next quarter|going forward|"
    r"we believe\s+(?:revenue|earnings|growth)|will be approximately)\b",
    re.IGNORECASE,
)

_RISK_SIGNALS = re.compile(
    r"\b(?:risk\s+factor|uncertain(?:ty)?|material(?:ly)?|adverse(?:ly)?|"
    r"could\s+(?:harm|impair|negatively)|subject\s+to|no\s+assurance|"
    r"litigation|regulatory\s+(?:risk|change)|competition\s+may)\b",
    re.IGNORECASE,
)

_FINANCIAL_HIGHLIGHT_SIGNALS = re.compile(
    r"\b(?:revenue\s+(?:of|increased|decreased|grew)|"
    r"net\s+income|earnings\s+per\s+share|gross\s+margin|"
    r"operating\s+(?:income|cash\s+flow)|EBITDA|EPS\s+(?:of|was|were))\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Quarter inference from text
# ---------------------------------------------------------------------------

_QUARTER_TEXT_MAP: list[tuple[re.Pattern[str], Quarter]] = [
    (re.compile(r"\b(?:Q1|first\s+quarter|three\s+months\s+ended\s+(?:march|jun))\b", re.I), Quarter.Q1),
    (re.compile(r"\b(?:Q2|second\s+quarter|six\s+months\s+ended\s+(?:jun))\b", re.I), Quarter.Q2),
    (re.compile(r"\b(?:Q3|third\s+quarter|nine\s+months\s+ended\s+(?:sep))\b", re.I), Quarter.Q3),
    (re.compile(r"\b(?:Q4|fourth\s+quarter|twelve\s+months\s+ended|full.year|annual)\b", re.I), Quarter.Q4),
]


def _infer_quarter_from_text(text: str) -> Quarter | None:
    """
    Attempt to infer the reporting quarter from chunk text content.

    Args:
        text: Chunk text.

    Returns:
        Quarter if a strong signal is found, else None.
    """
    for pattern, quarter in _QUARTER_TEXT_MAP:
        if pattern.search(text):
            return quarter
    return None


def _infer_quarter_from_date(dt: date) -> Quarter:
    """
    Map a period-end date to a fiscal quarter.

    Uses standard calendar quarter mapping (Q1=Jan-Mar, etc.).

    Args:
        dt: Period-end date.

    Returns:
        Quarter enum value.
    """
    month = dt.month
    if month <= 3:
        return Quarter.Q1
    if month <= 6:
        return Quarter.Q2
    if month <= 9:
        return Quarter.Q3
    return Quarter.Q4


# ---------------------------------------------------------------------------
# Simple lexicon-based sentiment pre-labeller
# ---------------------------------------------------------------------------

_POSITIVE_LEXICON = re.compile(
    r"\b(?:record|strong|growth|beat|exceeded|outperform|robust|solid|"
    r"momentum|accelerat|expand|gain|improvement|confident|pleased|"
    r"increased|grew|higher|above|ahead|positive|success|win|delight|"
    r"launch(?:ed)?|innovat|compell)\b",
    re.IGNORECASE,
)

_NEGATIVE_LEXICON = re.compile(
    r"\b(?:decline|decreased|fell|miss(?:ed)?|below|lower|challeng|"
    r"difficult|headwind|pressure|weak(?:ness)?|concern|risk|uncer"
    r"tain|impact(?:ed)?|disappoint|slowdown|contraction|loss|"
    r"impairment|restructur|layoff|reduction)\b",
    re.IGNORECASE,
)


def _quick_sentiment(text: str) -> str:
    """
    Fast lexicon-based sentiment pre-label (not FinBERT — just a quick filter).

    Returns 'positive', 'negative', or 'neutral'.
    """
    pos_count = len(_POSITIVE_LEXICON.findall(text))
    neg_count = len(_NEGATIVE_LEXICON.findall(text))
    if pos_count == 0 and neg_count == 0:
        return "neutral"
    if pos_count > neg_count * 1.5:
        return "positive"
    if neg_count > pos_count * 1.5:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# MetadataTagger
# ---------------------------------------------------------------------------

class MetadataTagger:
    """
    Enriches DocumentChunk metadata using content-based rules.

    All operations are pure transformations — no I/O, no model inference.
    The original chunk is not mutated; a new DocumentChunk is returned.
    """

    def tag(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Apply all metadata enrichment rules to a single chunk.

        Enrichment steps:
            1. Section type reclassification via content signals
            2. Quarter confirmation / inference from text
            3. Guidance flag (contains_numbers + guidance signal)
            4. Speaker normalisation

        Args:
            chunk: Input DocumentChunk (will not be mutated).

        Returns:
            New DocumentChunk with enriched metadata fields.
        """
        # Work on a model copy so the original is unchanged
        data = chunk.model_dump()

        # --- Step 1: Section reclassification
        text = chunk.text
        current_section = chunk.section_type

        if current_section not in (SectionType.QA, SectionType.PREPARED_REMARKS):
            if _GUIDANCE_SIGNALS.search(text) and chunk.contains_numbers:
                data["section_type"] = SectionType.GUIDANCE
            elif (
                current_section == SectionType.UNKNOWN
                and _RISK_SIGNALS.search(text)
            ):
                data["section_type"] = SectionType.RISK_FACTORS
            elif (
                current_section == SectionType.UNKNOWN
                and _FINANCIAL_HIGHLIGHT_SIGNALS.search(text)
            ):
                data["section_type"] = SectionType.FINANCIAL_STATEMENTS

        # --- Step 2: Quarter confirmation
        if chunk.quarter == Quarter.Q4 and chunk.filing_type in (
            FilingType.FORM_10K, FilingType.FORM_10K_A
        ):
            pass  # 10-K is always Q4/annual — no change needed
        else:
            inferred = _infer_quarter_from_text(text)
            if inferred and inferred != chunk.quarter:
                logger.debug(
                    f"[{chunk.ticker}] Quarter override: "
                    f"{chunk.quarter.value} → {inferred.value} "
                    f"(chunk {chunk.chunk_index})"
                )
                data["quarter"] = inferred

        # --- Step 3: Guidance flag — ensure contains_numbers is accurate
        has_numbers = chunk.contains_numbers
        if not has_numbers:
            from src.processing.text_cleaner import count_financial_numbers
            has_numbers = count_financial_numbers(text) > 0
            data["contains_numbers"] = has_numbers

        # --- Step 4: Speaker normalisation (transcript chunks)
        speaker = chunk.speaker
        if speaker and speaker not in (
            "CEO", "CFO", "COO", "IR", "Analyst", "Operator", "QA_PAIR", ""
        ):
            # Normalise free-form roles
            if re.search(r"\bCEO\b|\bchief executive\b", speaker, re.I):
                data["speaker"] = "CEO"
            elif re.search(r"\bCFO\b|\bchief financial\b", speaker, re.I):
                data["speaker"] = "CFO"
            elif re.search(r"\banalyst\b|\bresearch\b", speaker, re.I):
                data["speaker"] = "Analyst"

        return DocumentChunk(**data)

    def tag_batch(
        self,
        chunks: list[DocumentChunk],
        log_interval: int = 100,
    ) -> list[DocumentChunk]:
        """
        Apply tag() to a list of chunks, logging progress.

        Args:
            chunks:       List of DocumentChunk objects.
            log_interval: Log a progress message every N chunks.

        Returns:
            List of enriched DocumentChunk objects (same length).
        """
        enriched: list[DocumentChunk] = []
        for i, chunk in enumerate(chunks):
            enriched.append(self.tag(chunk))
            if (i + 1) % log_interval == 0:
                logger.debug(f"Tagged {i + 1}/{len(chunks)} chunks")

        # Compute section distribution for logging
        section_counts: dict[str, int] = {}
        for c in enriched:
            key = c.section_type.value
            section_counts[key] = section_counts.get(key, 0) + 1

        logger.info(
            f"Batch tagging complete: {len(enriched)} chunks | "
            f"sections: {section_counts}"
        )
        return enriched

    def compute_quick_sentiment_labels(
        self,
        chunks: list[DocumentChunk],
    ) -> dict[str, str]:
        """
        Compute lexicon-based sentiment pre-labels for all chunks.

        Returns a mapping {chunk_id: 'positive'|'negative'|'neutral'}.
        This is used as a cheap first-pass filter before batching chunks
        to FinBERT — high-signal chunks can be prioritised.

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            Dict mapping chunk_id to sentiment label.
        """
        return {c.chunk_id: _quick_sentiment(c.text) for c in chunks}

    def get_guidance_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """
        Filter chunks that are likely to contain forward guidance.

        Args:
            chunks: Any list of DocumentChunk objects.

        Returns:
            Subset where section_type is GUIDANCE or text contains
            strong guidance signals.
        """
        return [
            c for c in chunks
            if c.section_type == SectionType.GUIDANCE
            or (c.contains_numbers and bool(_GUIDANCE_SIGNALS.search(c.text)))
        ]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

metadata_tagger = MetadataTagger()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date as d
    from src.ingestion.data_validator import ChunkingStrategy

    tagger = MetadataTagger()

    # Test 1: Guidance reclassification
    chunk_guidance = DocumentChunk(
        chunk_id="test-001",
        text=(
            "For Q4 2024, we expect diluted EPS of $1.55 to $1.65 and "
            "total revenue between $89 billion and $93 billion."
        ),
        ticker="AAPL",
        filing_type=FilingType.FORM_10Q,
        section_type=SectionType.MDA,    # should be reclassified
        quarter=Quarter.Q3,
        year=2024,
        filed_date="2024-08-02",
        chunk_index=5,
        total_chunks=120,
        word_count=30,
        contains_numbers=True,
        chunking_strategy=ChunkingStrategy.SECTION_AWARE,
    )

    tagged = tagger.tag(chunk_guidance)
    assert tagged.section_type == SectionType.GUIDANCE, (
        f"Expected GUIDANCE, got {tagged.section_type}"
    )
    print(f"✓ Guidance reclassification: MDA → {tagged.section_type.value}")

    # Test 2: Quarter inference from text
    chunk_q1 = DocumentChunk(
        chunk_id="test-002",
        text="During the first quarter ended March 31, 2024, revenue grew 8%.",
        ticker="MSFT",
        filing_type=FilingType.FORM_10Q,
        section_type=SectionType.MDA,
        quarter=Quarter.Q4,        # wrong — should be inferred as Q1
        year=2024,
        filed_date="2024-04-25",
        chunk_index=0,
        total_chunks=50,
        word_count=15,
        contains_numbers=True,
        chunking_strategy=ChunkingStrategy.SECTION_AWARE,
    )
    tagged_q1 = tagger.tag(chunk_q1)
    assert tagged_q1.quarter == Quarter.Q1, f"Expected Q1, got {tagged_q1.quarter}"
    print(f"✓ Quarter inference: Q4 → {tagged_q1.quarter.value}")

    # Test 3: Risk reclassification
    chunk_risk = DocumentChunk(
        chunk_id="test-003",
        text=(
            "We face regulatory risk in our international markets. "
            "There is no assurance that our products will maintain market share. "
            "Competition may adversely affect revenue and margins."
        ),
        ticker="GOOG",
        filing_type=FilingType.FORM_10K,
        section_type=SectionType.UNKNOWN,
        quarter=Quarter.Q4,
        year=2023,
        filed_date="2024-01-30",
        chunk_index=10,
        total_chunks=200,
        word_count=35,
        contains_numbers=False,
        chunking_strategy=ChunkingStrategy.SECTION_AWARE,
    )
    tagged_risk = tagger.tag(chunk_risk)
    assert tagged_risk.section_type == SectionType.RISK_FACTORS, (
        f"Expected RISK_FACTORS, got {tagged_risk.section_type}"
    )
    print(f"✓ Risk reclassification: UNKNOWN → {tagged_risk.section_type.value}")

    # Test 4: Batch tagging
    batch = [chunk_guidance, chunk_q1, chunk_risk]
    enriched = tagger.tag_batch(batch)
    assert len(enriched) == 3
    print(f"✓ Batch tagging: {len(enriched)} chunks processed")

    # Test 5: Quick sentiment
    sentiments = tagger.compute_quick_sentiment_labels(enriched)
    print(f"✓ Sentiment pre-labels: {sentiments}")

    print("\nmetadata_tagger smoke test passed ✓")
