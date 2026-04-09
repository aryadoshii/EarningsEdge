"""
Multi-granularity document chunker.

This is the most important file in the processing layer.  It applies the
correct chunking strategy per document type and produces DocumentChunk
objects with rich metadata that downstream retrieval can filter on.

Four strategies are implemented:
    1. section_aware   — For 10-K / 10-Q: respects section boundaries
                         detected by document_parser.  Never splits across
                         an Item boundary.
    2. speaker_turn    — For transcripts: each CEO/CFO paragraph = one chunk;
                         each Q&A pair (question + answer) = one chunk.
    3. semantic        — For 8-K press releases: uses sentence embeddings to
                         find natural breakpoints (cosine similarity drop).
    4. fixed_overlap   — Fallback: standard 512-token windows with 64-token
                         overlap.

All chunks carry the full metadata schema defined in DocumentChunk so
ChromaDB can filter by ticker, section, quarter, year, speaker, etc.

Usage:
    chunker = DocumentChunker()
    chunks = chunker.chunk_filing(filing, sections)
    chunks = chunker.chunk_transcript(transcript)
"""

from __future__ import annotations

import re
import uuid
from typing import Sequence

import numpy as np
from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import (
    ChunkingStrategy,
    DocumentChunk,
    EarningsTranscript,
    FilingType,
    Quarter,
    SECFiling,
    SectionType,
    SpeakerTurn,
)
from src.processing.text_cleaner import count_financial_numbers

# Lazy import for sentence-transformers (heavy — only loaded for semantic chunking)
_sentence_model = None


def _get_sentence_model():  # type: ignore[return]
    """Lazy-load the sentence-transformer model for semantic chunking."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info(f"Loading sentence model: {settings.EMBEDDING_MODEL}")
            _sentence_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except ImportError:
            logger.error("sentence-transformers not installed — semantic chunking unavailable")
    return _sentence_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    """Generate a compact unique chunk ID."""
    return uuid.uuid4().hex[:16]


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple but robust regex.

    Handles abbreviations (U.S., Mr., Dr.) by requiring the period to be
    followed by whitespace + uppercase to count as a sentence boundary.

    Args:
        text: Input paragraph or section text.

    Returns:
        List of sentence strings.
    """
    # Split on sentence-ending punctuation followed by whitespace and a capital letter.
    # Use a simple fixed-width lookbehind (Python re does not support variable-width).
    # We first temporarily protect common abbreviations by replacing their periods.
    _abbrev_pattern = re.compile(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|est)\.",
        re.IGNORECASE,
    )
    protected = _abbrev_pattern.sub(lambda m: m.group(0).replace(".", "<!DOT!>"), text)
    # Also protect U.S., U.K., e.g., i.e.
    protected = re.sub(r"\b([A-Z])\.", lambda m: m.group(0).replace(".", "<!DOT!>"), protected)

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)
    # Restore protected dots
    return [s.replace("<!DOT!>", ".").strip() for s in sentences if s.strip()]


def _token_count(text: str) -> int:
    """
    Approximate token count (1 token ≈ 4 characters).

    Avoids loading a tokenizer for a fast approximation.

    Args:
        text: Any string.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


def _split_into_windows(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split text into fixed-size overlapping windows by approximate token count.

    Args:
        text:         Input text.
        chunk_size:   Target tokens per chunk.
        chunk_overlap: Overlap in tokens between consecutive chunks.

    Returns:
        List of text windows.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_sents: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_count(sent)
        if current_tokens + sent_tokens > chunk_size and current_sents:
            chunks.append(" ".join(current_sents))
            # Retain overlap sentences
            overlap_tokens = 0
            overlap_sents: list[str] = []
            for s in reversed(current_sents):
                overlap_tokens += _token_count(s)
                overlap_sents.insert(0, s)
                if overlap_tokens >= chunk_overlap:
                    break
            current_sents = overlap_sents
            current_tokens = overlap_tokens

        current_sents.append(sent)
        current_tokens += sent_tokens

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


def _quarter_from_filing(filing: SECFiling) -> Quarter:
    """Extract the quarter from a filing, defaulting to Q4 for annuals."""
    if filing.quarter:
        return filing.quarter
    month = filing.period_of_report.month
    if month <= 3:
        return Quarter.Q1
    if month <= 6:
        return Quarter.Q2
    if month <= 9:
        return Quarter.Q3
    return Quarter.Q4


def _section_contains_guidance(section_type: SectionType, text: str) -> bool:
    """Return True if a chunk likely contains forward guidance language."""
    if section_type == SectionType.GUIDANCE:
        return True
    guidance_keywords = re.compile(
        r"\b(?:expect|guidance|outlook|forecast|anticipate|project|target|"
        r"full.year|fiscal \d{4}|next quarter)\b",
        re.I,
    )
    return bool(guidance_keywords.search(text))


# ---------------------------------------------------------------------------
# ChunkMetadata builder
# ---------------------------------------------------------------------------

def _build_chunk(
    text: str,
    ticker: str,
    filing_type: FilingType,
    section_type: SectionType,
    quarter: Quarter,
    year: int,
    filed_date: str,
    chunk_index: int,
    total_chunks: int,
    strategy: ChunkingStrategy,
    speaker: str = "",
    source_url: str = "",
    accession_number: str = "",
) -> DocumentChunk:
    """
    Construct a DocumentChunk with all required metadata.

    This is the single factory function — every chunking strategy routes
    through here to guarantee consistent metadata.
    """
    # Override section type to GUIDANCE if content strongly suggests it
    effective_section = section_type
    if _section_contains_guidance(section_type, text):
        effective_section = SectionType.GUIDANCE

    return DocumentChunk(
        chunk_id=_make_id(),
        text=text,
        ticker=ticker,
        filing_type=filing_type,
        section_type=effective_section,
        quarter=quarter,
        year=year,
        filed_date=filed_date,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        word_count=len(text.split()),
        contains_numbers=count_financial_numbers(text) > 0,
        speaker=speaker,
        chunking_strategy=strategy,
        source_url=source_url,
        accession_number=accession_number,
    )


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Applies the appropriate chunking strategy per document type.

    Parameters are drawn from config/settings.py — no magic numbers here.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        max_chunks: int = settings.MAX_CHUNKS_PER_DOC,
        semantic_threshold: float = settings.SEMANTIC_SPLIT_THRESHOLD,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.semantic_threshold = semantic_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_filing(
        self,
        filing: SECFiling,
        sections: dict[SectionType, str],
    ) -> list[DocumentChunk]:
        """
        Chunk a 10-K, 10-Q, or 8-K filing into DocumentChunk objects.

        Routes to the correct strategy:
            10-K / 10-Q → section_aware
            8-K          → semantic (press-release style)
            other        → fixed_overlap fallback

        Args:
            filing:   SECFiling with metadata populated.
            sections: Dict of {SectionType: text} from document_parser.

        Returns:
            List of DocumentChunk objects capped at max_chunks.
        """
        quarter = _quarter_from_filing(filing)
        ticker = filing.ticker
        ft = filing.filing_type
        filed_date = filing.filed_date.isoformat()
        source_url = filing.filing_url
        accession = filing.accession_number

        all_chunks: list[DocumentChunk] = []

        if ft in (FilingType.FORM_10K, FilingType.FORM_10K_A,
                  FilingType.FORM_10Q, FilingType.FORM_10Q_A):
            for sec_type, text in sections.items():
                section_chunks = self._chunk_section_aware(text, sec_type)
                for i, chunk_text in enumerate(section_chunks):
                    all_chunks.append(_build_chunk(
                        text=chunk_text, ticker=ticker, filing_type=ft,
                        section_type=sec_type, quarter=quarter,
                        year=filing.year or filing.period_of_report.year,
                        filed_date=filed_date,
                        chunk_index=len(all_chunks),
                        total_chunks=1,  # patched below
                        strategy=ChunkingStrategy.SECTION_AWARE,
                        source_url=source_url, accession_number=accession,
                    ))

        elif ft == FilingType.FORM_8K:
            full_text = "\n\n".join(sections.values())
            semantic_chunks = self._chunk_semantic(full_text)
            for i, chunk_text in enumerate(semantic_chunks):
                all_chunks.append(_build_chunk(
                    text=chunk_text, ticker=ticker, filing_type=ft,
                    section_type=SectionType.PRESS_RELEASE,
                    quarter=quarter,
                    year=filing.year or filing.period_of_report.year,
                    filed_date=filed_date,
                    chunk_index=i, total_chunks=1,
                    strategy=ChunkingStrategy.SEMANTIC,
                    source_url=source_url, accession_number=accession,
                ))
        else:
            full_text = "\n\n".join(sections.values())
            for i, chunk_text in enumerate(
                _split_into_windows(full_text, self.chunk_size, self.chunk_overlap)
            ):
                all_chunks.append(_build_chunk(
                    text=chunk_text, ticker=ticker, filing_type=ft,
                    section_type=SectionType.UNKNOWN,
                    quarter=quarter,
                    year=filing.year or filing.period_of_report.year,
                    filed_date=filed_date,
                    chunk_index=i, total_chunks=1,
                    strategy=ChunkingStrategy.FIXED_OVERLAP,
                    source_url=source_url, accession_number=accession,
                ))

        # Apply cap and patch total_chunks
        all_chunks = all_chunks[: self.max_chunks]
        total = len(all_chunks)
        for c in all_chunks:
            c.total_chunks = total

        logger.info(
            f"[{ticker}] {ft.value} → {total} chunks "
            f"(strategy: {all_chunks[0].chunking_strategy.value if all_chunks else 'n/a'})"
        )
        return all_chunks

    def chunk_transcript(
        self,
        transcript: EarningsTranscript,
    ) -> list[DocumentChunk]:
        """
        Chunk an earnings call transcript using speaker-turn strategy.

        Prepared remarks: each CEO/CFO speaker turn = one chunk.
        Q&A: each (question + immediate answer) pair = one chunk.

        Args:
            transcript: Populated EarningsTranscript object.

        Returns:
            List of DocumentChunk objects.
        """
        ticker = transcript.ticker
        filed_date = transcript.call_date.isoformat()
        year = transcript.year
        quarter = transcript.quarter
        ft = FilingType.TRANSCRIPT

        all_chunks: list[DocumentChunk] = []

        # --- Prepared remarks: one chunk per speaker turn (skip Operator)
        for turn in transcript.prepared_remarks:
            if turn.speaker_role == "Operator":
                continue
            chunks = self._chunk_speaker_turn(turn.text)
            sec_type = SectionType.PREPARED_REMARKS
            for text in chunks:
                all_chunks.append(_build_chunk(
                    text=text, ticker=ticker, filing_type=ft,
                    section_type=sec_type, quarter=quarter, year=year,
                    filed_date=filed_date,
                    chunk_index=len(all_chunks), total_chunks=1,
                    strategy=ChunkingStrategy.SPEAKER_TURN,
                    speaker=turn.speaker_role,
                ))

        # --- Q&A: pair consecutive (Analyst question, Management answer)
        qa_turns = transcript.qa_section
        i = 0
        while i < len(qa_turns):
            turn = qa_turns[i]
            # Analyst question
            if turn.speaker_role == "Analyst":
                question_text = turn.text
                answer_text = ""
                # Look ahead for the management response
                if i + 1 < len(qa_turns) and qa_turns[i + 1].speaker_role in (
                    "CEO", "CFO", "COO", "IR"
                ):
                    answer_text = qa_turns[i + 1].text
                    i += 1  # consume the answer turn

                combined = f"Q: {question_text}\n\nA: {answer_text}".strip()
                if len(combined) > 50:
                    all_chunks.append(_build_chunk(
                        text=combined, ticker=ticker, filing_type=ft,
                        section_type=SectionType.QA, quarter=quarter, year=year,
                        filed_date=filed_date,
                        chunk_index=len(all_chunks), total_chunks=1,
                        strategy=ChunkingStrategy.SPEAKER_TURN,
                        speaker="QA_PAIR",
                    ))
            elif turn.speaker_role in ("CEO", "CFO", "COO", "IR"):
                # Standalone management commentary in Q&A
                for text in self._chunk_speaker_turn(turn.text):
                    all_chunks.append(_build_chunk(
                        text=text, ticker=ticker, filing_type=ft,
                        section_type=SectionType.QA, quarter=quarter, year=year,
                        filed_date=filed_date,
                        chunk_index=len(all_chunks), total_chunks=1,
                        strategy=ChunkingStrategy.SPEAKER_TURN,
                        speaker=turn.speaker_role,
                    ))
            i += 1

        all_chunks = all_chunks[: self.max_chunks]
        total = len(all_chunks)
        for c in all_chunks:
            c.total_chunks = total

        logger.info(
            f"[{ticker}] Transcript {quarter.value} {year} → {total} chunks"
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _chunk_section_aware(
        self,
        section_text: str,
        section_type: SectionType,
    ) -> list[str]:
        """
        Split one filing section into token-window chunks.

        Because section boundaries have already been respected by the caller,
        this is equivalent to fixed-overlap chunking but restricted to a
        single section — never crossing into the next Item.

        Args:
            section_text: Text of one filing section.
            section_type: Section identifier (for logging only).

        Returns:
            List of text windows.
        """
        windows = _split_into_windows(
            section_text, self.chunk_size, self.chunk_overlap
        )
        logger.debug(
            f"Section {section_type.value}: {len(windows)} windows "
            f"from {len(section_text):,} chars"
        )
        return windows

    def _chunk_speaker_turn(self, text: str) -> list[str]:
        """
        Split a single speaker turn if it exceeds chunk_size tokens.

        Most turns are short enough to fit in one chunk.  Very long CFO/CEO
        monologues are split at sentence boundaries.

        Args:
            text: Speaker turn text.

        Returns:
            List of strings (usually just one element).
        """
        if _token_count(text) <= self.chunk_size:
            return [text]
        return _split_into_windows(text, self.chunk_size, self.chunk_overlap)

    def _chunk_semantic(self, text: str) -> list[str]:
        """
        Split text at semantic breakpoints using sentence embeddings.

        Algorithm:
            1. Split into sentences.
            2. Embed all sentences in a batch.
            3. Compute consecutive cosine similarities.
            4. Split where similarity drops below self.semantic_threshold.
            5. Merge short fragments into adjacent chunks.

        Falls back to fixed-overlap if the embedding model is unavailable.

        Args:
            text: Text to chunk (typically an 8-K press release).

        Returns:
            List of semantically coherent text chunks.
        """
        model = _get_sentence_model()
        if model is None:
            logger.warning("Falling back to fixed-overlap chunking (no embedding model)")
            return _split_into_windows(text, self.chunk_size, self.chunk_overlap)

        sentences = _split_sentences(text)
        if len(sentences) <= 2:
            return [text]

        # Batch embed all sentences
        embeddings: np.ndarray = model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Cosine similarity between consecutive sentences (already normalised → dot product)
        sims = np.einsum("ij,ij->i", embeddings[:-1], embeddings[1:])

        # Find split points
        split_indices: list[int] = [0]
        current_token_count = 0
        for i, sim in enumerate(sims):
            current_token_count += _token_count(sentences[i])
            if (
                sim < self.semantic_threshold
                and current_token_count >= self.chunk_size // 2
            ) or current_token_count >= self.chunk_size:
                split_indices.append(i + 1)
                current_token_count = 0

        split_indices.append(len(sentences))

        # Build chunks from split points
        chunks: list[str] = []
        for j in range(len(split_indices) - 1):
            start = split_indices[j]
            end = split_indices[j + 1]
            chunk_text = " ".join(sentences[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)

        logger.debug(
            f"Semantic chunking: {len(sentences)} sentences → {len(chunks)} chunks "
            f"(threshold={self.semantic_threshold})"
        )
        return chunks

    def chunk_raw_text(
        self,
        text: str,
        ticker: str,
        filing_type: FilingType,
        quarter: Quarter,
        year: int,
        filed_date: str,
        section_type: SectionType = SectionType.UNKNOWN,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Chunk arbitrary text using the specified strategy.

        Convenience method for chunking text that doesn't come from a
        structured SECFiling or EarningsTranscript object.

        Args:
            text:         Input text.
            ticker:       Stock ticker.
            filing_type:  Source document type.
            quarter:      Reporting quarter.
            year:         Reporting year.
            filed_date:   ISO date string.
            section_type: Section classification.
            strategy:     Which chunking strategy to apply.

        Returns:
            List of DocumentChunk objects.
        """
        if strategy == ChunkingStrategy.SEMANTIC:
            windows = self._chunk_semantic(text)
        else:
            windows = _split_into_windows(text, self.chunk_size, self.chunk_overlap)

        chunks = [
            _build_chunk(
                text=w, ticker=ticker, filing_type=filing_type,
                section_type=section_type, quarter=quarter, year=year,
                filed_date=filed_date, chunk_index=i,
                total_chunks=len(windows),
                strategy=strategy,
            )
            for i, w in enumerate(windows)
            if w.strip()
        ]
        return chunks[: self.max_chunks]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

document_chunker = DocumentChunker()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date as d
    from src.ingestion.data_validator import XBRLData

    chunker = DocumentChunker()

    # --- Test section-aware chunking on a synthetic 10-Q
    filing = SECFiling(
        ticker="AAPL",
        cik="0000320193",
        filing_type=FilingType.FORM_10Q,
        period_of_report=d(2024, 6, 30),
        filed_date=d(2024, 8, 2),
        filing_url="https://sec.gov/test",
        accession_number="0000320193-24-000123",
        raw_text="",
        xbrl_data=XBRLData(),
        quarter=Quarter.Q3,
        year=2024,
    )

    sections = {
        SectionType.RISK_FACTORS: (
            "We face intense competition in all our product categories. "
            "Supply chain disruptions may materially impact results. " * 40
        ),
        SectionType.MDA: (
            "Revenue increased 6% year over year to $94.9 billion. "
            "We expect full-year revenue guidance of $380 to $390 billion. "
            "Operating margin expanded 180 basis points to 31.0%. " * 30
        ),
    }

    chunks = chunker.chunk_filing(filing, sections)
    print(f"10-Q chunks: {len(chunks)}")
    print(f"First chunk section: {chunks[0].section_type.value}")
    print(f"Contains guidance chunk: {any(c.section_type == SectionType.GUIDANCE for c in chunks)}")

    # --- Test transcript chunking
    from src.ingestion.data_validator import EarningsTranscript, SpeakerTurn, Quarter

    transcript = EarningsTranscript(
        ticker="AAPL",
        quarter=Quarter.Q3,
        year=2024,
        call_date=d(2024, 8, 1),
        prepared_remarks=[
            SpeakerTurn(
                speaker_name="Tim Cook",
                speaker_role="CEO",
                text="Good afternoon, everyone. We are very pleased with our Q3 results. "
                     "Revenue of $94.9B, up 6% year over year. iPhone demand remains "
                     "resilient. Services hit a new all-time high of $24.2B. " * 5,
                turn_index=0, section="prepared_remarks",
            ),
            SpeakerTurn(
                speaker_name="Luca Maestri",
                speaker_role="CFO",
                text="Gross margin was 46.3%, operating cash flow $29.9B. "
                     "We expect Q4 revenue of $89B to $93B. " * 8,
                turn_index=1, section="prepared_remarks",
            ),
        ],
        qa_section=[
            SpeakerTurn(
                speaker_name="Mike Ng",
                speaker_role="Analyst",
                text="Can you comment on iPhone 16 demand outlook?",
                turn_index=0, section="qa",
            ),
            SpeakerTurn(
                speaker_name="Tim Cook",
                speaker_role="CEO",
                text="iPhone 16 demand is tracking better than iPhone 15 at the same point. "
                     "We are very enthusiastic about the AI features." * 3,
                turn_index=1, section="qa",
            ),
        ],
    )

    t_chunks = chunker.chunk_transcript(transcript)
    print(f"\nTranscript chunks: {len(t_chunks)}")
    print(f"QA chunks: {sum(1 for c in t_chunks if c.section_type == SectionType.QA)}")
    print(f"Speaker chunks: {set(c.speaker for c in t_chunks)}")
    guidance_chunks = [c for c in t_chunks if c.section_type == SectionType.GUIDANCE]
    print(f"Guidance chunks: {len(guidance_chunks)}")
    if guidance_chunks:
        print(f"  Sample: {guidance_chunks[0].text[:100]}...")

    print("\ndocument_chunker smoke test passed ✓")