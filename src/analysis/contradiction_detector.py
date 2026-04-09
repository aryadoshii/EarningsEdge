"""
NLI-based contradiction detector.

Uses cross-encoder/nli-deberta-v3-base to identify contradictions between
document chunks from the same ticker but different time periods or document
types.  This catches:

    - Management statements in transcripts that contradict the 10-Q MD&A
      filed the same quarter
    - Forward guidance in Q1 that was reversed by Q2 disclosures
    - Risk-factor language that contradicts earnings optimism

Algorithm:
    1. Retrieve all chunks for a ticker grouped by (quarter, year)
    2. For every pair of chunks across DIFFERENT quarters that discuss the
       same topic (cosine similarity > CONTRADICTION_SIMILARITY_THRESHOLD):
         a. Run NLI cross-encoder: ENTAILMENT / NEUTRAL / CONTRADICTION
         b. If CONTRADICTION score > CONTRADICTION_SCORE_THRESHOLD: flag it
    3. Special case: compare transcript prepared_remarks vs 10-Q MD&A for
       the SAME quarter — management should be internally consistent

The NLI model is loaded lazily and cached globally.

Usage:
    detector = ContradictionDetector()
    contradictions = detector.detect(ticker, chunks, embeddings)
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import (
    Contradiction,
    DocumentChunk,
    FilingType,
    SectionType,
)

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_nli_model = None
_embed_model = None


def _get_nli_model() -> Any:
    """Lazy-load the NLI cross-encoder."""
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            logger.info(f"Loading NLI model: {settings.NLI_MODEL}")
            _nli_model = CrossEncoder(
                settings.NLI_MODEL,
                num_labels=3,
                device="cpu",
            )
            logger.info("NLI cross-encoder loaded ✓")
        except Exception as exc:
            logger.error(f"Failed to load NLI model: {exc}")
    return _nli_model


def _get_embed_model() -> Any:
    """Lazy-load the embedding model for chunk similarity filtering."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info(f"Loading embedding model for similarity: {settings.EMBEDDING_MODEL}")
            _embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except Exception as exc:
            logger.error(f"Failed to load embedding model: {exc}")
    return _embed_model


# ---------------------------------------------------------------------------
# NLI label mapping
# ---------------------------------------------------------------------------

# DeBERTa NLI models output 3 logits: [CONTRADICTION, NEUTRAL, ENTAILMENT]
# Index 0 = contradiction, 1 = neutral, 2 = entailment
_CONTRADICTION_IDX = 0
_NEUTRAL_IDX = 1
_ENTAILMENT_IDX = 2


def _softmax(logits: list[float]) -> list[float]:
    """Convert raw logits to probabilities via softmax."""
    arr = np.array(logits, dtype=np.float64)
    arr -= arr.max()
    exp = np.exp(arr)
    return (exp / exp.sum()).tolist()


def _nli_scores(
    model: Any,
    premise: str,
    hypothesis: str,
) -> tuple[float, float, float]:
    """
    Score a (premise, hypothesis) pair with the NLI cross-encoder.

    Args:
        model:      Loaded CrossEncoder model.
        premise:    First text (typically the older chunk).
        hypothesis: Second text (typically the newer chunk).

    Returns:
        Tuple (contradiction_score, neutral_score, entailment_score),
        each in [0, 1] and summing to 1.
    """
    try:
        raw = model.predict([(premise, hypothesis)])
        probs = _softmax(raw[0].tolist())
        return probs[_CONTRADICTION_IDX], probs[_NEUTRAL_IDX], probs[_ENTAILMENT_IDX]
    except Exception as exc:
        logger.debug(f"NLI prediction error: {exc}")
        return 0.0, 1.0, 0.0


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-norm vectors."""
    return float(np.dot(a, b))


def _chunk_source_label(chunk: DocumentChunk) -> str:
    """Human-readable source label for a chunk (used in Contradiction objects)."""
    return (
        f"{chunk.ticker} {chunk.filing_type.value} "
        f"{chunk.quarter.value} {chunk.year} — {chunk.section_type.value}"
    )


# ---------------------------------------------------------------------------
# ContradictionDetector
# ---------------------------------------------------------------------------

class ContradictionDetector:
    """
    Detects contradictions between document chunks using NLI scoring.

    Attributes:
        similarity_threshold: Min cosine similarity to consider a pair
                               topically related (default from settings).
        contradiction_threshold: Min contradiction score to flag a conflict
                                  (default from settings).
        max_pairs_per_ticker: Cap on candidate pairs to evaluate per ticker
                               (avoids O(n²) blow-up on large corpora).
    """

    def __init__(
        self,
        similarity_threshold: float = settings.CONTRADICTION_SIMILARITY_THRESHOLD,
        contradiction_threshold: float = settings.CONTRADICTION_SCORE_THRESHOLD,
        max_pairs_per_ticker: int = 500,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.max_pairs_per_ticker = max_pairs_per_ticker

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        ticker: str,
        chunks: list[DocumentChunk],
        precomputed_embeddings: dict[str, np.ndarray] | None = None,
    ) -> list[Contradiction]:
        """
        Detect contradictions across all chunks for a given ticker.

        Args:
            ticker:                  Stock ticker (upper case).
            chunks:                  All chunks for this ticker.
            precomputed_embeddings:  Optional {chunk_id: embedding array}.
                                     If None, embeddings are computed here.

        Returns:
            List of Contradiction objects, sorted by contradiction_score desc.
        """
        ticker_chunks = [c for c in chunks if c.ticker == ticker.upper()]
        if len(ticker_chunks) < 2:
            return []

        logger.info(
            f"[{ticker}] Contradiction detection: {len(ticker_chunks)} chunks"
        )

        # Get or compute embeddings
        embeddings = precomputed_embeddings or self._embed_chunks(ticker_chunks)
        if not embeddings:
            logger.warning(f"[{ticker}] No embeddings available — skipping contradiction detection")
            return []

        # Build candidate pairs
        cross_quarter_pairs = self._get_cross_quarter_candidates(
            ticker_chunks, embeddings
        )
        intra_quarter_pairs = self._get_intra_quarter_candidates(ticker_chunks)

        all_pairs = cross_quarter_pairs + intra_quarter_pairs
        # Deduplicate
        seen: set[tuple[str, str]] = set()
        unique_pairs: list[tuple[DocumentChunk, DocumentChunk]] = []
        for a, b in all_pairs:
            key = (min(a.chunk_id, b.chunk_id), max(a.chunk_id, b.chunk_id))
            if key not in seen:
                seen.add(key)
                unique_pairs.append((a, b))

        # Cap to avoid excessive inference time
        if len(unique_pairs) > self.max_pairs_per_ticker:
            logger.warning(
                f"[{ticker}] {len(unique_pairs)} candidate pairs — "
                f"capping to {self.max_pairs_per_ticker}"
            )
            unique_pairs = unique_pairs[: self.max_pairs_per_ticker]

        logger.info(f"[{ticker}] Evaluating {len(unique_pairs)} candidate pairs")
        contradictions = self._score_pairs(unique_pairs)

        logger.info(
            f"[{ticker}] Found {len(contradictions)} contradiction(s) "
            f"above threshold {self.contradiction_threshold}"
        )
        return sorted(contradictions, key=lambda c: c.contradiction_score, reverse=True)

    # ------------------------------------------------------------------
    # Candidate pair selection
    # ------------------------------------------------------------------

    def _embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> dict[str, np.ndarray]:
        """
        Compute dense embeddings for all chunks using the embedding model.

        Args:
            chunks: Chunks to embed.

        Returns:
            Dict {chunk_id: unit-norm embedding array}.
        """
        model = _get_embed_model()
        if model is None:
            return {}

        texts = [c.text for c in chunks]
        try:
            vecs = model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return {c.chunk_id: vecs[i] for i, c in enumerate(chunks)}
        except Exception as exc:
            logger.error(f"Embedding failed during contradiction detection: {exc}")
            return {}

    def _get_cross_quarter_candidates(
        self,
        chunks: list[DocumentChunk],
        embeddings: dict[str, np.ndarray],
    ) -> list[tuple[DocumentChunk, DocumentChunk]]:
        """
        Find topically similar chunks from DIFFERENT quarters.

        Only chunks with precomputed embeddings are considered.  Pairs
        are filtered by cosine similarity ≥ self.similarity_threshold.

        Args:
            chunks:     All chunks for one ticker.
            embeddings: Precomputed embeddings by chunk_id.

        Returns:
            List of (chunk_a, chunk_b) candidate pairs.
        """
        # Only use chunks that have embeddings
        # Exclude financial_statements — numbers differ every quarter by design,
        # causing NLI to flag expected numeric changes as contradictions.
        # Meaningful contradictions only exist in language-driven sections.
        _LANGUAGE_SECTIONS = {
            SectionType.MDA,
            SectionType.RISK_FACTORS,
            SectionType.GUIDANCE,
            SectionType.PREPARED_REMARKS,
            SectionType.QA,
            SectionType.PRESS_RELEASE,
        }
        embedded = [
            c for c in chunks
            if c.chunk_id in embeddings and c.section_type in _LANGUAGE_SECTIONS
        ]
        pairs: list[tuple[DocumentChunk, DocumentChunk]] = []

        for ca, cb in combinations(embedded, 2):
            # Must be from different time periods
            if (ca.quarter, ca.year) == (cb.quarter, cb.year):
                continue
            # Prefer substantive sections (skip cover pages, standalone numbers)
            if ca.word_count < 30 or cb.word_count < 30:
                continue

            sim = _cosine_sim(embeddings[ca.chunk_id], embeddings[cb.chunk_id])
            if sim >= self.similarity_threshold:
                # Put older chunk first (as premise)
                if (ca.year, _quarter_ord(ca.quarter)) > (cb.year, _quarter_ord(cb.quarter)):
                    ca, cb = cb, ca
                pairs.append((ca, cb))

        logger.debug(
            f"Cross-quarter candidate pairs: {len(pairs)} "
            f"(similarity ≥ {self.similarity_threshold})"
        )
        return pairs

    def _get_intra_quarter_candidates(
        self,
        chunks: list[DocumentChunk],
    ) -> list[tuple[DocumentChunk, DocumentChunk]]:
        """
        Find transcript vs 10-Q pairs for the SAME quarter.

        Management should be internally consistent between what they say
        on the earnings call (transcript) and what they file in the 10-Q.
        Intra-quarter discrepancies are high-quality signals.

        Args:
            chunks: All chunks for one ticker.

        Returns:
            List of (transcript_chunk, filing_chunk) pairs.
        """
        # Group by (quarter, year)
        by_period: dict[
            tuple, dict[str, list[DocumentChunk]]
        ] = {}
        for c in chunks:
            key = (c.quarter, c.year)
            if key not in by_period:
                by_period[key] = {"transcript": [], "filing": []}
            if c.filing_type == FilingType.TRANSCRIPT:
                by_period[key]["transcript"].append(c)
            elif c.filing_type in (FilingType.FORM_10Q, FilingType.FORM_10K):
                by_period[key]["filing"].append(c)

        pairs: list[tuple[DocumentChunk, DocumentChunk]] = []
        for period, groups in by_period.items():
            t_chunks = groups["transcript"]
            f_chunks = [
                c for c in groups["filing"]
                if c.section_type in (SectionType.MDA, SectionType.GUIDANCE)
            ]
            if not t_chunks or not f_chunks:
                continue

            # Match transcript prepared remarks with filing MD&A sections
            for tc in t_chunks:
                if tc.section_type not in (
                    SectionType.PREPARED_REMARKS, SectionType.GUIDANCE
                ):
                    continue
                for fc in f_chunks:
                    pairs.append((tc, fc))

        logger.debug(f"Intra-quarter candidate pairs: {len(pairs)}")
        return pairs

    # ------------------------------------------------------------------
    # NLI scoring
    # ------------------------------------------------------------------

    def _score_pairs(
        self,
        pairs: list[tuple[DocumentChunk, DocumentChunk]],
    ) -> list[Contradiction]:
        """
        Run NLI cross-encoder on all candidate pairs.

        Pairs with contradiction score below threshold are discarded.

        Args:
            pairs: List of (premise_chunk, hypothesis_chunk) tuples.

        Returns:
            List of Contradiction objects for pairs exceeding threshold.
        """
        model = _get_nli_model()
        if model is None:
            logger.warning("NLI model unavailable — skipping contradiction scoring")
            return []

        contradictions: list[Contradiction] = []

        for premise_chunk, hyp_chunk in pairs:
            # Truncate texts to avoid excessively long cross-encoder inputs
            premise_text = premise_chunk.text[:512]
            hyp_text = hyp_chunk.text[:512]

            contra, neutral, entail = _nli_scores(model, premise_text, hyp_text)

            if contra >= self.contradiction_threshold:
                interpretation = self._interpret_contradiction(
                    premise_chunk, hyp_chunk, contra
                )
                contradictions.append(Contradiction(
                    chunk_a_id=premise_chunk.chunk_id,
                    chunk_b_id=hyp_chunk.chunk_id,
                    chunk_a_text=premise_chunk.text[:500],
                    chunk_b_text=hyp_chunk.text[:500],
                    chunk_a_source=_chunk_source_label(premise_chunk),
                    chunk_b_source=_chunk_source_label(hyp_chunk),
                    contradiction_score=round(contra, 4),
                    entailment_score=round(entail, 4),
                    neutral_score=round(neutral, 4),
                    interpretation=interpretation,
                ))
                logger.debug(
                    f"Contradiction detected: {_chunk_source_label(premise_chunk)} "
                    f"vs {_chunk_source_label(hyp_chunk)} "
                    f"— score={contra:.3f}"
                )

        return contradictions

    # ------------------------------------------------------------------
    # Interpretation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret_contradiction(
        chunk_a: DocumentChunk,
        chunk_b: DocumentChunk,
        score: float,
    ) -> str:
        """
        Generate a rule-based plain-English interpretation of a contradiction.

        This avoids an LLM call for every pair — interpretations are
        templated from the chunk metadata.  The synthesis_node in the RAG
        pipeline can enrich these with LLM-generated explanations.

        Args:
            chunk_a: Premise chunk.
            chunk_b: Hypothesis chunk.
            score:   NLI contradiction score.

        Returns:
            Interpretation string.
        """
        severity = "high" if score > 0.85 else "moderate"
        period_a = f"{chunk_a.quarter.value} {chunk_a.year}"
        period_b = f"{chunk_b.quarter.value} {chunk_b.year}"

        # Same quarter (intra-quarter transcript vs filing)
        if (chunk_a.quarter, chunk_a.year) == (chunk_b.quarter, chunk_b.year):
            return (
                f"[{severity.upper()} CONFIDENCE] Management statements in the "
                f"{chunk_a.filing_type.value} ({chunk_a.section_type.value}) "
                f"appear to contradict the "
                f"{chunk_b.filing_type.value} ({chunk_b.section_type.value}) "
                f"for {period_a}. Score: {score:.2f}."
            )

        # Cross-quarter
        return (
            f"[{severity.upper()} CONFIDENCE] Statements in {period_a} "
            f"({chunk_a.section_type.value}) appear to contradict "
            f"{period_b} ({chunk_b.section_type.value}). "
            f"Score: {score:.2f}. "
            f"This may indicate a material change in business conditions "
            f"or inconsistent management communication."
        )

    # ------------------------------------------------------------------
    # Convenience: batch detection for multiple tickers
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        chunks: list[DocumentChunk],
        precomputed_embeddings: dict[str, np.ndarray] | None = None,
    ) -> dict[str, list[Contradiction]]:
        """
        Run contradiction detection for all unique tickers in chunks.

        Args:
            chunks:                 Mixed-ticker chunk list.
            precomputed_embeddings: Optional precomputed embeddings.

        Returns:
            Dict {ticker: [Contradiction, ...]} for each ticker.
        """
        tickers = sorted({c.ticker for c in chunks})
        results: dict[str, list[Contradiction]] = {}
        for ticker in tickers:
            results[ticker] = self.detect(ticker, chunks, precomputed_embeddings)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUARTER_ORD: dict[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Annual": 5}


def _quarter_ord(q: Any) -> int:
    return _QUARTER_ORD.get(q.value if hasattr(q, "value") else str(q), 0)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

contradiction_detector = ContradictionDetector()


# ---------------------------------------------------------------------------
# Smoke test — logic only, no model inference
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date as d
    from src.ingestion.data_validator import ChunkingStrategy, Quarter

    def _chunk(cid: str, text: str, q: Quarter, yr: int,
                ft: FilingType, sec: SectionType) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=cid, text=text, ticker="AAPL",
            filing_type=ft, section_type=sec,
            quarter=q, year=yr, filed_date=f"{yr}-08-01",
            chunk_index=0, total_chunks=10, word_count=len(text.split()),
            contains_numbers=True,
            chunking_strategy=ChunkingStrategy.SECTION_AWARE,
        )

    chunks = [
        _chunk("t1", "iPhone demand is extremely strong. Revenue guidance of $93B.",
               Quarter.Q2, 2024, FilingType.TRANSCRIPT, SectionType.PREPARED_REMARKS),
        _chunk("t2", "We expect revenue between $89B and $93B next quarter.",
               Quarter.Q2, 2024, FilingType.FORM_10Q, SectionType.GUIDANCE),
        _chunk("t3", "Revenue declined significantly. Demand is very weak.",
               Quarter.Q3, 2024, FilingType.FORM_10Q, SectionType.MDA),
        _chunk("t4", "Gross margin was 46.3% for the quarter ended June 30.",
               Quarter.Q2, 2024, FilingType.FORM_10Q, SectionType.FINANCIAL_STATEMENTS),
    ]

    # Test intra-quarter candidate identification
    detector = ContradictionDetector()
    intra = detector._get_intra_quarter_candidates(chunks)
    print(f"Intra-quarter candidates: {len(intra)}")
    assert len(intra) >= 1, "Expected at least 1 intra-quarter pair"

    # Test source label formatting
    label = _chunk_source_label(chunks[0])
    assert "AAPL" in label and "Q2" in label
    print(f"Source label: '{label}' ✓")

    # Test interpretation generation
    interp = ContradictionDetector._interpret_contradiction(chunks[0], chunks[2], 0.87)
    assert "HIGH CONFIDENCE" in interp
    assert "Q2" in interp and "Q3" in interp
    print(f"Interpretation: '{interp[:80]}...' ✓")

    # Test softmax
    logits = [2.5, 0.5, 0.3]
    probs = _softmax(logits)
    assert abs(sum(probs) - 1.0) < 1e-5
    assert probs[0] > probs[1] > probs[2]
    print(f"Softmax: {[round(p, 3) for p in probs]} ✓")

    print("\ncontradiction_detector smoke test passed ✓")
