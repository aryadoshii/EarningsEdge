"""
LangGraph node functions for the EarningsEdge RAG pipeline.

Each function in this module is a LangGraph node — it receives the current
EarningsEdgeState, performs one discrete operation, and returns a partial
state dict with only the fields it modifies.

Nodes (in execution order):
    query_classifier_node   — classify query intent
    company_retrieval_node  — retrieve company-specific chunks from ChromaDB
    gap_detector_node       — identify missing context
    industry_retrieval_node — retrieve peer company chunks if gaps exist
    macro_retrieval_node    — retrieve macro/sector context
    contradiction_check_node — run NLI on retrieved chunks
    synthesis_node          — generate structured LLM analysis
    quality_check_node      — verify answer is grounded in context
    ragas_prep_node         — format inputs for RAGAS evaluation

All nodes are async coroutines.  Nodes that call the LLM use llm_client.
Nodes that access ChromaDB use the retriever from src.embeddings.retriever.

Usage (called internally by graph.py):
    result = await company_retrieval_node(state)
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from config.settings import settings
from src.rag.llm_client import llm_client
from src.rag.prompts import (
    SYSTEM_EARNINGS_ANALYST,
    SYSTEM_GUIDANCE_EXTRACTOR,
    SYSTEM_QUALITY_CHECKER,
    SYSTEM_QUERY_CLASSIFIER,
    build_classification_prompt,
    build_gap_detection_prompt,
    build_quality_check_prompt,
    build_synthesis_prompt,
    format_contradictions,
)

# ---------------------------------------------------------------------------
# State type (imported from graph.py — defined here as a forward reference
# to avoid circular imports; the actual TypedDict lives in graph.py)
# ---------------------------------------------------------------------------

State = dict[str, Any]  # type alias for node signatures

# ---------------------------------------------------------------------------
# Valid query intent categories
# ---------------------------------------------------------------------------

_VALID_INTENTS = {
    "earnings_analysis",
    "risk_assessment",
    "guidance_tracking",
    "comparative_analysis",
    "tone_drift",
    "macro_context",
    "general",
}

# ---------------------------------------------------------------------------
# Node 1: Query classifier
# ---------------------------------------------------------------------------


async def query_classifier_node(state: State) -> State:
    """
    Classify the query intent to guide downstream retrieval strategy.

    Sends the query to the LLM with the classification prompt and parses
    the response.  Falls back to 'general' if classification fails.

    Modifies state:
        query_intent: str — one of _VALID_INTENTS

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict with query_intent set.
    """
    query = state.get("query", "")
    ticker = state.get("ticker", "")
    logger.info(f"[{ticker}] Classifying query: '{query[:80]}'")

    try:
        response = await llm_client.complete(
            system_prompt=SYSTEM_QUERY_CLASSIFIER,
            user_prompt=build_classification_prompt(query),
        )
        intent = response.content.strip().lower().rstrip(".")
        if intent not in _VALID_INTENTS:
            logger.warning(f"Unrecognised intent '{intent}' — defaulting to 'general'")
            intent = "general"
    except Exception as exc:
        logger.error(f"Query classification failed: {exc} — using 'general'")
        intent = "general"

    logger.info(f"[{ticker}] Query intent: {intent}")
    return {"query_intent": intent}


# ---------------------------------------------------------------------------
# Node 2: Company retrieval
# ---------------------------------------------------------------------------


async def company_retrieval_node(state: State) -> State:
    """
    Retrieve company-specific chunks from ChromaDB.

    Uses the query_intent to set metadata filters:
        earnings_analysis → filter to MDA + guidance + financial_statements
        risk_assessment   → filter to risk_factors
        guidance_tracking → filter to guidance section only
        tone_drift        → all sections, weighted toward prepared_remarks
        other             → no section filter

    Modifies state:
        retrieved_chunks:   list of chunk dicts
        company_context:    concatenated chunk texts for the LLM

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict.
    """
    ticker = state.get("ticker", "")
    query = state.get("query", "")
    intent = state.get("query_intent", "general")
    hop_count = state.get("hop_count", 0)

    logger.info(f"[{ticker}] Company retrieval (hop {hop_count + 1}, intent={intent})")

    # Build metadata filter based on intent
    metadata_filter = _build_metadata_filter(ticker, intent)

    try:
        from src.embeddings.retriever import retriever
        chunks = await retriever.retrieve(
            query=query,
            metadata_filter=metadata_filter,
            top_k=settings.TOP_K_RETRIEVAL,
        )
    except Exception as exc:
        logger.error(f"[{ticker}] Company retrieval failed: {exc}")
        chunks = []

    # Format for LLM context
    company_context = _format_context(chunks)
    logger.info(f"[{ticker}] Retrieved {len(chunks)} company chunks")

    return {
        "retrieved_chunks": chunks,
        "company_context": company_context,
        "hop_count": hop_count + 1,
    }


# ---------------------------------------------------------------------------
# Node 3: Gap detector
# ---------------------------------------------------------------------------


async def gap_detector_node(state: State) -> State:
    """
    Identify what information is missing from the retrieved context.

    Asks the LLM to analyse the retrieved context and flag gaps:
        - Missing time periods
        - Need for peer/industry data
        - Need for macro context
        - Missing financial metrics

    Modifies state:
        needs_more_retrieval: bool
        gap_analysis:         dict with gap details

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict.
    """
    ticker = state.get("ticker", "")
    query = state.get("query", "")
    company_context = state.get("company_context", "")
    hop_count = state.get("hop_count", 0)

    if hop_count >= settings.MAX_RAG_HOPS:
        logger.info(f"[{ticker}] Max hops reached — skipping gap detection")
        return {"needs_more_retrieval": False, "gap_analysis": {}}

    # Summarise the retrieved context for the gap prompt
    context_summary = company_context[:1500] if company_context else "No context retrieved."

    try:
        gap_data = await llm_client.complete_json(
            system_prompt="You are a retrieval gap analyser. Output only JSON.",
            user_prompt=build_gap_detection_prompt(ticker, query, context_summary),
        )
    except Exception as exc:
        logger.error(f"[{ticker}] Gap detection failed: {exc}")
        gap_data = {"has_gaps": False}

    has_gaps = bool(gap_data.get("has_gaps", False))
    needs_peer = bool(gap_data.get("needs_peer_data", False))
    needs_macro = bool(gap_data.get("needs_macro_data", False))

    needs_more = has_gaps and hop_count < settings.MAX_RAG_HOPS

    logger.info(
        f"[{ticker}] Gap analysis: has_gaps={has_gaps}  "
        f"peer={needs_peer}  macro={needs_macro}  "
        f"needs_more_retrieval={needs_more}"
    )

    return {
        "needs_more_retrieval": needs_more,
        "gap_analysis": gap_data,
    }


# ---------------------------------------------------------------------------
# Node 4: Industry retrieval
# ---------------------------------------------------------------------------


async def industry_retrieval_node(state: State) -> State:
    """
    Retrieve peer company chunks to fill comparative gaps.

    Only called when gap_detector identifies comparative analysis is needed.
    Retrieves chunks from known sector peers stored in ChromaDB.

    Modifies state:
        industry_context: str — peer context for LLM

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict.
    """
    ticker = state.get("ticker", "")
    query = state.get("query", "")
    gap_analysis = state.get("gap_analysis", {})

    logger.info(f"[{ticker}] Industry retrieval — supplementing with peer context")

    # Determine peer tickers — in production these come from a sector map
    # For now we use a simple hard-coded peer group
    peer_tickers = _get_peer_tickers(ticker)

    if not peer_tickers:
        logger.debug(f"[{ticker}] No peer tickers configured")
        return {"industry_context": ""}

    industry_chunks: list[dict[str, Any]] = []
    try:
        from src.embeddings.retriever import retriever
        for peer in peer_tickers[:3]:  # cap at 3 peers
            peer_filter = {"ticker": peer}
            peer_chunks = await retriever.retrieve(
                query=query,
                metadata_filter=peer_filter,
                top_k=3,  # fewer chunks per peer
            )
            industry_chunks.extend(peer_chunks)
    except Exception as exc:
        logger.error(f"[{ticker}] Industry retrieval failed: {exc}")

    industry_context = _format_context(industry_chunks, label_prefix="PEER")
    logger.info(
        f"[{ticker}] Retrieved {len(industry_chunks)} peer chunks "
        f"from {peer_tickers[:3]}"
    )

    return {"industry_context": industry_context}


# ---------------------------------------------------------------------------
# Node 5: Macro retrieval
# ---------------------------------------------------------------------------


async def macro_retrieval_node(state: State) -> State:
    """
    Retrieve macro / sector context chunks.

    Searches ChromaDB for documents tagged with ticker='__MACRO__' — a
    special namespace where Fed statements, sector reports, and economic
    data are stored during the ingest phase.

    Modifies state:
        macro_context: str

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict.
    """
    ticker = state.get("ticker", "")
    query = state.get("query", "")
    gap_analysis = state.get("gap_analysis", {})

    if not gap_analysis.get("needs_macro_data", False):
        return {"macro_context": ""}

    logger.info(f"[{ticker}] Macro retrieval")

    try:
        from src.embeddings.retriever import retriever
        macro_chunks = await retriever.retrieve(
            query=query,
            metadata_filter={"ticker": "__MACRO__"},
            top_k=5,
        )
    except Exception as exc:
        logger.error(f"Macro retrieval failed: {exc}")
        macro_chunks = []

    macro_context = _format_context(macro_chunks, label_prefix="MACRO")
    logger.info(f"Macro retrieval: {len(macro_chunks)} chunks")

    return {"macro_context": macro_context}


# ---------------------------------------------------------------------------
# Node 6: Contradiction check
# ---------------------------------------------------------------------------


async def contradiction_check_node(state: State) -> State:
    """
    Run NLI contradiction detection on the retrieved chunks.

    Uses the ContradictionDetector to check for conflicts between
    retrieved chunks.  This runs on the combined company + industry
    chunks to surface cross-period and cross-source contradictions.

    Modifies state:
        contradictions_found: list of contradiction dicts

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict.
    """
    ticker = state.get("ticker", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    if not retrieved_chunks:
        return {"contradictions_found": []}

    logger.info(f"[{ticker}] Running contradiction check on {len(retrieved_chunks)} chunks")

    try:
        from src.analysis.contradiction_detector import contradiction_detector
        from src.ingestion.data_validator import DocumentChunk

        # Convert dicts back to DocumentChunk objects if needed
        doc_chunks = _ensure_doc_chunks(retrieved_chunks)

        contradictions = contradiction_detector.detect(
            ticker=ticker,
            chunks=doc_chunks,
        )
        # Serialise to dicts for state storage
        contra_dicts = [
            {
                "chunk_a_source": c.chunk_a_source,
                "chunk_b_source": c.chunk_b_source,
                "contradiction_score": c.contradiction_score,
                "interpretation": c.interpretation,
                "chunk_a_text": c.chunk_a_text[:200],
                "chunk_b_text": c.chunk_b_text[:200],
            }
            for c in contradictions
        ]
    except Exception as exc:
        logger.error(f"[{ticker}] Contradiction check failed: {exc}")
        contra_dicts = []

    logger.info(f"[{ticker}] Contradictions found: {len(contra_dicts)}")
    return {"contradictions_found": contra_dicts}


# ---------------------------------------------------------------------------
# Node 7: Synthesis
# ---------------------------------------------------------------------------


async def synthesis_node(state: State) -> State:
    """
    Generate the structured earnings quality analysis.

    Combines all retrieved context, quality scores, tone drift data, and
    contradiction flags into the final structured LLM response.

    Modifies state:
        final_answer: str — the structured analysis

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict.
    """
    ticker = state.get("ticker", "")
    query = state.get("query", "")
    quality_score_obj = state.get("earnings_quality_score_obj")
    tone_drift_report = state.get("tone_drift_report", {})
    contradictions = state.get("contradictions_found", [])

    # Combine all context sources
    context_parts = []
    if state.get("company_context"):
        context_parts.append(f"=== COMPANY CONTEXT ({ticker}) ===\n{state['company_context']}")
    if state.get("industry_context"):
        context_parts.append(f"=== PEER INDUSTRY CONTEXT ===\n{state['industry_context']}")
    if state.get("macro_context"):
        context_parts.append(f"=== MACRO CONTEXT ===\n{state['macro_context']}")

    combined_context = "\n\n".join(context_parts) or "No context available."

    # Extract quality score fields
    if quality_score_obj:
        composite = quality_score_obj.get("composite_score", 0.0)
        signal = quality_score_obj.get("signal", "NEUTRAL")
        drift_comp = quality_score_obj.get("sentiment_drift_component", 0.0)
        guid_comp = quality_score_obj.get("guidance_accuracy_component", 0.0)
        accr_comp = quality_score_obj.get("accruals_component", 0.0)
        anal_comp = quality_score_obj.get("analyst_revision_component", 0.0)
    else:
        composite = drift_comp = guid_comp = accr_comp = anal_comp = 0.0
        signal = "NEUTRAL"

    alert_level = tone_drift_report.get("alert_level", "GREEN") if tone_drift_report else "GREEN"
    alert_reason = tone_drift_report.get("alert_reason", "") if tone_drift_report else ""

    # Build the synthesis prompt
    period = f"{state.get('quarter', 'Q?')} {state.get('year', '')}"
    user_prompt = build_synthesis_prompt(
        ticker=ticker,
        period=period,
        context=combined_context[:6000],  # fit in context window
        quality_score=composite,
        signal=signal,
        sentiment_drift_component=drift_comp,
        guidance_accuracy_component=guid_comp,
        accruals_component=accr_comp,
        analyst_revision_component=anal_comp,
        alert_level=alert_level,
        alert_reason=alert_reason,
        contradiction_count=len(contradictions),
        contradiction_summary=format_contradictions(contradictions),
    )

    logger.info(f"[{ticker}] Generating synthesis (context={len(combined_context)} chars)")

    try:
        response = await llm_client.complete(
            system_prompt=SYSTEM_EARNINGS_ANALYST,
            user_prompt=user_prompt,
        )
        answer = response.content
    except Exception as exc:
        logger.error(f"[{ticker}] Synthesis failed: {exc}")
        answer = f"Analysis generation failed: {exc}"

    return {
        "final_answer": answer,
        "context_used": combined_context[:3000],
    }


# ---------------------------------------------------------------------------
# Node 8: Quality check
# ---------------------------------------------------------------------------


async def quality_check_node(state: State) -> State:
    """
    Verify the generated answer is grounded in the retrieved context.

    Sends the answer + context to the LLM quality checker.  If the
    grounding score is below settings.QUALITY_GATE_THRESHOLD and max hops
    have not been reached, sets needs_more_retrieval=True to trigger
    another retrieval loop.

    Modifies state:
        quality_score:        float in [0, 1]
        needs_more_retrieval: bool

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict.
    """
    ticker = state.get("ticker", "")
    answer = state.get("final_answer", "")
    context = state.get("context_used", state.get("company_context", ""))
    hop_count = state.get("hop_count", 0)

    if not answer:
        return {"quality_score": 0.0, "needs_more_retrieval": False}

    try:
        result = await llm_client.complete_json(
            system_prompt=SYSTEM_QUALITY_CHECKER,
            user_prompt=build_quality_check_prompt(context, answer),
        )
        score = float(result.get("score", 0.5))
        issues = result.get("issues", [])
        verdict = result.get("verdict", "partially_grounded")
    except Exception as exc:
        logger.error(f"[{ticker}] Quality check failed: {exc}")
        score = 0.5
        issues = []
        verdict = "unknown"

    needs_more = (
        score < settings.QUALITY_GATE_THRESHOLD
        and hop_count < settings.MAX_RAG_HOPS
    )

    logger.info(
        f"[{ticker}] Quality check: score={score:.2f}  verdict={verdict}  "
        f"issues={len(issues)}  needs_more={needs_more}"
    )
    if issues:
        for issue in issues[:3]:
            logger.debug(f"  Grounding issue: {issue}")

    return {
        "quality_score": score,
        "needs_more_retrieval": needs_more,
    }


# ---------------------------------------------------------------------------
# Node 9: RAGAS preparation
# ---------------------------------------------------------------------------


async def ragas_prep_node(state: State) -> State:
    """
    Format all pipeline inputs/outputs for RAGAS evaluation.

    Packages the query, answer, retrieved contexts, and an auto-generated
    ground truth into the ragas_input dict.

    Modifies state:
        ragas_input: dict with keys query, answer, contexts, ground_truth

    Args:
        state: Current EarningsEdgeState.

    Returns:
        Partial state dict.
    """
    ticker = state.get("ticker", "")
    query = state.get("query", "")
    answer = state.get("final_answer", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Extract text from retrieved chunks
    contexts = [
        c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
        for c in retrieved_chunks[:10]
    ]
    contexts = [c for c in contexts if c]

    # Generate a ground truth summary using the LLM
    try:
        gt_response = await llm_client.complete(
            system_prompt="You are a concise financial summariser.",
            user_prompt=(
                f"In one sentence, state the factual ground truth that answers "
                f"this question: '{query}'\n\nBased on this answer:\n{answer[:500]}"
            ),
        )
        ground_truth = gt_response.content.strip()
    except Exception as exc:
        logger.warning(f"[{ticker}] Ground truth generation failed: {exc}")
        ground_truth = ""

    ragas_input = {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }

    logger.info(
        f"[{ticker}] RAGAS input prepared: "
        f"{len(contexts)} contexts, ground_truth={bool(ground_truth)}"
    )

    return {"ragas_input": ragas_input}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_metadata_filter(ticker: str, intent: str) -> dict[str, Any]:
    """
    Build a ChromaDB metadata filter dict from ticker and query intent.

    Args:
        ticker: Stock ticker.
        intent: Classified query intent.

    Returns:
        Filter dict passable to ChromaDB's where clause.
    """
    base: dict[str, Any] = {"ticker": ticker}

    section_filters = {
        "earnings_analysis": ["mda", "guidance", "financial_statements"],
        "risk_assessment":   ["risk_factors"],
        "guidance_tracking": ["guidance"],
        "tone_drift":        ["prepared_remarks", "mda", "qa"],
    }

    sections = section_filters.get(intent)
    if sections and settings.METADATA_FILTER_ENABLED:
        base["section_type"] = {"$in": sections}

    return base


def _format_context(
    chunks: list[Any],
    label_prefix: str = "SOURCE",
    max_chars_per_chunk: int = 600,
) -> str:
    """
    Format retrieved chunks into a single context string for the LLM.

    Each chunk is labelled with its source metadata.

    Args:
        chunks:              List of chunk dicts or DocumentChunk objects.
        label_prefix:        Prefix for source labels.
        max_chars_per_chunk: Truncation limit per chunk.

    Returns:
        Formatted multi-section context string.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            ticker = chunk.get("ticker", "?")
            filing_type = chunk.get("filing_type", "?")
            quarter = chunk.get("quarter", "?")
            year = chunk.get("year", "?")
            section = chunk.get("section_type", "?")
        else:
            text = getattr(chunk, "text", "")
            ticker = getattr(chunk, "ticker", "?")
            filing_type = getattr(getattr(chunk, "filing_type", "?"), "value", "?")
            quarter = getattr(getattr(chunk, "quarter", "?"), "value", "?")
            year = getattr(chunk, "year", "?")
            section = getattr(getattr(chunk, "section_type", "?"), "value", "?")

        label = (
            f"[{label_prefix} {i}: {ticker} {filing_type} "
            f"{quarter} {year} — {section}]"
        )
        parts.append(f"{label}\n{text[:max_chars_per_chunk]}")

    return "\n\n".join(parts)


def _ensure_doc_chunks(chunks: list[Any]) -> list[Any]:
    """
    Ensure chunks are DocumentChunk objects (not raw dicts).

    The state stores chunks as dicts for JSON serialisability.
    The contradiction detector needs DocumentChunk objects.

    Args:
        chunks: Mixed list of dicts or DocumentChunk objects.

    Returns:
        List where dict entries are converted to DocumentChunk.
    """
    from src.ingestion.data_validator import (
        ChunkingStrategy,
        DocumentChunk,
        FilingType,
        Quarter,
        SectionType,
    )

    result = []
    for c in chunks:
        if isinstance(c, dict):
            try:
                result.append(DocumentChunk(
                    chunk_id=c.get("chunk_id", "unknown"),
                    text=c.get("text", ""),
                    ticker=c.get("ticker", ""),
                    filing_type=FilingType(c.get("filing_type", "10-Q")),
                    section_type=SectionType(c.get("section_type", "unknown")),
                    quarter=Quarter(c.get("quarter", "Q1")),
                    year=int(c.get("year", 2024)),
                    filed_date=str(c.get("filed_date", "2024-01-01")),
                    chunk_index=int(c.get("chunk_index", 0)),
                    total_chunks=int(c.get("total_chunks", 1)),
                    word_count=int(c.get("word_count", 0)),
                    chunking_strategy=ChunkingStrategy(
                        c.get("chunking_strategy", "fixed_overlap")
                    ),
                ))
            except Exception:
                pass  # skip malformed chunks
        else:
            result.append(c)
    return result


# Minimal peer map — extended by analyst_fetcher in production
_PEER_MAP: dict[str, list[str]] = {
    "AAPL": ["MSFT", "GOOGL", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "AMZN"],
    "GOOGL": ["MSFT", "META", "AMZN"],
    "AMZN": ["MSFT", "GOOGL", "AAPL"],
    "META": ["GOOGL", "SNAP", "TWTR"],
    "NVDA": ["AMD", "INTC", "QCOM"],
    "TSLA": ["F", "GM", "RIVN"],
    "JPM":  ["BAC", "WFC", "GS"],
    "JNJ":  ["PFE", "MRK", "ABBV"],
}


def _get_peer_tickers(ticker: str) -> list[str]:
    """Return peer tickers for a given ticker from the peer map."""
    return _PEER_MAP.get(ticker.upper(), [])


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test helpers (no async/LLM needed)
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Revenue grew 8% to $94.9B. We expect Q4 revenue of $89B-$93B.",
            "ticker": "AAPL",
            "filing_type": "10-Q",
            "section_type": "guidance",
            "quarter": "Q3",
            "year": 2024,
            "filed_date": "2024-08-02",
        }
    ]

    # Test _format_context
    ctx = _format_context(chunks)
    assert "AAPL" in ctx and "SOURCE 1" in ctx
    print(f"_format_context: {len(ctx)} chars ✓")

    # Test _build_metadata_filter
    f1 = _build_metadata_filter("AAPL", "guidance_tracking")
    assert f1["ticker"] == "AAPL"
    assert "section_type" in f1
    print(f"_build_metadata_filter (guidance_tracking): {f1} ✓")

    f2 = _build_metadata_filter("MSFT", "general")
    assert "section_type" not in f2
    print(f"_build_metadata_filter (general): {f2} ✓")

    # Test _get_peer_tickers
    peers = _get_peer_tickers("AAPL")
    assert "MSFT" in peers
    print(f"_get_peer_tickers(AAPL): {peers} ✓")

    # Test format_contradictions (from prompts)
    contras = [
        {
            "contradiction_score": 0.87,
            "chunk_a_source": "AAPL transcript Q2 2024",
            "chunk_b_source": "AAPL 10-Q Q2 2024",
            "interpretation": "Demand described as 'robust' vs 'challenging'",
        }
    ]
    formatted = format_contradictions(contras)
    assert "⚠️" in formatted
    print(f"format_contradictions: {len(formatted)} chars ✓")

    print("\nnodes smoke test passed ✓")
