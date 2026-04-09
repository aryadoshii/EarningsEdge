"""
LangGraph state machine for the EarningsEdge RAG pipeline.

Defines EarningsEdgeState and wires all nodes from nodes.py into a
directed graph with conditional edges that implement multi-hop retrieval.

Graph topology:
    START
      │
      ▼
    query_classifier ──────────────────────────────────────────────────┐
      │                                                                 │
      ▼                                                                 │
    company_retrieval                                                   │
      │                                                                 │
      ▼                                                                 │
    gap_detector ──── needs_more=False ──────────────────────────────► │
      │                                                                 │
      │ needs_more=True                                                 │
      ▼                                                                 │
    industry_retrieval                                                  │
      │                                                                 │
      ▼                                                                 │
    macro_retrieval                                                     │
      │                                                                 │
      ◄────────────────────────────────────────────────────────────────┘
      │
      ▼
    contradiction_check
      │
      ▼
    synthesis
      │
      ▼
    quality_check ──── score≥threshold ──────────────────────────────► ragas_prep ──► END
      │
      │ score<threshold AND hop_count<MAX_HOPS
      │
      └──────► company_retrieval (loop back — max 3 times)

Usage:
    pipeline = EarningsEdgePipeline()
    result   = await pipeline.run(ticker="AAPL", query="Analyse Q3 2024 earnings quality")
"""

from __future__ import annotations

from typing import Any, Literal

from loguru import logger

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore
    from typing_extensions import TypedDict             # type: ignore
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.warning(
        "langgraph not installed — EarningsEdgePipeline will run in "
        "sequential fallback mode without graph execution"
    )
    # Define a stub TypedDict so the module is importable
    from typing import TypedDict  # type: ignore  # noqa: F811

from config.settings import settings
from src.rag.nodes import (
    company_retrieval_node,
    contradiction_check_node,
    gap_detector_node,
    industry_retrieval_node,
    macro_retrieval_node,
    quality_check_node,
    query_classifier_node,
    ragas_prep_node,
    synthesis_node,
)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class EarningsEdgeState(TypedDict, total=False):
    """
    Shared mutable state that flows through the LangGraph pipeline.

    All fields are optional (total=False) so nodes only need to return
    the keys they modify — not the full state.
    """

    # Input fields (set by caller before graph execution)
    ticker: str
    query: str
    quarter: str       # e.g. "Q3"
    year: int          # e.g. 2024

    # Query classification
    query_intent: str  # one of the _VALID_INTENTS from nodes.py

    # Retrieval outputs
    retrieved_chunks: list[dict[str, Any]]
    company_context: str
    industry_context: str
    macro_context: str
    context_used: str

    # Analysis inputs (pre-computed before graph run, injected into state)
    earnings_quality_score_obj: dict[str, Any]
    tone_drift_report: dict[str, Any]

    # Contradiction detection
    contradictions_found: list[dict[str, Any]]

    # Generation
    final_answer: str

    # Quality gate
    quality_score: float
    needs_more_retrieval: bool
    hop_count: int
    gap_analysis: dict[str, Any]

    # RAGAS evaluation
    ragas_input: dict[str, Any]


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def _route_after_gap_check(state: EarningsEdgeState) -> Literal[
    "industry_retrieval", "contradiction_check"
]:
    """
    Route after gap detection.

    If gaps were found and we haven't hit max hops → retrieve more.
    Otherwise → proceed to contradiction check.
    """
    if state.get("needs_more_retrieval") and state.get("hop_count", 0) < settings.MAX_RAG_HOPS:
        return "industry_retrieval"
    return "contradiction_check"


def _route_after_quality_check(state: EarningsEdgeState) -> Literal[
    "company_retrieval", "ragas_prep"
]:
    """
    Route after quality check.

    If answer quality is below threshold and hops remain → loop back
    to company_retrieval for a richer context window.
    Otherwise → finalise with RAGAS prep.
    """
    quality_score = state.get("quality_score", 1.0)
    hop_count = state.get("hop_count", 0)

    if (
        quality_score < settings.QUALITY_GATE_THRESHOLD
        and hop_count < settings.MAX_RAG_HOPS
    ):
        logger.info(
            f"Quality score {quality_score:.2f} < {settings.QUALITY_GATE_THRESHOLD} — "
            f"re-retrieving (hop {hop_count})"
        )
        return "company_retrieval"

    return "ragas_prep"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """
    Construct and compile the EarningsEdge LangGraph.

    Returns:
        Compiled LangGraph CompiledStateGraph ready for .ainvoke() calls.
        Returns None if langgraph is not installed.
    """
    if not _LANGGRAPH_AVAILABLE:
        logger.error("langgraph not available — cannot build graph")
        return None

    graph = StateGraph(EarningsEdgeState)

    # Register all nodes
    graph.add_node("query_classifier",    query_classifier_node)
    graph.add_node("company_retrieval",   company_retrieval_node)
    graph.add_node("gap_detector",        gap_detector_node)
    graph.add_node("industry_retrieval",  industry_retrieval_node)
    graph.add_node("macro_retrieval",     macro_retrieval_node)
    graph.add_node("contradiction_check", contradiction_check_node)
    graph.add_node("synthesis",           synthesis_node)
    graph.add_node("quality_check",       quality_check_node)
    graph.add_node("ragas_prep",          ragas_prep_node)

    # Linear edges
    graph.add_edge(START,                 "query_classifier")
    graph.add_edge("query_classifier",    "company_retrieval")
    graph.add_edge("company_retrieval",   "gap_detector")
    graph.add_edge("industry_retrieval",  "macro_retrieval")
    graph.add_edge("macro_retrieval",     "contradiction_check")
    graph.add_edge("contradiction_check", "synthesis")
    graph.add_edge("synthesis",           "quality_check")
    graph.add_edge("ragas_prep",          END)

    # Conditional edges
    graph.add_conditional_edges(
        "gap_detector",
        _route_after_gap_check,
        {
            "industry_retrieval":  "industry_retrieval",
            "contradiction_check": "contradiction_check",
        },
    )
    graph.add_conditional_edges(
        "quality_check",
        _route_after_quality_check,
        {
            "company_retrieval": "company_retrieval",
            "ragas_prep":        "ragas_prep",
        },
    )

    compiled = graph.compile()
    logger.info("EarningsEdge LangGraph compiled ✓")
    return compiled


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------


class EarningsEdgePipeline:
    """
    High-level wrapper around the compiled LangGraph.

    Provides a clean .run() interface and handles pre-loading of analysis
    results (quality score, drift report) that are computed before the
    graph runs rather than inside it (to avoid re-computing on each hop).
    """

    def __init__(self) -> None:
        self._graph = build_graph()
        self._fallback_mode = not _LANGGRAPH_AVAILABLE

    async def run(
        self,
        ticker: str,
        query: str,
        quarter: str = "Q4",
        year: int = 2024,
        quality_score_obj: dict[str, Any] | None = None,
        tone_drift_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full EarningsEdge RAG pipeline for a ticker/query.

        Pre-computed analysis results (quality score, drift report) are
        injected into the initial state so the synthesis node can use them
        without re-running the expensive analysis layer inside the graph.

        Args:
            ticker:            Stock ticker (upper case).
            query:             Natural language question.
            quarter:           Reporting quarter e.g. "Q3".
            year:              Reporting year.
            quality_score_obj: Pre-computed EarningsQualityScore as dict.
            tone_drift_report: Pre-computed ToneDriftReport as dict.

        Returns:
            Final EarningsEdgeState dict after graph completion.
        """
        initial_state: EarningsEdgeState = {
            "ticker": ticker.upper(),
            "query": query,
            "quarter": quarter,
            "year": year,
            "hop_count": 0,
            "needs_more_retrieval": False,
            "retrieved_chunks": [],
            "company_context": "",
            "industry_context": "",
            "macro_context": "",
            "context_used": "",
            "contradictions_found": [],
            "final_answer": "",
            "quality_score": 0.0,
            "gap_analysis": {},
            "ragas_input": {},
            "earnings_quality_score_obj": quality_score_obj or {},
            "tone_drift_report": tone_drift_report or {},
        }

        logger.info(
            f"Pipeline starting: ticker={ticker}  "
            f"query='{query[:60]}'  "
            f"period={quarter} {year}"
        )

        if self._fallback_mode or self._graph is None:
            return await self._sequential_fallback(initial_state)

        try:
            final_state = await self._graph.ainvoke(initial_state)
            logger.info(
                f"[{ticker}] Pipeline complete: "
                f"quality={final_state.get('quality_score', 0):.2f}  "
                f"hops={final_state.get('hop_count', 0)}"
            )
            return final_state
        except Exception as exc:
            logger.error(f"[{ticker}] Graph execution failed: {exc}")
            return await self._sequential_fallback(initial_state)

    async def _sequential_fallback(
        self, state: EarningsEdgeState
    ) -> EarningsEdgeState:
        """
        Run all nodes sequentially when LangGraph is unavailable.

        This fallback guarantees the pipeline always produces a result
        even without the langgraph dependency installed.

        Args:
            state: Initial state dict.

        Returns:
            Final state after all nodes have run.
        """
        ticker = state.get("ticker", "")
        logger.warning(f"[{ticker}] Running in sequential fallback mode")

        node_sequence = [
            query_classifier_node,
            company_retrieval_node,
            gap_detector_node,
            industry_retrieval_node,
            macro_retrieval_node,
            contradiction_check_node,
            synthesis_node,
            quality_check_node,
            ragas_prep_node,
        ]

        current_state = dict(state)
        for node_fn in node_sequence:
            try:
                updates = await node_fn(current_state)  # type: ignore[arg-type]
                current_state.update(updates)
            except Exception as exc:
                logger.error(f"Node {node_fn.__name__} failed: {exc}")

        return current_state  # type: ignore[return-value]

    def get_graph_png(self) -> bytes | None:
        """
        Render the graph as a PNG for README / dashboard display.

        Returns:
            PNG bytes or None if rendering is unavailable.
        """
        if self._graph is None:
            return None
        try:
            return self._graph.get_graph().draw_mermaid_png()
        except Exception as exc:
            logger.debug(f"Graph PNG rendering failed: {exc}")
            return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

pipeline = EarningsEdgePipeline()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    # Test state routing logic (no LLM or graph needed)
    print("=== Routing logic tests ===")

    # Route after gap check — needs more retrieval
    state_gaps: EarningsEdgeState = {
        "ticker": "AAPL",
        "query": "test",
        "needs_more_retrieval": True,
        "hop_count": 1,
    }
    route = _route_after_gap_check(state_gaps)
    assert route == "industry_retrieval", f"Expected industry_retrieval, got {route}"
    print(f"Gap check routing (needs_more=True): → {route} ✓")

    # Route after gap check — no gaps
    state_no_gaps: EarningsEdgeState = {
        "ticker": "AAPL",
        "query": "test",
        "needs_more_retrieval": False,
        "hop_count": 1,
    }
    route2 = _route_after_gap_check(state_no_gaps)
    assert route2 == "contradiction_check"
    print(f"Gap check routing (no gaps): → {route2} ✓")

    # Route after quality check — low score, hops remaining
    state_low_quality: EarningsEdgeState = {
        "ticker": "AAPL",
        "query": "test",
        "quality_score": 0.40,
        "hop_count": 1,
    }
    route3 = _route_after_quality_check(state_low_quality)
    assert route3 == "company_retrieval"
    print(f"Quality check routing (low score, hops left): → {route3} ✓")

    # Route after quality check — high score
    state_good: EarningsEdgeState = {
        "ticker": "AAPL",
        "query": "test",
        "quality_score": 0.90,
        "hop_count": 1,
    }
    route4 = _route_after_quality_check(state_good)
    assert route4 == "ragas_prep"
    print(f"Quality check routing (good score): → {route4} ✓")

    # Route after quality check — max hops reached, still low quality
    state_max_hops: EarningsEdgeState = {
        "ticker": "AAPL",
        "query": "test",
        "quality_score": 0.20,
        "hop_count": settings.MAX_RAG_HOPS,
    }
    route5 = _route_after_quality_check(state_max_hops)
    assert route5 == "ragas_prep", "Should finalise even with low quality at max hops"
    print(f"Quality check routing (max hops reached): → {route5} ✓")

    if _LANGGRAPH_AVAILABLE:
        print(f"\nLangGraph available ✓ — graph buildable")
        g = build_graph()
        assert g is not None
    else:
        print("\nLangGraph not installed — sequential fallback mode ✓")

    print("\ngraph smoke test passed ✓")
