"""
Prompt templates for EarningsEdge RAG pipeline.

All prompts are defined here as module-level constants — never inline in
node code.  This makes them easy to version, A/B test via MLflow, and swap
without touching orchestration logic.

Prompts are plain strings with {placeholder} slots filled via str.format()
or f-strings at call time.  No LangChain PromptTemplate dependency — keeps
this module import-free and independently testable.

Sections:
    SYSTEM_PROMPTS   — Role-setting prompts passed as the system message
    SYNTHESIS_PROMPTS — Main analysis generation
    EXTRACTION_PROMPTS — Structured data extraction (guidance, entities)
    CLASSIFICATION_PROMPTS — Query intent classification
    EVALUATION_PROMPTS — Self-grounding quality check
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_EARNINGS_ANALYST = """\
You are EarningsEdge, an institutional-grade financial analysis AI built for
quantitative analysts and portfolio managers.

Your mandate:
- Analyse earnings quality, management credibility, and financial statement integrity
- Detect tone drift, guidance accuracy patterns, and contradictions in management language
- Surface actionable signals grounded strictly in the provided source documents

Hard rules you NEVER break:
1. Only make claims that are directly supported by the retrieved context below
2. Every factual claim MUST include a source citation: [Source: {filing_type} {quarter} {year}]
3. When you detect a contradiction between management statements and reported numbers, flag it
   explicitly with a ⚠️ CONFLICT badge
4. Never hallucinate financial figures — if a number isn't in the context, say so
5. Distinguish between what management SAID (transcript) vs what was FILED (10-Q/10-K)
6. Your output must be structured exactly as specified — no deviation
"""

SYSTEM_GUIDANCE_EXTRACTOR = """\
You are a financial data extraction engine. Your only job is to extract
forward-looking numerical guidance from financial text and output valid JSON.
You do not summarise, explain, or add commentary. Output ONLY a JSON array.
"""

SYSTEM_QUERY_CLASSIFIER = """\
You are a query intent classifier for a financial RAG system.
Classify the user query into exactly one of these categories and output
only the category name — no explanation, no punctuation.
"""

SYSTEM_QUALITY_CHECKER = """\
You are a grounding verifier for a financial AI system.
Your job is to check whether an answer is supported by the provided context.
Output only a JSON object with keys: score (0.0-1.0), issues (list of strings).
"""

# ---------------------------------------------------------------------------
# Synthesis prompt — the main analytical output
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """\
You are analysing {ticker} for the period ending {period}.

RETRIEVED CONTEXT:
{context}

EARNINGS QUALITY SCORE: {quality_score} ({signal})
Component breakdown:
  - Sentiment Drift:    {sentiment_drift_component:+.3f}
  - Guidance Accuracy:  {guidance_accuracy_component:+.3f}
  - Accruals Quality:   {accruals_component:+.3f}
  - Analyst Revisions:  {analyst_revision_component:+.3f}

TONE DRIFT ALERT: {alert_level}
{alert_reason}

CONTRADICTIONS DETECTED: {contradiction_count}
{contradiction_summary}

Generate a structured analysis in EXACTLY this format:

## Executive Summary
[3 sentences maximum. State the earnings quality verdict, the primary signal driver, and the key risk. Be specific — cite numbers.]

## Key Signals
[Bulleted list. Each bullet must end with [Source: <filing_type> <quarter> <year> — <section>]. Minimum 4 bullets, maximum 8.]

## Risk Flags
[List contradictions, tone drift alerts, and accruals anomalies. Use ⚠️ CONFLICT for NLI-detected contradictions. Use 🔴 DRIFT for RED tone alerts. If none detected, write "None identified in available context."]

## Earnings Quality Assessment
Score: {quality_score} → {signal}
[2-3 sentences interpreting what this score means for this specific company, referencing the component breakdown above.]

## Analyst Recommendation Context
[What do the analyst revisions and price targets suggest? How does the EarningsEdge signal align or diverge from street consensus? 2-3 sentences.]
"""

# ---------------------------------------------------------------------------
# Guidance extraction prompt
# ---------------------------------------------------------------------------

GUIDANCE_EXTRACTION_PROMPT = """\
Extract all forward-looking numerical guidance from the following financial text.

For each piece of guidance found, output a JSON object with these fields:
  - metric: one of ["EPS", "revenue", "capex", "margin", "growth", "other"]
  - value_low: lower bound of range (number, null if not a range)
  - value_high: upper bound of range (number, null if not a range)
  - unit: "USD" | "%" | "USD/share" | "bps" | "shares" | "other"
  - fiscal_period_referenced: string like "Q4 2024" or "FY2025" (empty string if not stated)
  - confidence_level: "explicit" (direct number stated) or "implicit" (inferred from context)

Rules:
- Convert all dollar values to full numbers (e.g. "$94.9 billion" → 94900000000)
- If guidance is given as a single value (not a range), set value_low to that value and value_high to null
- If no guidance is found anywhere in the text, return an empty array: []
- Output ONLY valid JSON — no markdown, no explanation, no preamble

TEXT:
{text}
"""

# ---------------------------------------------------------------------------
# Query classification prompt
# ---------------------------------------------------------------------------

QUERY_CLASSIFICATION_PROMPT = """\
Classify this financial analysis query into exactly one category.

Categories:
  earnings_analysis      — questions about earnings quality, beats/misses, EPS, revenue
  risk_assessment        — questions about risks, headwinds, competitive threats
  guidance_tracking      — questions about forward guidance, outlook, projections
  comparative_analysis   — questions comparing this company to peers or prior periods
  tone_drift             — questions about management language changes, credibility
  macro_context          — questions about sector trends, Fed policy, macro conditions
  general                — anything that doesn't fit above

Query: {query}

Output only the category name:"""

# ---------------------------------------------------------------------------
# Gap detection prompt
# ---------------------------------------------------------------------------

GAP_DETECTION_PROMPT = """\
You have retrieved context to answer this query about {ticker}:

QUERY: {query}

RETRIEVED CONTEXT SUMMARY:
{context_summary}

Identify what information is MISSING from the retrieved context that would
be needed to answer the query well. Consider:
  - Specific time periods not covered
  - Peer/competitor data not present
  - Macro context not included
  - Specific financial metrics not found

Output ONLY a JSON object with:
  {{
    "has_gaps": true/false,
    "missing_periods": ["Q1 2023", ...],
    "needs_peer_data": true/false,
    "needs_macro_data": true/false,
    "missing_metrics": ["revenue", "EPS", ...],
    "gap_summary": "one sentence describing main gap"
  }}
"""

# ---------------------------------------------------------------------------
# Industry / peer context prompt
# ---------------------------------------------------------------------------

PEER_CONTEXT_PROMPT = """\
You are supplementing analysis of {ticker} with industry peer context.

PRIMARY COMPANY CONTEXT:
{company_context}

PEER COMPANY CONTEXT ({peer_tickers}):
{peer_context}

Based on this combined context, how does {ticker}'s earnings quality and
management tone compare to its peers? Focus on:
  1. Relative sentiment drift direction
  2. Guidance accuracy vs peers
  3. Accruals quality relative to sector
  
Be specific — cite peer company names and quarters. Keep response under 200 words.
"""

# ---------------------------------------------------------------------------
# Macro context injection prompt
# ---------------------------------------------------------------------------

MACRO_CONTEXT_PROMPT = """\
Relevant macro context for interpreting {ticker}'s earnings:

MACRO / SECTOR CONTEXT:
{macro_context}

How does this macro context affect the interpretation of {ticker}'s
earnings quality signals? Specifically:
  - Does the macro environment explain any tone deterioration?
  - Are sector headwinds masking company-specific issues?
  - Does the earnings quality score need adjusting for macro tailwinds/headwinds?

Keep response under 150 words. Be specific.
"""

# ---------------------------------------------------------------------------
# Quality check prompt
# ---------------------------------------------------------------------------

QUALITY_CHECK_PROMPT = """\
Check whether the following financial analysis answer is grounded in the
provided context. 

CONTEXT (retrieved chunks):
{context}

ANSWER TO CHECK:
{answer}

For each factual claim in the answer, verify it can be traced to the context.
Flag any claims that:
  - Cite specific numbers not present in context
  - Make assertions about periods not covered in context
  - Reference companies not mentioned in context
  - Draw conclusions that go beyond what the context supports

Output ONLY this JSON (no markdown, no preamble):
{{
  "score": <float 0.0-1.0, where 1.0 = fully grounded>,
  "issues": [<list of specific ungrounded claims, empty if none>],
  "verdict": "grounded" | "partially_grounded" | "hallucinated"
}}
"""

# ---------------------------------------------------------------------------
# RAGAS preparation prompt
# ---------------------------------------------------------------------------

RAGAS_CONTEXT_PREP_PROMPT = """\
Summarise what question this answer is trying to address, in one sentence,
suitable as a ground-truth reference for RAG evaluation.

ORIGINAL QUERY: {query}
ANSWER: {answer}

Output only the one-sentence ground truth summary:"""

# ---------------------------------------------------------------------------
# Prompt builder helpers
# ---------------------------------------------------------------------------


def build_synthesis_prompt(
    ticker: str,
    period: str,
    context: str,
    quality_score: float,
    signal: str,
    sentiment_drift_component: float,
    guidance_accuracy_component: float,
    accruals_component: float,
    analyst_revision_component: float,
    alert_level: str,
    alert_reason: str,
    contradiction_count: int,
    contradiction_summary: str,
) -> str:
    """
    Fill the SYNTHESIS_PROMPT template with computed values.

    Args:
        ticker:                     Stock ticker.
        period:                     Reporting period label e.g. "Q3 2024".
        context:                    Concatenated retrieved chunk texts.
        quality_score:              Composite score float.
        signal:                     "LONG" / "SHORT" / "NEUTRAL".
        sentiment_drift_component:  Drift component value.
        guidance_accuracy_component: Guidance accuracy value.
        accruals_component:         Accruals value.
        analyst_revision_component: Analyst revision value.
        alert_level:                "GREEN" / "YELLOW" / "RED".
        alert_reason:               Human-readable alert description.
        contradiction_count:        Number of NLI-detected contradictions.
        contradiction_summary:      Formatted contradiction descriptions.

    Returns:
        Filled prompt string.
    """
    return SYNTHESIS_PROMPT.format(
        ticker=ticker,
        period=period,
        context=context,
        quality_score=f"{quality_score:+.3f}",
        signal=signal,
        sentiment_drift_component=sentiment_drift_component,
        guidance_accuracy_component=guidance_accuracy_component,
        accruals_component=accruals_component,
        analyst_revision_component=analyst_revision_component,
        alert_level=alert_level,
        alert_reason=alert_reason,
        contradiction_count=contradiction_count,
        contradiction_summary=contradiction_summary or "None detected.",
    )


def build_gap_detection_prompt(
    ticker: str,
    query: str,
    context_summary: str,
) -> str:
    """Fill the GAP_DETECTION_PROMPT template."""
    return GAP_DETECTION_PROMPT.format(
        ticker=ticker,
        query=query,
        context_summary=context_summary,
    )


def build_quality_check_prompt(context: str, answer: str) -> str:
    """Fill the QUALITY_CHECK_PROMPT template."""
    return QUALITY_CHECK_PROMPT.format(context=context[:4000], answer=answer)


def build_guidance_extraction_prompt(text: str) -> str:
    """Fill the GUIDANCE_EXTRACTION_PROMPT template."""
    return GUIDANCE_EXTRACTION_PROMPT.format(text=text[:3000])


def build_classification_prompt(query: str) -> str:
    """Fill the QUERY_CLASSIFICATION_PROMPT template."""
    return QUERY_CLASSIFICATION_PROMPT.format(query=query)


def format_contradictions(contradictions: list[dict]) -> str:
    """
    Format a list of contradiction dicts into a readable summary string.

    Args:
        contradictions: List of Contradiction-like dicts with keys
                        chunk_a_source, chunk_b_source, contradiction_score,
                        interpretation.

    Returns:
        Formatted multi-line string.
    """
    if not contradictions:
        return "None detected."

    lines: list[str] = []
    for i, c in enumerate(contradictions[:5], 1):  # cap at 5 for prompt length
        lines.append(
            f"  {i}. ⚠️ CONFLICT (score={c.get('contradiction_score', 0):.2f})\n"
            f"     Source A: {c.get('chunk_a_source', 'unknown')}\n"
            f"     Source B: {c.get('chunk_b_source', 'unknown')}\n"
            f"     {c.get('interpretation', '')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify all placeholders fill correctly
    prompt = build_synthesis_prompt(
        ticker="AAPL",
        period="Q3 2024",
        context="[Sample context text from 10-Q MD&A...]",
        quality_score=0.45,
        signal="LONG",
        sentiment_drift_component=0.60,
        guidance_accuracy_component=0.50,
        accruals_component=0.46,
        analyst_revision_component=0.65,
        alert_level="GREEN",
        alert_reason="Tone stable or improving",
        contradiction_count=0,
        contradiction_summary="",
    )
    assert "AAPL" in prompt
    assert "LONG" in prompt
    assert "+0.600" in prompt
    print(f"Synthesis prompt: {len(prompt)} chars ✓")

    gap_prompt = build_gap_detection_prompt(
        ticker="AAPL",
        query="How has iPhone revenue trended over the last 4 quarters?",
        context_summary="Retrieved Q3 2024 10-Q MD&A and earnings transcript.",
    )
    assert "has_gaps" in gap_prompt
    print(f"Gap detection prompt: {len(gap_prompt)} chars ✓")

    cls_prompt = build_classification_prompt(
        "What does management say about AI revenue growth next year?"
    )
    assert "guidance_tracking" in cls_prompt
    print(f"Classification prompt: {len(cls_prompt)} chars ✓")

    guidance_prompt = build_guidance_extraction_prompt(
        "We expect revenue of $90B to $93B and EPS of $1.55 to $1.65 for Q4 2024."
    )
    assert "JSON" in guidance_prompt
    print(f"Guidance extraction prompt: {len(guidance_prompt)} chars ✓")

    contradictions = [
        {
            "contradiction_score": 0.87,
            "chunk_a_source": "AAPL transcript Q2 2024 — prepared_remarks",
            "chunk_b_source": "AAPL 10-Q Q2 2024 — mda",
            "interpretation": "Management stated demand was 'robust' on the call but the 10-Q cites 'challenging demand environment'.",
        }
    ]
    formatted = format_contradictions(contradictions)
    assert "⚠️ CONFLICT" in formatted
    print(f"Contradiction format: {len(formatted)} chars ✓")

    print("\nprompts smoke test passed ✓")
