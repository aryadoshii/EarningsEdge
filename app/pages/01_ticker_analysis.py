"""Ticker Analysis — EarningsEdge."""
from __future__ import annotations
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
st.set_page_config(page_title="Ticker Analysis · EarningsEdge", layout="wide", initial_sidebar_state="expanded")

from app.components.theme import inject_theme, sidebar_nav, score_display, component_bar
inject_theme()
sidebar_nav("ticker")

from config.settings import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ticker_data_path(ticker: str) -> Path:
    return settings.processed_data_path / ticker.upper()

def _has_chunks(ticker: str) -> bool:
    return (_ticker_data_path(ticker) / "chunks.json").exists()

def _has_analysis(ticker: str) -> bool:
    return (_ticker_data_path(ticker) / "analysis.json").exists()

def _data_age_days(ticker: str) -> float | None:
    """Return age of chunks.json in days, or None if missing."""
    import time
    p = _ticker_data_path(ticker) / "chunks.json"
    if not p.exists():
        return None
    return (time.time() - p.stat().st_mtime) / 86_400

def _run_full_pipeline(ticker: str) -> None:
    """
    Run ingest → embed → analyze for *ticker* inside a live st.status block.
    Writes progress to the UI as each stage completes.
    """
    from src.pipeline_runner import (
        run_ingest_pipeline,
        run_embed_pipeline,
        run_analysis_pipeline,
    )

    with st.status(f"Setting up {ticker} — this takes 2-4 minutes on first run…", expanded=True) as status:
        st.write("**Step 1 / 3** — Fetching SEC filings from EDGAR…")
        try:
            ingest_result = asyncio.run(run_ingest_pipeline(ticker))
            n_filings = len(ingest_result.get("filings", []))
            n_chunks  = len(ingest_result.get("chunks", []))
            st.write(f"✓ Fetched {n_filings} filings → {n_chunks} chunks")
        except Exception as exc:
            status.update(label="Ingestion failed", state="error")
            st.error(f"Ingestion error: {exc}")
            st.stop()

        st.write("**Step 2 / 3** — Embedding chunks into ChromaDB (BGE-large)…")
        try:
            n_embedded = asyncio.run(run_embed_pipeline(ticker))
            st.write(f"✓ Embedded {n_embedded} chunks")
        except Exception as exc:
            status.update(label="Embedding failed", state="error")
            st.error(f"Embedding error: {exc}")
            st.stop()

        st.write("**Step 3 / 3** — Running analysis (FinBERT · NLI · scorer)…")
        try:
            asyncio.run(run_analysis_pipeline(ticker))
            st.write("✓ Analysis complete")
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error(f"Analysis error: {exc}")
            st.stop()

        status.update(label=f"{ticker} ready — running RAG query…", state="complete")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="padding:0.5rem 0 0.75rem;border-bottom:1px solid var(--border);'
        'margin-bottom:0.9rem;"><div class="ee-label">Analysis Controls</div></div>',
        unsafe_allow_html=True,
    )
    ticker  = st.text_input("Ticker", value="AAPL").upper().strip()
    quarter = st.selectbox("Quarter", ["Q1","Q2","Q3","Q4"], index=0)
    year    = st.number_input("Year", min_value=2019, max_value=2026, value=2025)
    query   = st.text_area("Custom Query", placeholder="Leave blank for full analysis", height=72)

    # Data-freshness indicator
    age = _data_age_days(ticker)
    if age is None:
        st.markdown(
            '<div style="margin:0.4rem 0 0.6rem;padding:0.5rem 0.7rem;'
            'background:rgba(220,80,60,0.08);border:1px solid rgba(220,80,60,0.25);'
            'border-radius:8px;font-family:\'Space Mono\',monospace;font-size:0.62rem;'
            'color:var(--red);">◈ No local data — pipeline will run automatically</div>',
            unsafe_allow_html=True,
        )
    elif age > 7:
        st.markdown(
            f'<div style="margin:0.4rem 0 0.6rem;padding:0.5rem 0.7rem;'
            f'background:rgba(200,140,0,0.08);border:1px solid rgba(200,140,0,0.25);'
            f'border-radius:8px;font-family:\'Space Mono\',monospace;font-size:0.62rem;'
            f'color:#b8860b;">⚠ Data is {age:.0f}d old — consider refreshing</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="margin:0.4rem 0 0.6rem;padding:0.5rem 0.7rem;'
            f'background:rgba(60,160,80,0.08);border:1px solid rgba(60,160,80,0.25);'
            f'border-radius:8px;font-family:\'Space Mono\',monospace;font-size:0.62rem;'
            f'color:var(--green);">✓ Data cached ({age:.1f}d ago)</div>',
            unsafe_allow_html=True,
        )

    col_run, col_refresh = st.columns([2, 1])
    with col_run:
        run_btn = st.button("▶ Run Analysis", use_container_width=True, type="primary")
    with col_refresh:
        refresh_btn = st.button("↺", use_container_width=True, help="Force re-ingest (even if data exists)")

    st.markdown(
        '<div style="margin-top:0.75rem;padding:0.65rem 0.75rem;background:rgba(255,250,244,0.5);'
        'border:1px solid var(--border);border-radius:12px;">'
        '<div class="ee-label" style="margin-bottom:0.25rem;">Model</div>'
        f'<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;color:var(--text-muted);">{settings.GROQ_MODEL}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="ee-page-header ee-fade-in" style="padding-bottom:1.25rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;">'
    '<h1 style="margin:0;font-size:1.75rem;letter-spacing:-0.03em;font-family:\'Sora\',sans-serif;font-weight:600;">Ticker Analysis</h1>'
    '<p style="margin:0.2rem 0 0;font-size:0.8rem;color:var(--text-muted);">Earnings quality scoring · Management credibility · Tone drift · NLI contradictions</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pipeline trigger logic
# ---------------------------------------------------------------------------

if (run_btn or refresh_btn) and ticker:
    force_refresh = bool(refresh_btn)
    needs_ingest  = force_refresh or not _has_chunks(ticker)
    needs_analysis = force_refresh or not _has_analysis(ticker)

    # Run pipeline stages as needed
    if needs_ingest:
        _run_full_pipeline(ticker)
    elif needs_analysis:
        # Chunks exist but analysis is missing — run analysis only
        from src.pipeline_runner import run_embed_pipeline, run_analysis_pipeline
        with st.status(f"Running analysis for {ticker}…", expanded=True) as status:
            st.write("Embedding…")
            asyncio.run(run_embed_pipeline(ticker))
            st.write("Analysing…")
            asyncio.run(run_analysis_pipeline(ticker))
            status.update(label="Analysis complete", state="complete")

    # Now run the RAG query
    with st.spinner(f"Querying {ticker} {quarter} {year}…"):
        try:
            from src.rag.multi_hop_chain import multi_hop_chain
            from src.ingestion.data_validator import Quarter as Q
            result = asyncio.run(
                multi_hop_chain.analyse_with_full_pipeline(
                    ticker=ticker, query=query or None,
                    quarter={"Q1":Q.Q1,"Q2":Q.Q2,"Q3":Q.Q3,"Q4":Q.Q4}[quarter],
                    year=int(year),
                )
            )
            st.session_state["analysis_result"] = result
            st.session_state["analysis_ticker"]  = ticker
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

result = st.session_state.get("analysis_result")

# Clear cached result if ticker changed
if result and st.session_state.get("analysis_ticker") != ticker:
    result = None

if not result:
    st.markdown(
        '<div class="ee-card" style="text-align:center;padding:3.5rem 2rem;border-style:dashed;">'
        '<div style="font-size:2rem;opacity:0.2;margin-bottom:0.75rem;">◈</div>'
        '<div style="font-family:\'Sora\',sans-serif;font-size:0.875rem;color:var(--text-muted);">'
        'Select a ticker and click <strong style="color:var(--taupe);">Run Analysis</strong></div>'
        '<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;color:var(--text-dim);margin-top:0.4rem;">'
        'Data is fetched automatically — no terminal commands needed</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

qs     = result.quality_score_obj
score  = result.composite_score
signal = result.signal
dr     = result.tone_drift_report or {}
alert  = dr.get("alert_level", "GREEN")
contras = result.contradictions or []

# ── Hero: score + components ──────────────────────────────────────────────────
col_hero, col_comps = st.columns([1, 1])
with col_hero:
    st.markdown(
        '<div class="ee-card" style="text-align:center;padding:2rem 1.25rem;">'
        + '<div class="ee-label" style="text-align:center;margin-bottom:0.5rem;">'
        + f'<span class="ee-ticker">{result.ticker}</span>&nbsp;&nbsp;<span class="ee-alert {alert}">● {alert}</span>'
        + '</div>'
        + score_display(score, signal)
        + '<div style="display:flex;justify-content:center;gap:1.25rem;margin-top:0.75rem;'
        + 'font-family:\'Space Mono\',monospace;font-size:0.65rem;color:var(--text-muted);">'
        + f'<span>Chunks <strong style="color:var(--text);">{result.retrieved_chunk_count}</strong></span>'
        + f'<span>Hops <strong style="color:var(--text);">{result.hops_used}</strong></span>'
        + f'<span>Ground <strong style="color:var(--text);">{result.grounding_score:.0%}</strong></span>'
        + f'<span>{result.total_latency_ms/1000:.1f}s</span>'
        + '</div></div>',
        unsafe_allow_html=True,
    )

with col_comps:
    html = '<div class="ee-card"><div class="ee-label" style="margin-bottom:0.6rem;">Component Breakdown</div>'
    for label, (val, wt) in {
        "Sentiment Drift":   (qs.get("sentiment_drift_component",0),   settings.SCORE_WEIGHT_SENTIMENT_DRIFT),
        "Guidance Accuracy": (qs.get("guidance_accuracy_component",0), settings.SCORE_WEIGHT_GUIDANCE_ACCURACY),
        "Accruals Quality":  (qs.get("accruals_component",0),          settings.SCORE_WEIGHT_ACCRUALS),
        "Analyst Revisions": (qs.get("analyst_revision_component",0),  settings.SCORE_WEIGHT_ANALYST_REVISION),
    }.items():
        html += component_bar(label, val, wt)
    ar = qs.get("accruals_ratio")
    if ar is not None:
        quality = "✓ Cash-backed" if ar < 0 else "⚠ Accrual-heavy"
        html += (
            '<div style="margin-top:0.75rem;padding-top:0.6rem;border-top:1px solid var(--border);'
            'font-family:\'Space Mono\',monospace;font-size:0.65rem;color:var(--text-muted);">'
            f'Accruals ratio: <span style="color:var(--text);">{ar:.4f}</span> &nbsp;·&nbsp; {quality}</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Contradictions ────────────────────────────────────────────────────────────
if contras:
    st.markdown(
        '<div class="ee-label" style="margin-bottom:0.6rem;">Contradictions Detected '
        f'<span class="ee-pill short" style="margin-left:0.4rem;">{len(contras)}</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    for i, c in enumerate(contras[:4]):
        col = c1 if i % 2 == 0 else c2
        with col:
            sc   = c.get("contradiction_score", 0)
            src_a = c.get("chunk_a_source","")[-42:]
            src_b = c.get("chunk_b_source","")[-42:]
            interp = c.get("interpretation","")[:115]
            st.markdown(
                '<div class="ee-contra">'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.68rem;color:var(--red);font-weight:700;">⚠ NLI CONFLICT · {sc:.3f}</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.62rem;color:var(--text-muted);margin:0.2rem 0;">{src_a}<br>↔ {src_b}</div>'
                f'<div style="font-family:\'Sora\',sans-serif;font-size:0.78rem;color:var(--text);line-height:1.5;">{interp}…</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

# ── LLM report ────────────────────────────────────────────────────────────────
if result.final_answer:
    st.markdown('<div class="ee-label" style="margin-bottom:0.6rem;">Intelligence Report</div>', unsafe_allow_html=True)
    with st.expander("View Full Analysis", expanded=True):
        st.markdown(result.final_answer)
else:
    st.markdown(
        '<div class="ee-card" style="text-align:center;padding:1.75rem;border-style:dashed;">'
        '<div style="font-family:\'Space Mono\',monospace;font-size:0.72rem;color:var(--text-muted);">'
        'No LLM analysis generated · Check Groq API key in .env</div></div>',
        unsafe_allow_html=True,
    )
