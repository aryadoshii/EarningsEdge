"""EarningsEdge — Main landing page."""
from __future__ import annotations
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="EarningsEdge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.theme import inject_theme, sidebar_nav, score_display, component_bar
inject_theme()
sidebar_nav("main")

from config.settings import settings

# ---------------------------------------------------------------------------
# Helpers (mirrors 01_ticker_analysis.py)
# ---------------------------------------------------------------------------

def _ticker_dir(ticker: str) -> Path:
    return settings.processed_data_path / ticker.upper()

def _has_chunks(ticker: str) -> bool:
    return (_ticker_dir(ticker) / "chunks.json").exists()

def _has_analysis(ticker: str) -> bool:
    return (_ticker_dir(ticker) / "analysis.json").exists()

def _data_age_days(ticker: str) -> float | None:
    import time
    p = _ticker_dir(ticker) / "chunks.json"
    if not p.exists():
        return None
    return (time.time() - p.stat().st_mtime) / 86_400

def _run_full_pipeline(ticker: str) -> int:
    """
    Run ingest → embed → analyze. Returns number of chunks produced.
    Displays live progress via st.status. Returns 0 if no filings found.
    """
    from src.pipeline_runner import (
        run_ingest_pipeline,
        run_embed_pipeline,
        run_analysis_pipeline,
    )

    with st.status(f"Setting up {ticker} — fetching SEC filings…", expanded=True) as status:
        st.write("**Step 1 / 3** — Fetching SEC filings from EDGAR…")
        try:
            ingest_result = asyncio.run(run_ingest_pipeline(ticker))
        except Exception as exc:
            status.update(label="Ingestion failed", state="error")
            st.error(f"Ingestion error: {exc}")
            return 0

        n_filings = len(ingest_result.get("filings", []))
        n_chunks  = len(ingest_result.get("chunks", []))

        if n_chunks == 0:
            status.update(label="No filings found", state="error")
            return 0

        st.write(f"✓ {n_filings} filings → {n_chunks} chunks")

        st.write("**Step 2 / 3** — Embedding chunks into ChromaDB…")
        try:
            asyncio.run(run_embed_pipeline(ticker))
            st.write(f"✓ Embedded {n_chunks} chunks")
        except Exception as exc:
            status.update(label="Embedding failed", state="error")
            st.error(f"Embedding error: {exc}")
            return 0

        st.write("**Step 3 / 3** — Running analysis (FinBERT · NLI · scorer)…")
        try:
            asyncio.run(run_analysis_pipeline(ticker))
            st.write("✓ Analysis complete")
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error(f"Analysis error: {exc}")
            return 0

        status.update(label=f"{ticker} ready", state="complete")
        return n_chunks

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ee-fade-in" style="padding:2.25rem 0 2.4rem;text-align:center;">'
    '<div style="max-width:820px;margin:0 auto;">'
    '<h1 style="margin:0;font-size:3rem;font-weight:800;letter-spacing:-0.05em;line-height:1;">'
    '<span style="color:var(--taupe);">Earnings</span>Edge</h1>'
    '<p style="margin:0.65rem 0 0;font-size:0.92rem;font-weight:600;color:var(--text-muted);">'
    'Institutional-grade earnings intelligence · SEC filings · NLP signals · Composite scoring'
    '</p></div></div>',
    unsafe_allow_html=True,
)

# ── Quick analysis form ───────────────────────────────────────────────────────
st.markdown('<div class="ee-label" style="margin-bottom:0.65rem;">Quick Analysis</div>', unsafe_allow_html=True)
with st.form("qa"):
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        ticker_input = st.text_input("Ticker Symbol", value="AAPL", placeholder="AAPL, MSFT…").upper().strip()
    with c2:
        quarter_input = st.selectbox("Quarter", ["Q1","Q2","Q3","Q4"])
    with c3:
        year_input = st.number_input("Year", min_value=2019, max_value=2026, value=2025)
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("▶ Analyse", use_container_width=True)

if submitted and ticker_input:
    # ── Auto-ingest if needed ──────────────────────────────────────────────
    needs_ingest = not _has_chunks(ticker_input)
    needs_analysis = not _has_analysis(ticker_input)

    if needs_ingest:
        n = _run_full_pipeline(ticker_input)
        if n == 0:
            st.markdown(
                '<div class="ee-card" style="border-color:rgba(184,132,118,0.28);'
                'background:rgba(217,187,176,0.18);margin-top:1rem;">'
                '<div style="font-family:\'Space Mono\',monospace;font-size:0.68rem;color:var(--red);'
                'font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">'
                '⚠ No SEC Filings Found</div>'
                f'<div style="font-family:\'Sora\',sans-serif;font-size:0.82rem;color:var(--text);'
                f'line-height:1.6;margin-bottom:0.5rem;">'
                f'<strong>{ticker_input}</strong> returned no filings from SEC EDGAR. '
                f'This usually means:</div>'
                f'<ul style="font-family:\'Sora\',sans-serif;font-size:0.8rem;color:var(--text-muted);'
                f'line-height:1.8;margin:0 0 0 1rem;">'
                f'<li>The ticker is not listed on a US exchange (e.g. LSE, TSX, Euronext)</li>'
                f'<li>The ticker symbol is incorrect — try the US-listed equivalent</li>'
                f'<li>The company does not file with the SEC (foreign private issuers may use Form 20-F)</li>'
                f'</ul>'
                f'<div style="margin-top:0.75rem;padding:0.6rem 0.8rem;'
                f'background:rgba(232,213,158,0.18);border-radius:8px;'
                f'font-family:\'Space Mono\',monospace;font-size:0.65rem;color:var(--text-muted);">'
                f'Examples of US-listed tickers: AAPL · MSFT · NVDA · TSLA · JPM · META · AMZN'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.stop()
    elif needs_analysis:
        from src.pipeline_runner import run_embed_pipeline, run_analysis_pipeline
        with st.status(f"Running analysis for {ticker_input}…", expanded=True) as status:
            asyncio.run(run_embed_pipeline(ticker_input))
            asyncio.run(run_analysis_pipeline(ticker_input))
            status.update(label="Analysis complete", state="complete")

    # ── RAG query ─────────────────────────────────────────────────────────
    with st.spinner(f"Querying {ticker_input} {quarter_input} {year_input}…"):
        try:
            from src.rag.multi_hop_chain import multi_hop_chain
            from src.ingestion.data_validator import Quarter as Q
            q_map = {"Q1":Q.Q1,"Q2":Q.Q2,"Q3":Q.Q3,"Q4":Q.Q4}
            result = asyncio.run(
                multi_hop_chain.analyse_with_full_pipeline(
                    ticker=ticker_input, quarter=q_map[quarter_input], year=int(year_input),
                )
            )
            st.session_state["last_result"] = result
            st.session_state["last_ticker"] = ticker_input
            qs    = result.quality_score_obj
            score = result.composite_score
            signal = result.signal

            col_score, col_comps = st.columns([1, 1])
            with col_score:
                st.markdown(
                    '<div class="ee-card" style="text-align:center;padding:2rem 1rem;">'
                    + score_display(score, signal) + '</div>',
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
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)

            if result.final_answer:
                with st.expander("📄 Full Analysis Report", expanded=True):
                    st.markdown(result.final_answer)

        except Exception as e:
            st.error(f"Analysis error: {e}")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── Architecture cards ────────────────────────────────────────────────────────
st.markdown('<div class="ee-label" style="margin-bottom:0.65rem;">Architecture</div>', unsafe_allow_html=True)
arch_cols = st.columns(3)
arch_items = [
    ("01 — Ingest", "var(--gold)",
     "SEC EDGAR 10-K/10-Q/8-K", "Earnings call transcripts",
     "Analyst consensus · yfinance", "XBRL financial data"),
    ("02 — Analyse", "var(--taupe)",
     "FinBERT sentiment scoring", "NLI contradiction detection",
     "NER guidance extraction", "Accruals quality ratio"),
    ("03 — Signal", "var(--green)",
     "Composite quality score [-1,+1]", "LangGraph multi-hop RAG",
     "Groq / Gemini synthesis", "RAGAS-evaluated answers"),
]
for col, (title, color, *items) in zip(arch_cols, arch_items):
    with col:
        item_html = "".join(f'<div style="margin-bottom:0.25rem;">{it}</div>' for it in items)
        st.markdown(
            '<div class="ee-card" style="height:100%;">'
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;color:{color};'
            f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.6rem;">{title}</div>'
            f'<div style="font-family:\'Sora\',sans-serif;font-size:0.8rem;color:var(--text);line-height:1.75;">'
            f'{item_html}</div></div>',
            unsafe_allow_html=True,
        )
