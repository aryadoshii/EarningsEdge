"""EarningsEdge — Home / Overview page."""
from __future__ import annotations

import streamlit as st

from config.settings import settings

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

# ── Thesis ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ee-card ee-fade-in" style="text-align:center;padding:1.5rem 2rem;margin-bottom:1.75rem;">'
    '<div style="font-family:\'Sora\',sans-serif;font-size:0.9rem;color:var(--text);line-height:1.7;max-width:760px;margin:0 auto;">'
    'Management tone tends to deteriorate before guidance is cut. EarningsEdge reads '
    '10-K/10-Q filings the way a quant research desk would — extracting sentiment, '
    'hedging, contradictions, and guidance accuracy from the language itself — and '
    'combines that with the classic accruals-quality anomaly into one composite signal.'
    '</div></div>',
    unsafe_allow_html=True,
)

# ── Architecture / pipeline ───────────────────────────────────────────────────
st.markdown('<div class="ee-label" style="margin-bottom:0.65rem;">How It Works</div>', unsafe_allow_html=True)
arch_cols = st.columns(3)
for col, (title, color, *items) in zip(arch_cols, [
    (
        "01 — Ingest", "var(--gold)",
        "SEC EDGAR 10-K / 10-Q / 8-K filings — full text + XBRL financials",
        "Earnings call transcripts — prepared remarks &amp; Q&amp;A",
        "Analyst consensus &amp; revisions via yfinance",
        "Section-aware chunking → BGE-large-en-v1.5 embeddings → ChromaDB",
    ),
    (
        "02 — Analyse", "var(--taupe)",
        "FinBERT sentiment scoring, quarter over quarter",
        "DeBERTa-v3 NLI — flags contradictions across quarters and vs. the same quarter's filing",
        "spaCy + regex NER — extracts EPS / revenue / capex / margin guidance",
        "XBRL accruals ratio — (Net Income − Operating Cash Flow) / Total Assets",
    ),
    (
        "03 — Signal", "var(--green)",
        "Composite earnings quality score, weighted &amp; normalised to [-1, +1]",
        "9-node LangGraph multi-hop RAG — retrieval, gap detection, contradiction check",
        "Groq llama-3.3-70b (primary) → Gemini 1.5 Flash (fallback) synthesis",
        "Every claim cited back to filing type, quarter, and section",
    ),
]):
    with col:
        item_html = "".join(f'<div style="margin-bottom:0.4rem;">{it}</div>' for it in items)
        st.markdown(
            '<div class="ee-card" style="height:100%;">'
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;color:{color};'
            f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.6rem;">{title}</div>'
            f'<div style="font-family:\'Sora\',sans-serif;font-size:0.8rem;color:var(--text);line-height:1.6;">'
            f'{item_html}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)

# ── What goes into the score ──────────────────────────────────────────────────
st.markdown(
    '<div class="ee-label" style="margin-bottom:0.65rem;">What Goes Into The Score</div>',
    unsafe_allow_html=True,
)

# Accent colors chosen from the existing palette (gold, taupe already used
# broadly as neutral brand accents) plus rose (defined but otherwise unused)
# and one new muted slate-blue — deliberately avoiding green/red/yellow,
# which already mean LONG/SHORT/alert-level elsewhere in the app.
_factors = [
    ("Sentiment Drift",   settings.SCORE_WEIGHT_SENTIMENT_DRIFT,   "var(--gold)",
     "Is management sounding more confident or more nervous each quarter?"),
    ("Guidance Accuracy", settings.SCORE_WEIGHT_GUIDANCE_ACCURACY, "var(--rose)",
     "Did the company actually deliver what it promised last quarter?"),
    ("Accruals Quality",  settings.SCORE_WEIGHT_ACCRUALS,          "var(--taupe)",
     "Are the profits backed by real cash, or just accounting?"),
    ("Analyst Revisions", settings.SCORE_WEIGHT_ANALYST_REVISION,  "var(--slate)",
     "Are Wall Street analysts turning more bullish or bearish?"),
]

weight_cols = st.columns(4)
for col, (label, weight, color, sentence) in zip(weight_cols, _factors):
    with col:
        st.markdown(
            f'<div class="ee-card" style="border-top:3px solid {color} !important;'
            'text-align:center;padding:1.35rem 1.1rem;height:100%;">'
            f'<div style="font-family:\'Sora\',sans-serif;font-weight:700;font-size:0.8rem;'
            f'color:var(--text);margin-bottom:0.5rem;">{label}</div>'
            f'<div class="ee-score" style="font-size:2.1rem;color:{color};margin-bottom:0.55rem;">'
            f'{weight:.0%}</div>'
            f'<div style="font-family:\'Sora\',sans-serif;font-size:0.74rem;color:var(--text-muted);'
            f'line-height:1.45;">{sentence}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

st.caption("Fixed weights, not live values — see Ticker Analysis for a ticker's actual computed score.")

st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)

# ── Call to action ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ee-card ee-fade-in" style="text-align:center;padding:2rem;">'
    '<div class="ee-label" style="margin-bottom:0.5rem;">Ready To Run An Analysis?</div>'
    '<div style="font-family:\'Sora\',sans-serif;font-size:0.85rem;color:var(--text-muted);">'
    'Head to Ticker Analysis to pull SEC filings, compute the composite score, and query the RAG '
    'pipeline for any US-listed ticker.'
    '</div></div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)
st.page_link("pages/01_ticker_analysis.py", label="Go to Ticker Analysis →", icon="🔍")
