"""EarningsEdge — app entry point. Uses st.navigation for instant page switching."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="EarningsEdge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.theme import inject_theme, sidebar_nav
inject_theme()

# ── Define pages ──────────────────────────────────────────────────────────────
pages = [
    st.Page("pages/home.py",                title="Overview",        icon="🏠", default=True),
    st.Page("pages/00_watchlist.py",         title="Watchlist",       icon="⭐"),
    st.Page("pages/01_ticker_analysis.py",   title="Ticker Analysis", icon="🔍"),
    st.Page("pages/02_tone_drift.py",        title="Tone Drift",      icon="📊"),
    st.Page("pages/03_backtest_results.py",  title="Backtest Results",icon="📈"),
    st.Page("pages/04_rag_evaluation.py",    title="RAG Evaluation",  icon="🎯"),
]

pg = st.navigation(pages, position="hidden")  # hide default nav — we use our own

# ── Sidebar (rendered once, persists across all pages) ────────────────────────
sidebar_nav(pg.title)

# ── Run current page ──────────────────────────────────────────────────────────
pg.run()
