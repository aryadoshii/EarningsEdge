"""Watchlist & Batch Ingestion — EarningsEdge."""
from __future__ import annotations
import asyncio, time
from pathlib import Path

import streamlit as st

from config.settings import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ticker_dir(ticker: str) -> Path:
    return settings.processed_data_path / ticker.upper()

def _data_age_days(ticker: str) -> float | None:
    p = _ticker_dir(ticker) / "chunks.json"
    if not p.exists():
        return None
    return (time.time() - p.stat().st_mtime) / 86_400

def _status_badge(ticker: str) -> tuple[str, str]:
    """Return (label, css_color) describing a ticker's data state."""
    age = _data_age_days(ticker)
    if age is None:
        return "Not ingested", "var(--red)"
    if age > 7:
        return f"Stale ({age:.0f}d)", "#b8860b"
    return f"Fresh ({age:.1f}d)", "var(--green)"

def _run_pipeline_for(ticker: str, progress_placeholder) -> bool:
    """
    Run full ingest → embed → analyze for a single ticker.
    Writes status into *progress_placeholder*.
    Returns True on success.
    """
    from src.pipeline_runner import (
        run_ingest_pipeline,
        run_embed_pipeline,
        run_analysis_pipeline,
    )
    try:
        progress_placeholder.markdown(
            f'<span style="font-family:\'Space Mono\',monospace;font-size:0.7rem;">'
            f'[{ticker}] Fetching SEC filings…</span>', unsafe_allow_html=True
        )
        ingest_result = asyncio.run(run_ingest_pipeline(ticker))
        n_chunks = len(ingest_result.get("chunks", []))

        progress_placeholder.markdown(
            f'<span style="font-family:\'Space Mono\',monospace;font-size:0.7rem;">'
            f'[{ticker}] Embedding {n_chunks} chunks…</span>', unsafe_allow_html=True
        )
        asyncio.run(run_embed_pipeline(ticker))

        progress_placeholder.markdown(
            f'<span style="font-family:\'Space Mono\',monospace;font-size:0.7rem;">'
            f'[{ticker}] Running analysis…</span>', unsafe_allow_html=True
        )
        asyncio.run(run_analysis_pipeline(ticker))
        return True
    except Exception as exc:
        progress_placeholder.markdown(
            f'<span style="color:var(--red);font-family:\'Space Mono\',monospace;font-size:0.7rem;">'
            f'[{ticker}] Failed: {exc}</span>', unsafe_allow_html=True
        )
        return False

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="ee-page-header ee-fade-in" style="padding-bottom:1.25rem;'
    'border-bottom:1px solid var(--border);margin-bottom:1.5rem;">'
    '<h1 style="margin:0;font-size:1.75rem;letter-spacing:-0.03em;'
    'font-family:\'Sora\',sans-serif;font-weight:600;">Watchlist</h1>'
    '<p style="margin:0.2rem 0 0;font-size:0.8rem;color:var(--text-muted);">'
    'Manage tickers · Batch ingest · Data freshness</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Watchlist state (persisted in session)
# ---------------------------------------------------------------------------

if "watchlist" not in st.session_state:
    # Pre-populate from any tickers that already have local data
    existing = sorted(
        d.name for d in settings.processed_data_path.iterdir()
        if d.is_dir() and (d / "chunks.json").exists()
    )
    st.session_state["watchlist"] = existing or []

watchlist: list[str] = st.session_state["watchlist"]

# ---------------------------------------------------------------------------
# Add tickers
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([3, 1])
with col_input:
    new_raw = st.text_input(
        "Add tickers (comma-separated)",
        placeholder="e.g.  MSFT, NVDA, TSLA",
        label_visibility="collapsed",
    )
with col_btn:
    if st.button("Add", use_container_width=True):
        added = 0
        for t in [x.strip().upper() for x in new_raw.split(",") if x.strip()]:
            if t and t not in watchlist:
                watchlist.append(t)
                added += 1
        if added:
            st.success(f"Added {added} ticker(s)")
            st.rerun()

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Watchlist table
# ---------------------------------------------------------------------------

if not watchlist:
    st.markdown(
        '<div class="ee-card" style="text-align:center;padding:3rem 2rem;border-style:dashed;">'
        '<div style="font-size:2rem;opacity:0.2;margin-bottom:0.75rem;">◈</div>'
        '<div style="font-family:\'Sora\',sans-serif;font-size:0.875rem;color:var(--text-muted);">'
        'Your watchlist is empty — add tickers above</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Render table header
st.markdown(
    '<div style="display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:0.5rem;'
    'padding:0.4rem 0.75rem;font-family:\'Space Mono\',monospace;font-size:0.62rem;'
    'color:var(--text-muted);border-bottom:1px solid var(--border);margin-bottom:0.25rem;">'
    '<span>TICKER</span><span>STATUS</span><span>DATA PATH</span><span></span>'
    '</div>',
    unsafe_allow_html=True,
)

tickers_to_remove: list[str] = []
for ticker in watchlist:
    label, color = _status_badge(ticker)
    data_path = _ticker_dir(ticker)
    rel_path   = str(data_path.relative_to(Path.cwd())) if data_path.exists() else "—"

    col_t, col_s, col_p, col_del = st.columns([1, 1, 1, 0.15])
    with col_t:
        st.markdown(
            f'<div style="font-family:\'Sora\',sans-serif;font-weight:600;'
            f'font-size:0.9rem;padding:0.35rem 0;">{ticker}</div>',
            unsafe_allow_html=True,
        )
    with col_s:
        st.markdown(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;'
            f'color:{color};padding:0.35rem 0;">{label}</div>',
            unsafe_allow_html=True,
        )
    with col_p:
        st.markdown(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.62rem;'
            f'color:var(--text-dim);padding:0.35rem 0;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap;" title="{rel_path}">{rel_path}</div>',
            unsafe_allow_html=True,
        )
    with col_del:
        if st.button("✕", key=f"del_{ticker}", help=f"Remove {ticker} from watchlist"):
            tickers_to_remove.append(ticker)

for t in tickers_to_remove:
    watchlist.remove(t)
if tickers_to_remove:
    st.rerun()

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Batch ingestion controls
# ---------------------------------------------------------------------------

st.markdown('<div class="ee-label" style="margin-bottom:0.75rem;">Batch Ingestion</div>', unsafe_allow_html=True)

stale_tickers  = [t for t in watchlist if (_data_age_days(t) or 999) > 7]
new_tickers    = [t for t in watchlist if _data_age_days(t) is None]

col_a, col_b, col_c = st.columns(3)
with col_a:
    ingest_new = st.button(
        f"▶ Ingest new ({len(new_tickers)})",
        disabled=not new_tickers,
        use_container_width=True,
        type="primary",
    )
with col_b:
    refresh_stale = st.button(
        f"↺ Refresh stale ({len(stale_tickers)})",
        disabled=not stale_tickers,
        use_container_width=True,
    )
with col_c:
    refresh_all = st.button(
        f"⟳ Re-ingest all ({len(watchlist)})",
        use_container_width=True,
    )

targets: list[str] = []
if ingest_new:
    targets = new_tickers
elif refresh_stale:
    targets = stale_tickers
elif refresh_all:
    targets = watchlist

if targets:
    st.markdown(
        f'<div style="font-family:\'Space Mono\',monospace;font-size:0.72rem;'
        f'color:var(--text-muted);margin-bottom:0.75rem;">'
        f'Ingesting: {" · ".join(targets)}</div>',
        unsafe_allow_html=True,
    )

    progress_bar = st.progress(0)
    status_text  = st.empty()
    results: dict[str, bool] = {}

    for i, ticker in enumerate(targets):
        success = _run_pipeline_for(ticker, status_text)
        results[ticker] = success
        progress_bar.progress((i + 1) / len(targets))

    # Summary
    succeeded = [t for t, ok in results.items() if ok]
    failed    = [t for t, ok in results.items() if not ok]
    status_text.empty()

    if succeeded:
        st.success(f"Ingested: {', '.join(succeeded)}")
    if failed:
        st.error(f"Failed: {', '.join(failed)}")

    st.rerun()

# ---------------------------------------------------------------------------
# Quick-jump to analysis
# ---------------------------------------------------------------------------

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="ee-label" style="margin-bottom:0.5rem;">Quick Analysis</div>', unsafe_allow_html=True)

fresh = [t for t in watchlist if (_data_age_days(t) or 999) <= 7]
if fresh:
    cols = st.columns(min(len(fresh), 6))
    for col, ticker in zip(cols, fresh):
        with col:
            if st.button(ticker, use_container_width=True, key=f"jump_{ticker}"):
                st.switch_page("pages/01_ticker_analysis.py")
else:
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;'
        'color:var(--text-dim);">Ingest some tickers first to enable quick jump.</div>',
        unsafe_allow_html=True,
    )
