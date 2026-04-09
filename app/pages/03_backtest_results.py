"""Backtest Results — EarningsEdge."""
from __future__ import annotations
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
st.set_page_config(page_title="Backtest · EarningsEdge", layout="wide", initial_sidebar_state="expanded")

from app.components.theme import inject_theme, sidebar_nav
inject_theme()
sidebar_nav("backtest")

from config.settings import settings

with st.sidebar:
    st.markdown('<div style="padding:0.5rem 0 0.75rem;border-bottom:1px solid var(--border);margin-bottom:0.9rem;"><div class="ee-label">Backtest Controls</div></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;color:var(--text-muted);line-height:2.1;">'
        f'Period: {settings.BACKTEST_START_DATE} → {settings.BACKTEST_END_DATE}<br>'
        f'Holding: {settings.HOLDING_PERIOD_DAYS} days<br>'
        f'Capital: ${settings.INITIAL_CAPITAL:,.0f}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_bt = st.button("▶ Run Backtest", use_container_width=True, type="primary")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    filter_dir = st.multiselect("Signal Filter", ["LONG","SHORT"], default=["LONG","SHORT"])

st.markdown(
    '<div class="ee-fade-in" style="padding-bottom:1.25rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;">'
    '<h1 style="margin:0;font-size:1.75rem;letter-spacing:-0.03em;font-family:\'Sora\',sans-serif;font-weight:600;">Backtest Results</h1>'
    '<p style="margin:0.2rem 0 0;font-size:0.8rem;color:var(--text-muted);">Signal backtesting · Equity curve · Risk-adjusted performance · Benchmark comparison</p>'
    '</div>',
    unsafe_allow_html=True,
)

if run_bt:
    with st.spinner("Running backtest engine…"):
        try:
            from src.backtest.backtester import backtester
            from src.backtest.signal_generator import signal_generator
            from src.backtest.metrics import compute_all_metrics
            from src.backtest.benchmark import compute_benchmark_comparison
            scores = st.session_state.get("quality_scores", [])
            if not scores:
                st.warning("No quality scores. Run `make analyze TICKER=X` first.")
                st.stop()
            signals = signal_generator.generate(scores)
            trades, equity, benchmark = asyncio.run(backtester.run(signals))
            metrics = compute_all_metrics(trades, benchmark)
            comparison = compute_benchmark_comparison(equity, benchmark)
            st.session_state.update({"bt_trades":trades,"bt_equity":equity,"bt_benchmark":benchmark,"bt_metrics":metrics,"bt_comparison":comparison})
        except Exception as e:
            st.error(f"Backtest failed: {e}")

metrics    = st.session_state.get("bt_metrics")
trades_raw = st.session_state.get("bt_trades", [])
equity_s   = st.session_state.get("bt_equity")
bench_s    = st.session_state.get("bt_benchmark")

if not metrics:
    # Placeholder
    st.markdown(
        '<div class="ee-card" style="text-align:center;padding:3rem 2rem;margin-bottom:1.5rem;border-style:dashed;">'
        '<div style="font-size:2rem;opacity:0.15;margin-bottom:0.6rem;">◎</div>'
        '<div style="font-family:\'Sora\',sans-serif;font-size:0.875rem;color:var(--text-muted);">'
        'Click <strong style="color:var(--taupe);">Run Backtest</strong> to execute the signal simulation</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    targets_cols = st.columns(4)
    for col, (label, val, color) in zip(targets_cols, [
        ("Sharpe Ratio",    "> 1.2",   "var(--gold)"),
        ("Hit Rate",        "> 60%",   "var(--green)"),
        ("Annual Return",   "> 15%",   "var(--taupe)"),
        ("Info Coefficient","0.05–0.1","var(--text-muted)"),
    ]):
        with col:
            st.markdown(
                '<div class="ee-card" style="text-align:center;padding:1.1rem 0.9rem;">'
                f'<div class="ee-label">{label}</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:1.15rem;font-weight:700;color:{color};margin-top:0.3rem;">{val}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
o = metrics.get("overall", {})
b = metrics.get("benchmark", {})
c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.metric("Sharpe Ratio",    f"{o.get('sharpe',0):.3f}")
with c2: st.metric("Hit Rate",        f"{o.get('hit_rate',0):.1%}")
with c3: st.metric("Annual Return",   f"{o.get('annualised_return',0):.1%}",
                   delta=f"{o.get('annualised_return',0)-b.get('annualised_return',0):+.1%} vs SPY" if b else None)
with c4: st.metric("Max Drawdown",    f"{o.get('max_drawdown',0):.1%}", delta_color="inverse")
with c5: st.metric("Info Coefficient",f"{o.get('ic',0):.4f}")
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Equity curve ──────────────────────────────────────────────────────────────
import plotly.graph_objects as go
if equity_s is not None and not equity_s.empty:
    st.markdown('<div class="ee-label" style="margin-bottom:0.5rem;">Equity Curve</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_s.index, y=equity_s.values, name="EarningsEdge",
        line=dict(color="#C9B276",width=2.5), fill="tozeroy", fillcolor="rgba(201,178,118,0.12)"))
    if bench_s is not None and not bench_s.empty:
        fig.add_trace(go.Scatter(x=bench_s.index, y=bench_s.values, name="SPY",
            line=dict(color="rgba(173,156,142,0.75)",width=1.5,dash="dot")))
    fig.add_hline(y=settings.INITIAL_CAPITAL, line_dash="dot", line_color="rgba(95,78,63,0.14)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,250,244,0.6)",
        font=dict(family="'Space Mono',monospace",color="#6E6056",size=9),
        height=330, margin=dict(l=0,r=0,t=10,b=0), showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
                    font=dict(size=9),bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(95,78,63,0.08)",linecolor="rgba(95,78,63,0.12)"),
        yaxis=dict(gridcolor="rgba(95,78,63,0.08)",linecolor="rgba(95,78,63,0.12)",
                   tickprefix="$",tickformat=",.0f"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Long vs Short ─────────────────────────────────────────────────────────────
lm = metrics.get("long",{}); sm = metrics.get("short",{})
if lm or sm:
    st.markdown('<div class="ee-label" style="margin-bottom:0.5rem;">Long vs Short Leg</div>', unsafe_allow_html=True)
    lc, sc_col = st.columns(2)
    for col, (leg_name, m, color) in zip([lc, sc_col],[("Long Leg",lm,"var(--green)"),("Short Leg",sm,"var(--red)")]):
        with col:
            rows_html = ""
            for lbl, val in [("Trades",str(m.get("trade_count",0))),("Sharpe",f"{m.get('sharpe',0):.3f}"),
                              ("Hit Rate",f"{m.get('hit_rate',0):.1%}"),("Ann. Return",f"{m.get('annualised_return',0):.1%}"),
                              ("Avg Win",f"{m.get('avg_win',0):.2%}"),("IC",f"{m.get('ic',0):.4f}")]:
                rows_html += (
                    '<div style="background:rgba(255,255,255,0.26);border:1px solid var(--border);border-radius:10px;padding:0.45rem 0.6rem;">'
                    f'<div class="ee-label" style="margin-bottom:1px;">{lbl}</div>'
                    f'<div style="font-family:\'Space Mono\',monospace;font-size:0.85rem;font-weight:700;color:var(--text);">{val}</div>'
                    '</div>'
                )
            st.markdown(
                '<div class="ee-card">'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;color:{color};'
                f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.65rem;">{leg_name}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;">{rows_html}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

# ── Trade log ─────────────────────────────────────────────────────────────────
from src.backtest.backtester import Backtester
if trades_raw:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ee-label" style="margin-bottom:0.5rem;">Trade Log</div>', unsafe_allow_html=True)
    df_log = Backtester.to_dataframe([t for t in trades_raw if t.direction.value in filter_dir])
    if not df_log.empty:
        st.dataframe(df_log, use_container_width=True, height=260)
        st.download_button("⬇ Download CSV", df_log.to_csv(index=False), "trade_log.csv", "text/csv")
