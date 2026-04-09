"""Tone Drift — EarningsEdge."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
st.set_page_config(page_title="Tone Drift · EarningsEdge", layout="wide", initial_sidebar_state="expanded")

from app.components.theme import inject_theme, sidebar_nav
inject_theme()
sidebar_nav("drift")

st.markdown(
    '<div class="ee-fade-in" style="padding-bottom:1.25rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;">'
    '<h1 style="margin:0;font-size:1.75rem;letter-spacing:-0.03em;font-family:\'Sora\',sans-serif;font-weight:600;">Tone Drift Analysis</h1>'
    '<p style="margin:0.2rem 0 0;font-size:0.8rem;color:var(--text-muted);">Quarter-over-quarter shifts in management language · Hedging frequency · Specificity decay</p>'
    '</div>',
    unsafe_allow_html=True,
)

result = st.session_state.get("analysis_result")
if not result:
    st.markdown(
        '<div class="ee-card" style="text-align:center;padding:4rem 2rem;border-style:dashed;">'
        '<div style="font-size:2.5rem;opacity:0.15;margin-bottom:0.75rem;">⟁</div>'
        '<div style="font-family:\'Sora\',sans-serif;font-size:0.875rem;color:var(--text-muted);">'
        'Run an analysis on the <strong style="color:var(--taupe);">Ticker Analysis</strong> page first</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

dr        = result.tone_drift_report or {}
quarters  = dr.get("quarters_analysed", [])
sentiment = dr.get("sentiment_trend", [])
hedging   = dr.get("hedging_trend", [])
spec      = dr.get("specificity_trend", [])
alert     = dr.get("alert_level", "GREEN")
reason    = dr.get("alert_reason", "No significant drift detected")
consec    = dr.get("consecutive_deterioration_count", 0)

if not quarters:
    st.info("No tone drift data available for this ticker.")
    st.stop()

# ── Alert banner ──────────────────────────────────────────────────────────────
alert_styles = {
    "GREEN":  ("#7F9275", "rgba(127,146,117,0.10)", "rgba(127,146,117,0.24)"),
    "YELLOW": ("#C7AB63", "rgba(199,171,99,0.10)", "rgba(199,171,99,0.24)"),
    "RED":    ("#B88476", "rgba(217,187,176,0.22)", "rgba(184,132,118,0.24)"),
}
ac, abg, aborder = alert_styles.get(alert, alert_styles["GREEN"])
st.markdown(
    f'<div style="background:{abg};border:1px solid {aborder};border-left:3px solid {ac};'
    f'border-radius:12px;padding:0.9rem 1.1rem;margin-bottom:1.25rem;">'
    '<div style="display:flex;align-items:center;justify-content:space-between;">'
    f'<div><span class="ee-alert {alert}" style="margin-right:0.6rem;">{alert}</span>'
    f'<span style="font-family:\'Sora\',sans-serif;font-size:0.85rem;color:var(--text);">{reason}</span></div>'
    f'<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;color:var(--text-muted);">'
    f'Consec. det.: <strong style="color:{ac};">{consec}</strong></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

# ── KPI metrics ───────────────────────────────────────────────────────────────
if sentiment:
    c1,c2,c3,c4 = st.columns(4)
    delta_s  = sentiment[-1]-sentiment[-2] if len(sentiment)>1 else 0
    delta_h  = hedging[-1]-hedging[-2]     if len(hedging)>1  else 0
    delta_sp = spec[-1]-spec[-2]           if len(spec)>1     else 0
    with c1: st.metric("Current Sentiment",  f"{sentiment[-1]:+.3f}", delta=f"{delta_s:+.3f}")
    with c2: st.metric("Hedging Score",      f"{hedging[-1]:.2f}" if hedging else "N/A",
                       delta=f"{delta_h:+.2f}", delta_color="inverse")
    with c3: st.metric("Specificity Score",  f"{spec[-1]:.2f}" if spec else "N/A",
                       delta=f"{delta_sp:+.2f}")
    with c4: st.metric("Quarters Analysed",  str(len(quarters)))
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,250,244,0.6)",
    font=dict(family="'Space Mono', monospace", color="#6E6056", size=9),
    showlegend=False, margin=dict(l=0,r=0,t=32,b=0),
    xaxis=dict(gridcolor="rgba(95,78,63,0.08)", linecolor="rgba(95,78,63,0.12)", tickfont=dict(size=8)),
    yaxis=dict(gridcolor="rgba(95,78,63,0.08)", linecolor="rgba(95,78,63,0.12)"),
)

if quarters and sentiment:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Net Sentiment Score", "Hedging Language Frequency",
                        "Guidance Specificity", "QoQ Sentiment Delta"],
        vertical_spacing=0.16, horizontal_spacing=0.09,
    )
    fig.add_trace(go.Scatter(x=quarters, y=sentiment, mode="lines+markers",
        line=dict(color="#C9B276",width=2.5), marker=dict(size=7,color="#C9B276",line=dict(color="#FFF8F1",width=1.5)),
        fill="tozeroy", fillcolor="rgba(201,178,118,0.12)"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(95,78,63,0.18)", row=1, col=1)

    if hedging:
        hc = ["#B88476" if h>8 else "#C7AB63" if h>4 else "#7F9275" for h in hedging]
        fig.add_trace(go.Bar(x=quarters, y=hedging, marker_color=hc, marker_line_width=0, opacity=0.8), row=1, col=2)

    if spec:
        fig.add_trace(go.Scatter(x=quarters, y=spec, mode="lines+markers",
            line=dict(color="#AD9C8E",width=2.5), marker=dict(size=7,color="#AD9C8E",line=dict(color="#FFF8F1",width=1.5)),
            fill="tozeroy", fillcolor="rgba(173,156,142,0.12)"), row=2, col=1)

    if len(sentiment) > 1:
        deltas = [sentiment[i]-sentiment[i-1] for i in range(1,len(sentiment))]
        dc = ["#7F9275" if d>0 else "#B88476" for d in deltas]
        fig.add_trace(go.Bar(x=quarters[1:], y=deltas, marker_color=dc, marker_line_width=0, opacity=0.85), row=2, col=2)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(95,78,63,0.18)", row=2, col=2)

    fig.update_layout(**_layout, height=500)
    for ann in fig.layout.annotations:
        ann.font.update(color="#6E6056", size=9, family="'Sora', sans-serif")
    for axis in [fig.layout.xaxis2, fig.layout.xaxis3, fig.layout.xaxis4,
                 fig.layout.yaxis2, fig.layout.yaxis3, fig.layout.yaxis4]:
        if axis:
            axis.update(gridcolor="rgba(95,78,63,0.08)", linecolor="rgba(95,78,63,0.12)")
    st.plotly_chart(fig, use_container_width=True)

# ── Mini sentiment timeline ───────────────────────────────────────────────────
if len(quarters) >= 2 and sentiment:
    st.markdown('<div class="ee-label" style="margin-bottom:0.5rem;">Sentiment Timeline</div>', unsafe_allow_html=True)
    max_s = max(abs(s) for s in sentiment) or 1
    bars = ""
    for q, s in zip(quarters, sentiment):
        h = max(8, int(abs(s)/max_s*50))
        color = "#7F9275" if s > 0.05 else "#B88476" if s < -0.05 else "#C7AB63"
        op = 0.45 + 0.55 * abs(s)/max_s
        bars += f'<div title="{q}: {s:+.3f}" style="flex:1;height:{h}px;background:{color};opacity:{op:.2f};border-radius:3px 3px 0 0;min-width:10px;"></div>'
    st.markdown(
        '<div class="ee-card" style="padding:0.75rem 1rem;">'
        f'<div style="display:flex;gap:3px;align-items:flex-end;height:56px;">{bars}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
