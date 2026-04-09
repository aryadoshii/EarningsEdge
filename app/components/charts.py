"""
Reusable Plotly chart functions for the EarningsEdge Streamlit dashboard.

All functions return a plotly.graph_objects.Figure ready for
st.plotly_chart(fig, use_container_width=True).

Charts:
    quality_score_gauge      — composite score semicircle gauge
    component_bar_chart      — score component breakdown horizontal bars
    sentiment_trend_line     — quarter-by-quarter net sentiment line
    hedging_heatmap          — hedging language intensity by quarter
    equity_curve             — strategy vs benchmark equity growth
    trade_return_histogram   — distribution of trade returns
    ragas_gauge              — RAGAS metric gauge cluster
    contradiction_timeline   — dots on timeline showing conflict events
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# EarningsEdge colour palette
_GREEN   = "#7F9275"
_RED     = "#B88476"
_YELLOW  = "#C7AB63"
_BLUE    = "#AD9C8E"
_NEUTRAL = "#8F7F73"
_BG      = "#FBF5EC"
_SURFACE = "#F3E8DA"
_TEXT    = "#241D17"

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_SURFACE,
    font=dict(color=_TEXT, family="Sora, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ---------------------------------------------------------------------------
# Quality score gauge
# ---------------------------------------------------------------------------

def quality_score_gauge(score: float, signal: str, ticker: str) -> go.Figure:
    """
    Semicircle gauge showing the composite earnings quality score.

    Args:
        score:  Composite score in [-1, +1].
        signal: "LONG" / "SHORT" / "NEUTRAL".
        ticker: Ticker for the title.

    Returns:
        Plotly Figure.
    """
    color = _GREEN if score > 0.3 else _RED if score < -0.3 else _YELLOW
    pct = (score + 1) / 2 * 100   # map [-1,+1] to [0,100]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 3),
        number={"font": {"size": 42, "color": color}},
        delta={"reference": 0, "relative": False},
        title={"text": f"{ticker} Earnings Quality<br><span style='font-size:14px'>{signal}</span>"},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": _TEXT},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": _SURFACE,
            "borderwidth": 0,
            "threshold": {
                "line": {"color": _TEXT, "width": 2},
                "thickness": 0.75,
                "value": 0,
            },
            "steps": [
                {"range": [-1, -0.3], "color": "#E8D1C8"},
                {"range": [-0.3, 0.3], "color": "#EFE3D4"},
                {"range": [0.3, 1],   "color": "#DCE5D7"},
            ],
        },
    ))
    fig.update_layout(**_LAYOUT_BASE, height=280)
    return fig


# ---------------------------------------------------------------------------
# Component bar chart
# ---------------------------------------------------------------------------

def component_bar_chart(
    components: dict[str, float],
    weights: dict[str, float],
) -> go.Figure:
    """
    Horizontal bar chart showing score component breakdown.

    Args:
        components: {component_name: value} — each in [-1, +1].
        weights:    {component_name: weight} — fraction of composite.

    Returns:
        Plotly Figure.
    """
    labels = list(components.keys())
    values = [components[k] for k in labels]
    weight_labels = [f"{weights.get(k, 0):.0%}" for k in labels]
    colors = [_GREEN if v > 0 else _RED for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f} (wt {w})" for v, w in zip(values, weight_labels)],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Score Component Breakdown",
        xaxis=dict(range=[-1.2, 1.2], zeroline=True,
                   zerolinecolor=_NEUTRAL, zerolinewidth=1),
        height=250,
    )
    return fig


# ---------------------------------------------------------------------------
# Sentiment trend line
# ---------------------------------------------------------------------------

def sentiment_trend_line(
    quarters: list[str],
    sentiment_scores: list[float],
    hedging_scores: list[float],
    alert_level: str = "GREEN",
) -> go.Figure:
    """
    Dual-axis line chart of sentiment score and hedging language.

    Args:
        quarters:        List of quarter labels e.g. ["Q1 2023", "Q2 2023"].
        sentiment_scores: Net sentiment score per quarter.
        hedging_scores:  Hedging language frequency per quarter.
        alert_level:     "GREEN" / "YELLOW" / "RED" for background shade.

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Background shade based on alert level
    bg_color = {"GREEN": "#EAF1E7", "YELLOW": "#F2EAD6", "RED": "#F2DFD8"}.get(
        alert_level, _SURFACE
    )

    fig.add_trace(go.Scatter(
        x=quarters, y=sentiment_scores, name="Net Sentiment",
        line=dict(color=_BLUE, width=3),
        mode="lines+markers",
        marker=dict(size=10),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=quarters, y=hedging_scores, name="Hedging Score",
        line=dict(color=_YELLOW, width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=7, symbol="diamond"),
    ), secondary_y=True)

    # Zero line on sentiment axis
    fig.add_hline(y=0, line_dash="dash", line_color=_NEUTRAL,
                  line_width=1, secondary_y=False)

    fig.update_layout(
        **_LAYOUT_BASE,
        title="Quarter-over-Quarter Tone Drift",
        plot_bgcolor=bg_color,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
    )
    fig.update_yaxes(title_text="Net Sentiment Score", secondary_y=False)
    fig.update_yaxes(title_text="Hedging Frequency",   secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Hedging language heatmap
# ---------------------------------------------------------------------------

def hedging_heatmap(
    quarters: list[str],
    hedging_by_section: dict[str, list[float]],
) -> go.Figure:
    """
    Heatmap of hedging language frequency by quarter and section.

    Args:
        quarters: List of quarter labels.
        hedging_by_section: {section_name: [value per quarter]}.

    Returns:
        Plotly Figure.
    """
    sections = list(hedging_by_section.keys())
    z = [hedging_by_section[s] for s in sections]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=quarters,
        y=sections,
        colorscale=[[0, _SURFACE], [0.5, _YELLOW], [1, _RED]],
        showscale=True,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Hedging Language Intensity by Section",
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def equity_curve(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    initial_capital: float = 100_000,
) -> go.Figure:
    """
    Line chart comparing strategy and benchmark equity growth.

    Args:
        strategy_equity:  pd.Series with DatetimeIndex (strategy).
        benchmark_equity: pd.Series with DatetimeIndex (SPY). Optional.
        initial_capital:  Starting equity for annotation.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    if not strategy_equity.empty:
        fig.add_trace(go.Scatter(
            x=strategy_equity.index,
            y=strategy_equity.values,
            name="EarningsEdge Strategy",
            line=dict(color=_GREEN, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,200,150,0.07)",
        ))

    if benchmark_equity is not None and not benchmark_equity.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity.values,
            name="SPY Buy-and-Hold",
            line=dict(color=_NEUTRAL, width=1.5, dash="dot"),
        ))

    fig.add_hline(
        y=initial_capital,
        line_dash="dash", line_color=_NEUTRAL, line_width=1,
        annotation_text="Initial Capital",
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Trade return histogram
# ---------------------------------------------------------------------------

def trade_return_histogram(returns: list[float]) -> go.Figure:
    """
    Distribution of gross returns across all trades.

    Args:
        returns: List of gross return floats (e.g. 0.05 = 5%).

    Returns:
        Plotly Figure.
    """
    returns_pct = [r * 100 for r in returns]
    colors = [_GREEN if r > 0 else _RED for r in returns_pct]

    fig = go.Figure(go.Histogram(
        x=returns_pct,
        nbinsx=30,
        marker_color=_BLUE,
        marker_line=dict(width=0.5, color=_BG),
        opacity=0.85,
        name="Trade returns",
    ))
    fig.add_vline(x=0, line_dash="solid", line_color=_NEUTRAL, line_width=2)

    mean_r = sum(returns_pct) / len(returns_pct) if returns_pct else 0
    fig.add_vline(
        x=mean_r, line_dash="dash", line_color=_YELLOW, line_width=1.5,
        annotation_text=f"Mean: {mean_r:.1f}%",
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title="Trade Return Distribution",
        xaxis_title="Gross Return (%)",
        yaxis_title="Count",
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# RAGAS gauge cluster
# ---------------------------------------------------------------------------

def ragas_gauges(scores: dict[str, float | None]) -> go.Figure:
    """
    Four-panel gauge cluster showing RAGAS metric scores.

    Args:
        scores: Dict with keys faithfulness, answer_relevance,
                context_precision, context_recall — each in [0, 1] or None.

    Returns:
        Plotly Figure with 2×2 gauge subplots.
    """
    metric_labels = {
        "faithfulness":      "Faithfulness",
        "answer_relevance":  "Answer Relevance",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{"type": "indicator"}] * 2] * 2,
                        vertical_spacing=0.15)

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (r, c), (key, label) in zip(positions, metric_labels.items()):
        value = scores.get(key)
        color = (
            _GREEN  if value is not None and value >= 0.80 else
            _YELLOW if value is not None and value >= 0.65 else
            _RED    if value is not None else _NEUTRAL
        )
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value if value is not None else 0,
            title={"text": label, "font": {"size": 13}},
            number={"font": {"size": 22, "color": color},
                    "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": _SURFACE,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 0.65],  "color": "#2D1B1B"},
                    {"range": [0.65, 0.8], "color": "#2D2B1B"},
                    {"range": [0.8, 1],   "color": "#1B2D22"},
                ],
            },
        ), row=r, col=c)

    fig.update_layout(
        **_LAYOUT_BASE,
        title="RAG Evaluation — RAGAS Scores",
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Contradiction timeline
# ---------------------------------------------------------------------------

def contradiction_timeline(
    contradictions: list[dict[str, Any]],
    ticker: str,
) -> go.Figure:
    """
    Scatter plot showing contradictions on a timeline.

    Args:
        contradictions: List of contradiction dicts with keys
                        chunk_a_source, contradiction_score, interpretation.
        ticker:         Ticker for title.

    Returns:
        Plotly Figure.
    """
    if not contradictions:
        fig = go.Figure()
        fig.update_layout(
            **_LAYOUT_BASE,
            title=f"{ticker} — No contradictions detected",
            height=200,
        )
        return fig

    sources = [c.get("chunk_a_source", "Unknown")[:40] for c in contradictions]
    scores  = [c.get("contradiction_score", 0) for c in contradictions]
    texts   = [c.get("interpretation", "")[:80] for c in contradictions]

    fig = go.Figure(go.Scatter(
        x=sources,
        y=scores,
        mode="markers+text",
        marker=dict(
            size=[s * 30 + 10 for s in scores],
            color=[_RED if s > 0.85 else _YELLOW for s in scores],
            opacity=0.85,
            line=dict(color=_BG, width=1),
        ),
        text=[f"{s:.2f}" for s in scores],
        textposition="top center",
        hovertext=texts,
        hoverinfo="x+y+text",
        name="Contradiction Score",
    ))
    fig.add_hline(
        y=0.85, line_dash="dot", line_color=_RED, line_width=1,
        annotation_text="High confidence threshold",
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"{ticker} — Detected Contradictions",
        xaxis_title="Source Document",
        yaxis=dict(title="Contradiction Score", range=[0, 1.1]),
        height=320,
    )
    return fig
