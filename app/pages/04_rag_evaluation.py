"""RAG Evaluation — EarningsEdge."""
from __future__ import annotations
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
st.set_page_config(page_title="RAG Evaluation · EarningsEdge", layout="wide", initial_sidebar_state="expanded")

from app.components.theme import inject_theme, sidebar_nav
inject_theme()
sidebar_nav("ragas")

with st.sidebar:
    st.markdown('<div style="padding:0.5rem 0 0.75rem;border-bottom:1px solid var(--border);margin-bottom:0.9rem;"><div class="ee-label">Evaluation</div></div>', unsafe_allow_html=True)
    eval_btn = st.button("▶ Run RAGAS", use_container_width=True, type="primary")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    mlflow_ticker = st.text_input("Filter runs", value="")
    n_runs = st.slider("Recent runs", 5, 50, 10)

st.markdown(
    '<div class="ee-fade-in" style="padding-bottom:1.25rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;">'
    '<h1 style="margin:0;font-size:1.75rem;letter-spacing:-0.03em;font-family:\'Sora\',sans-serif;font-weight:600;">RAG Evaluation</h1>'
    '<p style="margin:0.2rem 0 0;font-size:0.8rem;color:var(--text-muted);">RAGAS faithfulness · Answer relevance · Context precision · Context recall</p>'
    '</div>',
    unsafe_allow_html=True,
)

result = st.session_state.get("analysis_result")

if eval_btn:
    if not result or not result.ragas_input:
        st.error("No analysis result. Run a ticker analysis first.")
    else:
        with st.spinner("Running RAGAS evaluation…"):
            try:
                from src.evaluation.ragas_evaluator import ragas_evaluator
                ri = result.ragas_input
                ev = asyncio.run(ragas_evaluator.evaluate(
                    query=ri.get("query",""), answer=ri.get("answer",""),
                    contexts=ri.get("contexts",[]), ground_truth=ri.get("ground_truth",""),
                ))
                st.session_state["last_ragas_eval"] = ev
                try:
                    from src.evaluation.mlflow_tracker import mlflow_tracker
                    mlflow_tracker.log_ragas_evaluation(ev)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"RAGAS evaluation failed: {e}")

eval_obj = st.session_state.get("last_ragas_eval")

metrics_def = [
    ("faithfulness",      "Faithfulness",      (0.80, 0.90)),
    ("answer_relevance",  "Answer Relevance",  (0.75, 0.85)),
    ("context_precision", "Context Precision", (0.70, 0.80)),
    ("context_recall",    "Context Recall",    (0.65, 0.80)),
]

if eval_obj:
    st.markdown('<div class="ee-label" style="margin-bottom:0.6rem;">RAGAS Scores</div>', unsafe_allow_html=True)
    score_cols = st.columns(4)
    for col, (key, label, (lo, hi)) in zip(score_cols, metrics_def):
        val = getattr(eval_obj, key, None)
        if val is not None:
            color  = "#7F9275" if val >= hi else "#C7AB63" if val >= lo else "#B88476"
            badge  = "EXCELLENT" if val >= hi else "ACCEPTABLE" if val >= lo else "IMPROVE"
            fill   = int(val * 100)
        else:
            color, badge, fill, val = "#9B8A7D", "N/A", 0, 0.0
        with col:
            st.markdown(
                '<div class="ee-card" style="text-align:center;padding:1.4rem 1rem;">'
                f'<div class="ee-label">{label}</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:1.85rem;font-weight:700;color:{color};line-height:1.1;margin:0.35rem 0;">{val:.3f}</div>'
                '<div style="width:100%;height:3px;background:rgba(173,156,142,0.16);border-radius:2px;margin:0.5rem 0;">'
                f'<div style="width:{fill}%;height:100%;background:{color};border-radius:2px;"></div></div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.6rem;font-weight:700;'
                f'letter-spacing:0.1em;color:{color};">{badge}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    # Mean score
    mean = eval_obj.mean_score
    mc   = "#7F9275" if mean >= 0.8 else "#C7AB63" if mean >= 0.65 else "#B88476"
    st.markdown(
        '<div class="ee-card" style="padding:1.1rem 1.3rem;margin-top:0.75rem;">'
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">'
        '<div class="ee-label">Mean RAGAS Score</div>'
        f'<div style="font-family:\'Space Mono\',monospace;font-size:1rem;font-weight:700;color:{mc};">{mean:.4f}</div>'
        '</div>'
        '<div style="width:100%;height:5px;background:rgba(173,156,142,0.16);border-radius:3px;">'
        f'<div style="width:{int(mean*100)}%;height:100%;background:{mc};border-radius:3px;"></div></div>'
        '<div style="display:flex;justify-content:space-between;margin-top:0.45rem;'
        'font-family:\'Space Mono\',monospace;font-size:0.62rem;color:var(--text-dim);">'
        f'<span>Model: {eval_obj.model_used or "N/A"}</span>'
        f'<span>Latency: {eval_obj.latency_ms:.0f}ms</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    # Empty state + benchmark targets
    st.markdown(
        '<div class="ee-card" style="text-align:center;padding:3rem 2rem;margin-bottom:1.5rem;border-style:dashed;">'
        '<div style="font-size:2rem;opacity:0.15;margin-bottom:0.6rem;">◉</div>'
        '<div style="font-family:\'Sora\',sans-serif;font-size:0.875rem;color:var(--text-muted);">'
        'Click <strong style="color:var(--taupe);">Run RAGAS</strong> to score the last response</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ee-label" style="margin-bottom:0.6rem;">Quality Benchmarks</div>', unsafe_allow_html=True)
    bench_cols = st.columns(4)
    for col, (label, acc, exc, color) in zip(bench_cols, [
        ("Faithfulness","≥ 0.80","≥ 0.90","var(--gold)"),
        ("Answer Relevance","≥ 0.75","≥ 0.85","var(--taupe)"),
        ("Context Precision","≥ 0.70","≥ 0.80","var(--green)"),
        ("Context Recall","≥ 0.65","≥ 0.80","var(--text-muted)"),
    ]):
        with col:
            st.markdown(
                '<div class="ee-card" style="padding:1.1rem 1rem;">'
                f'<div class="ee-label" style="margin-bottom:0.5rem;">{label}</div>'
                f'<div style="font-family:\'Sora\',sans-serif;font-size:0.72rem;color:var(--text-muted);">'
                f'Acceptable: <strong style="color:{color};">{acc}</strong><br>'
                f'Excellent: <strong style="color:{color};">{exc}</strong></div>'
                '</div>',
                unsafe_allow_html=True,
            )

# ── MLflow table ──────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="ee-label" style="margin-bottom:0.6rem;">Experiment History</div>', unsafe_allow_html=True)
try:
    from src.evaluation.mlflow_tracker import mlflow_tracker, _mlflow_available
    if _mlflow_available:
        runs_df = mlflow_tracker.get_recent_runs(
            n=n_runs, ticker=mlflow_ticker.upper().strip() or None,
        )
        if not runs_df.empty:
            st.dataframe(runs_df, use_container_width=True)
        else:
            st.markdown('<div class="ee-card" style="text-align:center;padding:1.5rem;"><div style="font-family:\'Space Mono\',monospace;font-size:0.72rem;color:var(--text-muted);">No runs yet · Logged automatically during analysis</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ee-card" style="padding:1rem;"><div style="font-family:\'Sora\',sans-serif;font-size:0.8rem;color:var(--text-muted);">MLflow available · Runs appear after analysis completes.</div></div>', unsafe_allow_html=True)
except Exception as e:
    st.markdown(f'<div class="ee-card"><div style="font-family:\'Space Mono\',monospace;font-size:0.72rem;color:var(--text-muted);">MLflow: {e}</div></div>', unsafe_allow_html=True)
