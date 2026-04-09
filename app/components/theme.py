"""EarningsEdge shared theme — Obsidian Terminal design system."""
from __future__ import annotations
import streamlit as st


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Sora:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg: #fbf5ec;
  --bg2: #f3e8da;
  --bg3: #fffaf4;
  --surface: rgba(255, 250, 244, 0.72);
  --surface-strong: rgba(244, 232, 218, 0.96);
  --glass: rgba(255, 255, 255, 0.46);
  --glass-h: rgba(255, 255, 255, 0.72);
  --border: rgba(95, 78, 63, 0.14);
  --border-strong: rgba(95, 78, 63, 0.22);
  --border-gold: rgba(232, 213, 158, 0.72);
  --gold: #e8d59e;
  --gold-dim: #cdb572;
  --gold-glow: rgba(232, 213, 158, 0.3);
  --rose: #d9bbb0;
  --taupe: #ad9c8e;
  --cyan: #ad9c8e;
  --cyan-dim: rgba(173, 156, 142, 0.16);
  --green: #7f9275;
  --green-dim: rgba(127, 146, 117, 0.14);
  --red: #b88476;
  --red-dim: rgba(184, 132, 118, 0.14);
  --yellow: #c7ab63;
  --yellow-dim: rgba(199, 171, 99, 0.14);
  --text: #241d17;
  --text-muted: #6e6056;
  --text-dim: #9b8a7d;
  --mono: 'Space Mono', monospace;
  --sans: 'Sora', sans-serif;
  --serif: 'Cormorant Garamond', serif;
  --r: 16px;
  --shadow: 0 18px 40px rgba(84, 66, 49, 0.08);
}

* { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: var(--sans) !important;
  color: var(--text) !important;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(circle at top left, rgba(232, 213, 158, 0.34) 0%, transparent 30%),
    radial-gradient(circle at 82% 12%, rgba(217, 187, 176, 0.2) 0%, transparent 25%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.2) 0%, transparent 28%);
  pointer-events: none;
  z-index: 0;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(173, 156, 142, 0.18) 0%, rgba(255, 250, 244, 0.92) 22%, rgba(247, 230, 202, 0.96) 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarNav"] { display: none !important; }

[data-testid="stPageLink"] a {
  display: flex !important;
  align-items: center !important;
  gap: 0.6rem !important;
  padding: 0.62rem 0.82rem !important;
  border-radius: 12px !important;
  color: var(--text-muted) !important;
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  text-decoration: none !important;
  transition: all 0.18s !important;
  border: 1px solid transparent !important;
  margin-bottom: 0.25rem !important;
}
[data-testid="stPageLink"] a:hover {
  background: rgba(255, 255, 255, 0.56) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
[data-testid="stPageLink"] a[aria-current="page"] {
  background: linear-gradient(135deg, rgba(232, 213, 158, 0.38) 0%, rgba(217, 187, 176, 0.22) 100%) !important;
  color: var(--text) !important;
  border-color: var(--border-gold) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45) !important;
}

.main .block-container {
  padding: 1.75rem 2.6rem 4rem !important;
  max-width: 1380px !important;
}

h1, h2, h3 {
  font-family: var(--serif) !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em !important;
  color: var(--text) !important;
}
p, li { font-family: var(--sans) !important; color: var(--text) !important; }

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
  display: none !important;
  visibility: hidden !important;
}

/* Always show sidebar collapse/expand controls */
[data-testid="stSidebarCollapsedControl"] {
  display: block !important;
  visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"] {
  display: block !important;
  visibility: visible !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.875rem !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--taupe) !important;
  box-shadow: 0 0 0 3px rgba(173, 156, 142, 0.16) !important;
  outline: none !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65) !important;
}

[data-testid="stButton"] > button,
[data-testid="stFormSubmitButton"] > button,
[data-testid="stDownloadButton"] > button {
  background: linear-gradient(135deg, #b5a292 0%, #9a887a 100%) !important;
  color: #fffaf4 !important;
  border: 1px solid rgba(126, 104, 87, 0.18) !important;
  border-radius: 12px !important;
  font-family: var(--sans) !important;
  font-weight: 700 !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.7rem 1.45rem !important;
  box-shadow: 0 10px 24px rgba(126, 104, 87, 0.16) !important;
  transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 14px 28px rgba(126, 104, 87, 0.2) !important;
}

[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 1.1rem 1.25rem !important;
  transition: all 0.2s !important;
  position: relative !important;
  overflow: hidden !important;
  box-shadow: var(--shadow) !important;
}
[data-testid="stMetric"]::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(232, 213, 158, 0.85), transparent);
}
[data-testid="stMetric"]:hover {
  background: var(--glass-h) !important;
  border-color: var(--border-gold) !important;
  transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--sans) !important;
  font-size: 0.65rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-size: 1.45rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; }

.stSuccess,
.stError,
.stWarning,
.stInfo {
  border-radius: 12px !important;
  border-width: 1px !important;
}
.stSuccess { background: var(--green-dim) !important; border-color: rgba(127, 146, 117, 0.24) !important; color: var(--green) !important; }
.stError   { background: var(--red-dim) !important; border-color: rgba(184, 132, 118, 0.26) !important; color: var(--red) !important; }
.stWarning { background: var(--yellow-dim) !important; border-color: rgba(199, 171, 99, 0.24) !important; color: var(--yellow) !important; }
.stInfo    { background: rgba(173, 156, 142, 0.12) !important; border-color: rgba(173, 156, 142, 0.24) !important; color: var(--taupe) !important; }

[data-testid="stDataFrame"],
[data-testid="stExpander"] {
  background: rgba(255, 250, 244, 0.52) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  overflow: hidden !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: rgba(173, 156, 142, 0.42); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(173, 156, 142, 0.62); }

::selection { background: rgba(232, 213, 158, 0.48); color: var(--text); }
</style>

<style>
.ee-card {
  background: linear-gradient(180deg, rgba(255, 250, 244, 0.88) 0%, rgba(244, 232, 218, 0.82) 100%);
  border: 1px solid var(--border) !important;
  border-radius: 18px;
  padding: 1.4rem 1.6rem;
  position: relative;
  overflow: hidden;
  transition: all 0.25s ease;
  box-shadow: var(--shadow);
}
.ee-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(232, 213, 158, 0.85), rgba(217, 187, 176, 0.7), transparent);
}
.ee-card:hover {
  background: linear-gradient(180deg, rgba(255, 252, 248, 0.94) 0%, rgba(244, 232, 218, 0.9) 100%);
  border-color: var(--border-strong) !important;
  box-shadow: 0 22px 44px rgba(84, 66, 49, 0.1);
  transform: translateY(-1px);
}
.ee-label {
  font-family: var(--sans);
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.ee-score {
  font-family: var(--mono);
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1;
}
.ee-score.long { color: var(--green); }
.ee-score.short { color: var(--red); }
.ee-score.neutral { color: var(--yellow); }
.ee-pill {
  display: inline-flex;
  align-items: center;
  padding: 0.3rem 0.72rem;
  border-radius: 999px;
  font-family: var(--mono);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.ee-pill.long { background: rgba(127, 146, 117, 0.12); color: var(--green); border: 1px solid rgba(127, 146, 117, 0.2); }
.ee-pill.short { background: rgba(184, 132, 118, 0.12); color: var(--red); border: 1px solid rgba(184, 132, 118, 0.2); }
.ee-pill.neutral { background: rgba(199, 171, 99, 0.12); color: var(--yellow); border: 1px solid rgba(199, 171, 99, 0.2); }
.ee-alert {
  display: inline-flex;
  align-items: center;
  padding: 0.24rem 0.62rem;
  border-radius: 999px;
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.ee-alert.GREEN { background: rgba(127, 146, 117, 0.12); color: var(--green); border: 1px solid rgba(127, 146, 117, 0.2); }
.ee-alert.YELLOW { background: rgba(199, 171, 99, 0.12); color: var(--yellow); border: 1px solid rgba(199, 171, 99, 0.2); }
.ee-alert.RED { background: rgba(184, 132, 118, 0.12); color: var(--red); border: 1px solid rgba(184, 132, 118, 0.2); }
.ee-ticker {
  font-family: var(--mono);
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--text);
  background: rgba(232, 213, 158, 0.32);
  border: 1px solid rgba(232, 213, 158, 0.7);
  padding: 0.22rem 0.58rem;
  border-radius: 999px;
}
.ee-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
.ee-dot.on { background: var(--green); box-shadow: 0 0 0 4px rgba(127, 146, 117, 0.12); }
.ee-dot.off { background: var(--red); box-shadow: 0 0 0 4px rgba(184, 132, 118, 0.12); }
.ee-comp-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.55rem 0;
  border-bottom: 1px solid rgba(95, 78, 63, 0.08);
}
.ee-comp-label {
  font-family: var(--sans);
  font-size: 0.72rem;
  color: var(--text-muted);
  width: 130px;
  flex-shrink: 0;
  line-height: 1.4;
}
.ee-comp-bar-wrap {
  flex: 1;
  height: 5px;
  background: rgba(173, 156, 142, 0.16);
  border-radius: 999px;
  overflow: hidden;
}
.ee-comp-bar {
  height: 100%;
  border-radius: 999px;
}
.ee-comp-value {
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 700;
  width: 54px;
  text-align: right;
  flex-shrink: 0;
}
.ee-contra {
  background: rgba(217, 187, 176, 0.18);
  border: 1px solid rgba(184, 132, 118, 0.22);
  border-left: 3px solid var(--red);
  border-radius: 0 12px 12px 0;
  padding: 0.9rem 1rem;
  margin: 0.4rem 0;
}
.ee-fade-in { animation: fi 0.5s ease forwards; }
@keyframes fi { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def sidebar_nav(active_page: str = "main") -> None:
    _NAV = [
        ("main",             "/",                  "🏠", "Overview"),
        ("watchlist",        "/watchlist",          "⭐", "Watchlist"),
        ("ticker",           "/ticker_analysis",   "🔍", "Ticker Analysis"),
        ("tone",             "/tone_drift",         "📊", "Tone Drift"),
        ("backtest",         "/backtest_results",  "📈", "Backtest Results"),
        ("rag",              "/rag_evaluation",    "🎯", "RAG Evaluation"),
    ]

    with st.sidebar:
        # ── Logo ─────────────────────────────────────────────────────
        st.markdown(
            '<div style="padding:1.25rem 0.5rem 1rem;border-bottom:1px solid var(--border);margin-bottom:0.75rem;">'
            '<div style="font-family:var(--serif);font-size:1.5rem;font-weight:700;letter-spacing:-0.04em;color:var(--text);">'
            '<span style="color:var(--taupe);">Earnings</span>Edge</div>'
            '<div style="font-family:var(--sans);font-size:0.6rem;color:var(--text-dim);letter-spacing:0.18em;'
            'text-transform:uppercase;margin-top:3px;">Intelligence Platform</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Nav links (plain <a> tags — works in all Streamlit versions) ──
        nav_html = '<nav style="display:flex;flex-direction:column;gap:0.15rem;">'
        for key, href, icon, label in _NAV:
            is_active = active_page == key
            bg     = "linear-gradient(135deg,rgba(232,213,158,.38),rgba(217,187,176,.22))" if is_active else "transparent"
            border = "1px solid rgba(232,213,158,.7)" if is_active else "1px solid transparent"
            color  = "var(--text)" if is_active else "var(--text-muted)"
            weight = "600" if is_active else "500"
            nav_html += (
                f'<a href="{href}" target="_self" style="'
                f'display:flex;align-items:center;gap:0.6rem;'
                f'padding:0.62rem 0.82rem;border-radius:12px;'
                f'background:{bg};border:{border};'
                f'color:{color};font-family:var(--sans);font-size:0.82rem;'
                f'font-weight:{weight};text-decoration:none;'
                f'transition:all 0.18s;">'
                f'<span>{icon}</span><span>{label}</span>'
                f'</a>'
            )
        nav_html += '</nav>'
        st.markdown(nav_html, unsafe_allow_html=True)




def score_display(score: float, signal: str) -> str:
    sl = signal.lower()
    sign = "+" if score > 0 else ""
    dot_color = {"long": "var(--green)", "short": "var(--red)", "neutral": "var(--yellow)"}.get(sl, "var(--taupe)")
    return (
        '<div class="ee-fade-in" style="text-align:center;padding:2rem 1rem;">'
        '<div class="ee-label" style="text-align:center;margin-bottom:0.75rem;">Earnings Quality Score</div>'
        f'<div class="ee-score {sl}">{sign}{score:.3f}</div>'
        '<div style="margin-top:0.75rem;display:flex;justify-content:center;align-items:center;gap:0.5rem;">'
        f'<div style="width:8px;height:8px;border-radius:50%;background:{dot_color};box-shadow:0 0 0 5px rgba(173,156,142,0.12);"></div>'
        f'<span class="ee-pill {sl}">{signal}</span>'
        '</div></div>'
    )


def component_bar(label: str, value: float, weight: float) -> str:
    pct = max(0, min(100, (value + 1) / 2 * 100))
    color = "var(--green)" if value > 0.1 else "var(--red)" if value < -0.1 else "var(--yellow)"
    sign = "+" if value >= 0 else ""
    weight_label = f"{weight:.0%} wt"
    return (
        '<div class="ee-comp-row">'
        f'<div class="ee-comp-label">{label}<br>'
        f'<span style="font-size:0.58rem;color:var(--text-dim);">{weight_label}</span></div>'
        '<div class="ee-comp-bar-wrap">'
        f'<div class="ee-comp-bar" style="width:{pct:.1f}%;background:{color};opacity:0.8;"></div>'
        '</div>'
        f'<div class="ee-comp-value" style="color:{color};">{sign}{value:.3f}</div>'
        '</div>'
    )
