<div align="center">

<img src="frontend/assets/banner.png" alt="EarningsEdge Banner" width="100%">

# EarningsEdge

**Earnings intelligence for your terminal. No Bloomberg required.**

<br>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--hop_RAG-00ADD8?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![FinBERT](https://img.shields.io/badge/FinBERT-Sentiment_NLP-FF6B35?style=flat-square)](https://huggingface.co/ProsusAI/finbert)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-6B4FBB?style=flat-square)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F26B21?style=flat-square)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

> *"Alpha is in the language before it's in the numbers."*

</div>

---

## What Is EarningsEdge?

EarningsEdge is a **single-user, local research tool** that processes SEC filings the same way a quant research desk does — except it runs on your laptop, costs nothing, and synthesizes insights in seconds.

It reads **10-K and 10-Q filings** directly from SEC EDGAR, extracts financial signals from management language using NLP, detects when executives contradict themselves across quarters, measures the gap between stated guidance and reported actuals, computes a **forensic accruals ratio** from XBRL data, and rolls everything into a single composite quality score — then answers analyst-style questions about any US-listed company using a **multi-hop LangGraph RAG pipeline**.

**Open Ticker Analysis, type a symbol, click Analyse — see the signal for yourself.**

---

## 📸 Screenshots

### 🏠 Home — Overview
<img src="frontend/assets/01_home.png" alt="EarningsEdge Home" width="100%"/>

*A descriptive landing page — pipeline overview, the four composite-score weights, and a link to Ticker Analysis. No ticker input or pipeline trigger lives here.*

> ⚠️ This screenshot predates a refactor this session that removed Home's Quick Analysis form — the page no longer looks like this. See [Known Limitations](#known-limitations).

---

### 📊 Ticker Analysis — Earnings Quality Score
<img src="frontend/assets/02_ticker_analysis.png" alt="Ticker Analysis" width="100%"/>

*Type any US-listed ticker and hit Analyse — the full pipeline (SEC fetch, embedding, NLP analysis, RAG synthesis) runs here. Composite quality score [-1, +1] with 4-component breakdown, GREEN / YELLOW / RED tone drift alert, and retrieval metadata.*

> ⚠️ This screenshot also predates this session's refactor of this page (sidebar controls moved into the main content area, added a "re-run full analysis" checkbox, changed how cached results are matched) — layout has since changed.

---

### 🔍 NLI Contradiction Detection
<img src="frontend/assets/03_contradictions.png" alt="Contradiction Detection" width="100%"/>

*DeBERTa-v3 cross-encoder flags semantic contradictions between quarters — e.g., "robust supply chain" in Q2 vs. "supply chain headwinds" in Q3.*

---

### 📈 Intelligence Report
<img src="frontend/assets/04_report.png" alt="Intelligence Report" width="100%"/>

*Full LLM-synthesised analyst report with source citations — ticker, quarter, year, and section — grounded in retrieved SEC filing context.*

---

### ⭐ Watchlist & Batch Ingestion
<img src="frontend/assets/05_watchlist.png" alt="Watchlist" width="100%"/>

*Add multiple tickers at once. Batch ingest new, refresh stale (>7 days), or re-ingest all — with per-ticker live status and a progress bar.*

---

### 📉 Tone Drift Monitor
<img src="frontend/assets/06_tone_drift.png" alt="Tone Drift" width="100%"/>

*Quarter-by-quarter FinBERT sentiment timeline. Consecutive deterioration triggers RED alert — management credibility signal before consensus cuts.*

---

### 🔁 Backtest Results
<img src="frontend/assets/07_backtest.png" alt="Backtest Results" width="100%"/>

*Composite score → long/short signal → returns. Sharpe ratio, hit rate, information coefficient (IC), alpha, and beta vs. SPY.*

> ⚠️ **Not currently wired up.** This page reads `st.session_state["quality_scores"]`, and nothing anywhere in the codebase writes that key — clicking "Run Backtest" always shows "No quality scores. Run `make analyze TICKER=X` first," even after running an analysis. The screenshot above shows the intended UI, not something you can currently reproduce. Details in [Known Limitations](#known-limitations).

---

## The Signal Engine — How the Score is Built

```
╔══════════════════════════════════════════════════════════════════════╗
║           EarningsQualityScore  ∈  [−1.0, +1.0]                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   0.30 × Sentiment Drift          (FinBERT · tone across quarters)  ║
║   0.25 × Guidance Accuracy        (stated vs. reported actuals)     ║
║   0.25 × Accruals Quality         ((NI − OCF) / Total Assets)       ║
║   0.20 × Analyst Revisions        (consensus direction · yfinance)  ║
║                                                                      ║
║   Score > +0.30  →  ▲  LONG                                         ║
║   Score < −0.30  →  ▼  SHORT                                        ║
║   Otherwise      →  —  NEUTRAL                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Why These Factors?

| Factor | The Finance Behind It |
|---|---|
| **Sentiment Drift** | Management tone deteriorates before guidance cuts, in theory. FinBERT is fine-tuned on Financial PhraseBank (financial *news* sentences) — applying it to 10-K/10-Q legalese is out-of-distribution, a real limitation of this signal that hasn't been corrected for. |
| **Guidance Accuracy** | Sandbagging (conservative guidance) predicts positive surprises. Overoptimistic guidance predicts misses. |
| **Accruals Ratio** | Sloan (1996): high accruals mean earnings aren't backed by cash. The ~10% annual alpha figure is from the academic accruals-anomaly literature generally — not a result this system has reproduced (its own measured IC is close to zero; see Known Limitations). |
| **Analyst Revisions** | Net revision direction is a proxy for informed money flow. Upgrades before earnings = smart money positioning. |

---

## Architecture

```
                     ┌──────────────────────────────────────────┐
                     │            DATA INGESTION                │
  SEC EDGAR ─────────┤  10-K · 10-Q  (XBRL + full text)         │
  Analyst Consensus ─┤  yfinance · price targets · revisions    │
                     └────────────────┬─────────────────────────┘
                                      │
                     ┌────────────────▼─────────────────────────┐
                     │          PROCESSING PIPELINE             │
                     │  Parser → Section-aware Chunker          │
                     │  MetadataTagger (section, ticker, date)  │
                     │  BGE-large-en-v1.5 → ChromaDB            │
                     └────────────────┬─────────────────────────┘
                                      │
         ┌────────────────────────────▼─────────────────────────────────┐
         │                   SIGNAL EXTRACTION                          │
         │  FinBERT Sentiment  ·  Tone Drift Detector (G/Y/R alert)     │
         │  DeBERTa NLI Contradiction Detection (cross-quarter)         │
         │  Regex NER — EPS · Revenue · CapEx · Margin guidance         │
         │  XBRL Accruals Ratio  (Net Income − OCF) / Total Assets      │
         └────────────────────────────┬─────────────────────────────────┘
                                      │
         ┌────────────────────────────▼─────────────────────────────────┐
         │             LANGGRAPH RAG PIPELINE  (9 nodes)                │
         │                                                              │
         │  query_classifier → company_retrieval → gap_detector        │
         │        │                                                     │
         │        ▼  (if peer/macro context is needed)                  │
         │  industry_retrieval → macro_retrieval                        │
         │        │                                                     │
         │        ▼                                                     │
         │  contradiction_check → synthesis → quality_check             │
         │        │  (re-retrieves via company_retrieval if ungrounded, │
         │        │   up to MAX_RAG_HOPS)                               │
         │        ▼                                                     │
         │  ragas_prep → END  (packages inputs for evaluation)          │
         └────────────────────────────┬─────────────────────────────────┘
                                      │
                     ┌────────────────▼─────────────────────────┐
                     │       GROQ / GEMINI LLM SYNTHESIS        │
                     │  llama-3.3-70b  (primary)                │
                     │  Gemini 1.5 Flash  (auto-failover)       │
                     │  Source-cited · LLM-graded for grounding │
                     └────────────────┬─────────────────────────┘
                                      │
                     ┌────────────────▼─────────────────────────┐
                     │          STREAMLIT DASHBOARD             │
                     │  Auto-ingest · Watchlist · Backtest      │
                     └──────────────────────────────────────────┘
```

*RAGAS/MLflow evaluation is a separate, manual step from the RAG Evaluation dashboard page — not an automatic part of the graph above; `ragas_prep` only packages the inputs it would need.*

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM Primary** | Groq `llama-3.3-70b-versatile` | Sub-second inference, free tier |
| **LLM Fallback** | Google Gemini 1.5 Flash | Auto-switches on rate limit |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | MTEB top-tier, 1024-dim, financial text |
| **Sentiment NLP** | `ProsusAI/finbert` | Fine-tuned on financial *news* (Financial PhraseBank) — not SEC filings; out-of-distribution on 10-K/10-Q text |
| **Contradiction** | `cross-encoder/nli-deberta-v3-base` | NLI scoring for cross-quarter consistency |
| **NER** | Regex rules (`ner_extractor.extract_guidance`) | Guidance extraction (EPS, revenue, capex) |
| **Orchestration** | LangGraph (9-node state machine) | Multi-hop retrieval with conditional routing |
| **Vector Store** | ChromaDB (local persistent) | Rich metadata filtering by section, quarter |
| **Validation** | Pydantic v2 (17 models + 8 enums) | Typed pipeline, zero unvalidated dicts |
| **Market Data** | yfinance | Price history, analyst consensus, revisions |
| **Backtesting** | pandas (trade-by-trade simulation) | Sharpe, IC, alpha, beta, hit rate |
| **RAG Evaluation** | LLM-as-judge fallback (native RAGAS only if `OPENAI_API_KEY` is set) | Faithfulness, answer relevance |
| **Experiment Tracking** | MLflow | Latency, grounding score, retrieval metrics |
| **Dashboard** | Streamlit + Plotly | Zero-click ingestion, live progress display |
| **Package Manager** | uv | 10–100× faster than pip |

---

## Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 1. Clone & Install

```bash
git clone https://github.com/aryadoshii/EarningsEdge.git
cd EarningsEdge
make setup
```

### 2. Configure API Keys

```bash
cp config/.env.example .env
```

Edit `.env` — all free tier:

```env
GROQ_API_KEY=your_key_here        # console.groq.com
GOOGLE_API_KEY=your_key_here      # aistudio.google.com
SEC_USER_AGENT=Name email@domain  # required by SEC fair-use policy
```

### 3. Launch

```bash
make run
# → http://localhost:8501
```

**That's it.** Open the Ticker Analysis page, type any US-listed ticker, click Analyse — the full pipeline runs automatically in the UI.

---

## Make Commands

```bash
make setup              # install deps + download spaCy transformer model (currently unused — NER runs on regex only)
make run                # launch Streamlit at localhost:8501
make ingest  TICKER=X   # ingest SEC filings + analyst data
make embed   TICKER=X   # embed chunks into ChromaDB
make analyze TICKER=X   # sentiment + drift + contradictions + scoring
make test               # run tests/test_rag_logic.py (module-level asserts, not pytest test functions)
make lint               # ruff + mypy
make mlflow             # MLflow UI at localhost:5000
make clean              # wipe processed data + ChromaDB
```

---

## Project Structure

```
earningsedge/
├── config/
│   ├── settings.py              # Pydantic-settings — all env vars + score weights
│   └── .env.example
├── src/
│   ├── ingestion/
│   │   ├── sec_fetcher.py       # EDGAR REST API — supports 10-K/10-Q/8-K + XBRL, but pipeline_runner only ever requests 10-K/10-Q
│   │   ├── transcript_fetcher.py# Earnings call transcript parser (not wired into the ingest pipeline — zero callers outside its own __main__)
│   │   ├── analyst_fetcher.py   # yfinance — consensus, targets, revision direction
│   │   └── data_validator.py    # 17 Pydantic v2 models + 8 enums — typed end-to-end
│   ├── processing/
│   │   ├── chunker.py           # 4 strategies implemented; only section_aware runs in the live pipeline (10-K/10-Q only — semantic needs 8-K, speaker_turn needs transcripts)
│   │   ├── document_parser.py   # Section extraction (MDA, Risk Factors, Guidance…)
│   │   ├── ner_extractor.py     # Regex guidance extraction (extract_guidance); spaCy entity extraction (extract_entities) has zero call sites
│   │   └── metadata_tagger.py   # Enriches chunks with financial metadata
│   ├── embeddings/
│   │   ├── embedder.py          # BGE-large-en-v1.5 + ChromaDB ingestion
│   │   └── retriever.py         # Metadata-filtered vector search
│   ├── analysis/
│   │   ├── sentiment_analyzer.py    # FinBERT scoring + quarter-level aggregation
│   │   ├── tone_drift_detector.py   # Cross-quarter drift + G/Y/R alert system
│   │   ├── contradiction_detector.py# DeBERTa NLI cross-quarter semantic conflicts
│   │   ├── guidance_accuracy.py     # Stated guidance → reported actuals matching
│   │   └── earnings_quality_scorer.py # Composite [-1,+1] score computation
│   ├── rag/
│   │   ├── graph.py             # LangGraph 9-node state machine
│   │   ├── multi_hop_chain.py   # MultiHopChain — main RAG entrypoint
│   │   ├── nodes.py             # All node implementations
│   │   ├── prompts.py           # Structured financial analysis prompts
│   │   └── llm_client.py        # Groq primary + Gemini fallback with auto-switch
│   ├── backtest/
│   │   ├── signal_generator.py  # Composite score → long/short signal
│   │   ├── backtester.py        # pandas trade-by-trade engine
│   │   └── metrics.py           # Sharpe, IC, alpha, beta, hit rate
│   └── evaluation/
│       ├── ragas_evaluator.py   # Faithfulness + answer relevance (real LLM scores); context_precision/recall are placeholders unless OPENAI_API_KEY enables native RAGAS
│       └── mlflow_tracker.py    # Experiment logging
├── app/
│   ├── main.py                  # st.navigation shell over 6 pages — no inputs or pipeline triggers of its own
│   ├── pages/
│   │   ├── 00_watchlist.py      # Batch ingestion + data freshness manager
│   │   ├── 01_ticker_analysis.py# Auto-ingest + RAG query + score display
│   │   ├── 02_tone_drift.py     # Sentiment timeline + alert dashboard
│   │   ├── 03_backtest_results.py
│   │   └── 04_rag_evaluation.py
│   └── components/
│       └── theme.py             # Obsidian Terminal design system
├── frontend/
│   └── assets/                  # banner.png + UI screenshots
├── src/pipeline_runner.py       # Programmatic API — ingest / embed / analyze
├── data/
│   ├── raw/                     # SEC filing HTML/XML
│   ├── processed/{TICKER}/      # chunks.json, analysis.json, filings.json
│   └── chroma_db/               # Persistent ChromaDB vector store
├── pyproject.toml
└── Makefile
```

---

## The Accruals Signal — Finance Deep Dive

The accruals anomaly is one of the most replicated findings in empirical asset pricing.

**Sloan (1996):** *"Do Stock Prices Fully Reflect Information in Accruals and Cash Flows about Future Earnings?"* — Journal of Accounting Research

```
Accruals Ratio = (Net Income − Operating Cash Flow) / Total Assets
```

When this ratio is **high**, earnings greatly exceed cash generation — a sign of aggressive revenue recognition or expense deferral. These inflated earnings mean-revert. A hedge portfolio long low-accrual and short high-accrual companies historically generates ~10% annual alpha with low market beta — **this figure is from the academic accruals-anomaly literature, not from backtesting this system**, whose own measured Information Coefficient is close to zero (see Known Limitations below).

EarningsEdge computes this from **XBRL-tagged financial data** in SEC filings — the same structured dataset used by institutional data vendors like FactSet and Bloomberg — and weights it 25% in the composite score.

---

## Supported Tickers

Works with any **US-listed company** that files with the SEC:

```
Tech & Growth    AAPL  MSFT  NVDA  GOOGL  META  AMZN  TSLA
Finance          JPM   GS    MS    BAC    BLK   BX    C
Healthcare       JNJ   PFE   UNH   ABBV   LLY
Energy           XOM   CVX   OXY   COP
Consumer         NKE   COST  MCD   SBUX   TGT
```

> Companies listed exclusively on non-US exchanges (LSE, TSX, Euronext) do not file with SEC EDGAR. Foreign private issuers using Form 20-F may have limited data coverage.

---

## Known Limitations

Validating this system end-to-end surfaced several honest findings worth stating directly rather than glossing over:

- **Realised Information Coefficient is close to zero.** The Spearman rank correlation between the composite score and forward returns comes out near 0 in backtesting — as currently weighted and computed, the signal has not demonstrated measurable predictive power.
- **Two of four components are look-ahead contaminated.** `sentiment_drift_component` and `analyst_revision_component` — 50% of the composite weight — are each computed once from present-day data (the full historical drift trend, and a 30-day analyst-revision snapshot) and reused unchanged for every historical quarter of a ticker's series. They're constant across time and leak future information into the past, which is almost certainly why the measured IC is flat.
- **Backtest entry dates are heuristic, not real.** Earnings announcement dates are estimated as the 15th of a fixed month per fiscal quarter rather than pulled from an actual corporate calendar — entry timing can be off by weeks.
- **No transaction costs, slippage, or borrow costs are modelled.** Reported Sharpe/return figures are gross, frictionless numbers.
- **The Backtest Results page is not currently functional** — see the note under the Backtest Results screenshot above.

None of this invalidates the underlying signal engineering — FinBERT-based tone drift, NLI contradiction detection, guidance-accuracy tracking, and XBRL-based accruals are each independently real and interesting. But the system has not yet demonstrated that its composite score predicts returns, and the honest next step is fixing the look-ahead leakage in two of its four components before the IC number means anything either way.

---

## Free API Keys

| Service | Required | Link | Cost |
|---|---|---|---|
| Groq | Yes | [console.groq.com](https://console.groq.com) | Free tier |
| Google AI Studio | Yes | [aistudio.google.com](https://aistudio.google.com) | Free tier |
| SEC EDGAR | No | Set `SEC_USER_AGENT` in `.env` | Free (fair use) |
| yfinance | No | — | Free |

---

<div align="center">

**Built with conviction. Priced at zero.**

*EarningsEdge · SEC EDGAR · FinBERT · LangGraph · ChromaDB · Groq*

</div>
