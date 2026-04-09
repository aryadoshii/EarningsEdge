# EarningsEdge — Makefile
# Usage: make <target> [TICKER=AAPL] [START=2020-01-01] [END=2024-12-31]

TICKER  ?= AAPL
PYTHON  := uv run python
PYTEST  := uv run pytest

.PHONY: help setup ingest embed analyze backtest run test lint mlflow clean fmt

# ── Default ─────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  EarningsEdge Pipeline Commands"
	@echo "  ────────────────────────────────────────"
	@echo "  make setup              Install all dependencies via uv"
	@echo "  make ingest TICKER=X    Download + process SEC filings & transcripts"
	@echo "  make embed  TICKER=X    Embed chunks → ChromaDB"
	@echo "  make analyze TICKER=X   Run full analysis (sentiment, drift, RAG)"
	@echo "  make backtest           Run backtesting engine on all tickers"
	@echo "  make run                Launch Streamlit dashboard"
	@echo "  make test               Run pytest suite"
	@echo "  make lint               Run ruff + mypy"
	@echo "  make fmt                Auto-fix formatting with ruff"
	@echo "  make mlflow             Open MLflow experiment UI"
	@echo "  make clean              Wipe data/processed and data/chroma_db"
	@echo ""

# ── Setup ───────────────────────────────────────────────────────────────────
setup:
	uv sync
	uv pip install pip
	uv run python -m spacy download en_core_web_trf
	@echo "Setup complete. Copy config/.env.example → .env and fill in API keys."

# ── Data ingestion ──────────────────────────────────────────────────────────
ingest:
	@echo "→ Ingesting data for $(TICKER)"
	$(PYTHON) -m src.ingestion.sec_fetcher --ticker $(TICKER)
	$(PYTHON) -m src.ingestion.transcript_fetcher --ticker $(TICKER)
	$(PYTHON) -m src.ingestion.analyst_fetcher --ticker $(TICKER)
	@echo "→ Processing documents"
	$(PYTHON) -m src.processing.document_parser --ticker $(TICKER)
	$(PYTHON) -m src.processing.chunker --ticker $(TICKER)
	$(PYTHON) -m src.processing.ner_extractor --ticker $(TICKER)
	$(PYTHON) -m src.processing.metadata_tagger --ticker $(TICKER)

# ── Embedding ───────────────────────────────────────────────────────────────
embed:
	@echo "→ Embedding chunks for $(TICKER) → ChromaDB"
	$(PYTHON) -m src.pipeline_runner --ticker $(TICKER) --ingest --embed

# ── Analysis ────────────────────────────────────────────────────────────────
analyze:
	@echo "→ Running analysis pipeline for $(TICKER)"
	$(PYTHON) -m src.pipeline_runner --ticker $(TICKER) --analyze

# ── Backtest ─────────────────────────────────────────────────────────────────
backtest:
	@echo "→ Running backtesting engine"
	$(PYTHON) -m src.backtest.backtester

# ── Streamlit ────────────────────────────────────────────────────────────────
run:
	uv run streamlit run app/main.py --server.port 8501

# ── Testing ──────────────────────────────────────────────────────────────────
test:
	$(PYTEST) --tb=short -q

test-v:
	$(PYTEST) --tb=long -v

# ── Code quality ─────────────────────────────────────────────────────────────
lint:
	uv run ruff check src/ app/ tests/ config/
	uv run mypy src/ app/ config/ --ignore-missing-imports

fmt:
	uv run ruff check --fix src/ app/ tests/ config/
	uv run ruff format src/ app/ tests/ config/

# ── MLflow ───────────────────────────────────────────────────────────────────
mlflow:
	uv run mlflow ui --backend-store-uri data/mlflow_runs --port 5000

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "Wiping processed data and ChromaDB..."
	rm -rf data/processed/*
	rm -rf data/chroma_db/*
	rm -rf data/.cache/*
	@echo "Clean complete."

clean-all: clean
	rm -rf data/raw/sec_filings/*
	rm -rf data/raw/transcripts/*
	rm -rf data/raw/analyst_data/*
	rm -rf data/mlflow_runs/*
	@echo "Full clean complete (raw data removed too)."