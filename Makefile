# EarningsEdge — Makefile
# Usage: make <target> [TICKER=AAPL] [START=2020-01-01] [END=2024-12-31]

TICKER  ?= AAPL
PYTHON  := uv run python
PYTEST  := uv run pytest

.PHONY: help setup ingest embed analyze run test lint mlflow clean fmt

# ── Default ─────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  EarningsEdge Pipeline Commands"
	@echo "  ────────────────────────────────────────"
	@echo "  make setup              Install all dependencies via uv"
	@echo "  make ingest TICKER=X    Download + process SEC filings & transcripts"
	@echo "  make embed  TICKER=X    Embed chunks → ChromaDB"
	@echo "  make analyze TICKER=X   Run full analysis (sentiment, drift, RAG)"
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
# Note: this used to shell out to sec_fetcher/transcript_fetcher/analyst_fetcher/
# document_parser/chunker/ner_extractor/metadata_tagger as separate --ticker
# invocations. Most of those __main__ blocks have no argparse (they just run
# their own hardcoded smoke tests) and never wrote to data/processed/, so
# `make ingest` silently did nothing. pipeline_runner.py is the actual,
# working ingest path — it's what `make embed` already used under the hood.
ingest:
	@echo "→ Ingesting data for $(TICKER)"
	$(PYTHON) -m src.pipeline_runner --ticker $(TICKER) --ingest

# ── Embedding ───────────────────────────────────────────────────────────────
embed:
	@echo "→ Embedding chunks for $(TICKER) → ChromaDB"
	$(PYTHON) -m src.pipeline_runner --ticker $(TICKER) --ingest --embed

# ── Analysis ────────────────────────────────────────────────────────────────
analyze:
	@echo "→ Running analysis pipeline for $(TICKER)"
	$(PYTHON) -m src.pipeline_runner --ticker $(TICKER) --analyze

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