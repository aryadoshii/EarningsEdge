"""
Embedding module for EarningsEdge.

Encodes DocumentChunk text into dense vectors using BAAI/bge-large-en-v1.5.
Chunks are processed in batches and stored directly into ChromaDB via
the vector_store module.

Usage:
    python -m src.embeddings.embedder --ticker AAPL
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import numpy as np
from loguru import logger

from config.settings import settings
from src.ingestion.data_validator import DocumentChunk

# Lazy model cache
_model: Any = None


def _get_model() -> Any:
    """Lazy-load the sentence-transformer embedding model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            _model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.success("Embedding model loaded ✓")
        except Exception as exc:
            logger.error(f"Failed to load embedding model: {exc}")
    return _model


class Embedder:
    """
    Encodes DocumentChunk objects into dense vectors and stores in ChromaDB.

    Attributes:
        batch_size: Number of chunks per encoding batch.
    """

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into normalised dense vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            Float32 numpy array of shape (len(texts), embedding_dim).
        """
        model = _get_model()
        if model is None:
            raise RuntimeError("Embedding model not available")

        all_vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_vectors.append(vecs)
            logger.debug(
                f"Embedded batch {i // self.batch_size + 1}/"
                f"{(len(texts) - 1) // self.batch_size + 1}"
            )

        return np.vstack(all_vectors).astype(np.float32)

    def embed_and_store(self, chunks: list[DocumentChunk]) -> int:
        """
        Embed chunks and upsert them into ChromaDB.

        Args:
            chunks: List of DocumentChunk objects to embed and store.

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks:
            logger.warning("embed_and_store called with empty chunk list")
            return 0

        from src.embeddings.vector_store import vector_store

        texts = [c.text for c in chunks]
        ids   = [c.chunk_id for c in chunks]
        metas = [c.to_chroma_metadata() for c in chunks]

        logger.info(f"Embedding {len(chunks)} chunks...")
        vectors = self.embed_texts(texts)

        stored = vector_store.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=metas,
        )
        logger.success(f"Stored {stored} chunks in ChromaDB")
        return stored

    def embed_ticker(self, ticker: str) -> int:
        """
        Load all processed chunks for a ticker and embed them into ChromaDB.

        Reads chunks saved by the processing pipeline from the processed
        data directory. Falls back to re-processing from raw filings if
        no processed chunks are found.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Number of chunks embedded.
        """
        import json
        from pathlib import Path

        ticker = ticker.upper()
        processed_dir = settings.processed_data_path / ticker

        if not processed_dir.exists():
            logger.warning(
                f"[{ticker}] No processed data at {processed_dir}. "
                "Run `make ingest` first."
            )
            return 0

        chunk_files = list(processed_dir.glob("chunks_*.json"))
        if not chunk_files:
            logger.warning(f"[{ticker}] No chunk files found in {processed_dir}")
            return 0

        all_chunks: list[DocumentChunk] = []
        for f in chunk_files:
            try:
                raw = json.loads(f.read_text())
                for item in raw:
                    try:
                        all_chunks.append(DocumentChunk(**item))
                    except Exception as exc:
                        logger.debug(f"Skipping malformed chunk: {exc}")
            except Exception as exc:
                logger.warning(f"Failed to load {f}: {exc}")

        if not all_chunks:
            logger.warning(f"[{ticker}] No valid chunks loaded")
            return 0

        logger.info(f"[{ticker}] Loaded {len(all_chunks)} chunks for embedding")
        return self.embed_and_store(all_chunks)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

embedder = Embedder()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to embed")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    logger.info(f"Starting embedding pipeline for {ticker}")

    # Quick connectivity check
    try:
        from src.embeddings.vector_store import vector_store
        info = vector_store.collection_info()
        logger.info(f"ChromaDB collection: {info}")
    except Exception as exc:
        logger.error(f"ChromaDB not accessible: {exc}")

    count = embedder.embed_ticker(ticker)

    if count > 0:
        logger.success(f"[{ticker}] Embedded {count} chunks ✓")
    else:
        logger.warning(
            f"[{ticker}] No chunks embedded. "
            "Ensure `make ingest` has been run first and processed data exists."
        )
        # Show what's in the processed directory
        processed_dir = settings.processed_data_path / ticker
        logger.info(f"Looking for data in: {processed_dir}")
        if processed_dir.exists():
            files = list(processed_dir.iterdir())
            logger.info(f"Found {len(files)} files: {[f.name for f in files[:5]]}")
        else:
            logger.info("Directory does not exist yet")