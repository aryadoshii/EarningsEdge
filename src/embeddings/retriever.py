"""
ChromaDB retriever for the EarningsEdge RAG pipeline.

Provides async retrieval of DocumentChunk objects from ChromaDB using
dense vector similarity search with optional metadata filtering.

The retriever embeds the query string on-the-fly using the same model
used to embed the stored chunks (BAAI/bge-large-en-v1.5), ensuring
vector space alignment.

Usage:
    from src.embeddings.retriever import retriever
    chunks = await retriever.retrieve(
        query="What is AAPL revenue guidance for Q4?",
        metadata_filter={"ticker": "AAPL", "section_type": "guidance"},
        top_k=10,
    )
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from config.settings import settings


class Retriever:
    """
    Async retriever that queries ChromaDB with dense embeddings.

    Embeds the query string and returns the top-k most similar chunks,
    with optional ChromaDB metadata filtering.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        top_k: int = settings.TOP_K_RETRIEVAL,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k chunks most similar to the query string.

        Runs the embedding in a thread pool to avoid blocking the event
        loop (sentence-transformers encode() is synchronous).

        Args:
            query:           Natural language query string.
            metadata_filter: Optional ChromaDB where-clause filter dict.
                             e.g. {"ticker": "AAPL", "section_type": "guidance"}
            top_k:           Number of chunks to return.

        Returns:
            List of result dicts with keys: chunk_id, text, metadata,
            distance, plus all flattened metadata fields (ticker, quarter,
            year, section_type, filing_type, etc.)
        """
        loop = asyncio.get_event_loop()

        # Embed query in thread pool
        try:
            query_vector: list[float] = await loop.run_in_executor(
                None, self._embed_query, query
            )
        except Exception as exc:
            logger.error(f"Query embedding failed: {exc}")
            return []

        # Query ChromaDB
        try:
            from src.embeddings.vector_store import vector_store

            # Build clean filter — remove None values and empty operators
            clean_filter = self._build_filter(metadata_filter)

            results = vector_store.query(
                query_embedding=query_vector,
                n_results=top_k,
                where=clean_filter,
            )
            logger.debug(
                f"Retrieved {len(results)} chunks "
                f"(query='{query[:50]}', filter={clean_filter})"
            )
            return results

        except Exception as exc:
            logger.error(f"ChromaDB retrieval failed: {exc}")
            return []

    async def retrieve_by_ticker(
        self,
        ticker: str,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
    ) -> list[dict[str, Any]]:
        """
        Convenience wrapper — retrieve chunks for a specific ticker only.

        Args:
            ticker: Stock ticker (case-insensitive).
            query:  Natural language query.
            top_k:  Number of chunks to return.

        Returns:
            List of result dicts filtered to the given ticker.
        """
        return await self.retrieve(
            query=query,
            metadata_filter={"ticker": ticker.upper()},
            top_k=top_k,
        )

    async def retrieve_by_section(
        self,
        ticker: str,
        section_type: str,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
    ) -> list[dict[str, Any]]:
        """
        Retrieve chunks filtered to a specific ticker and section type.

        Args:
            ticker:       Stock ticker.
            section_type: Section name e.g. "guidance", "risk_factors", "mda".
            query:        Natural language query.
            top_k:        Number of chunks to return.

        Returns:
            List of result dicts.
        """
        return await self.retrieve(
            query=query,
            metadata_filter={
                "ticker": ticker.upper(),
                "section_type": section_type,
            },
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_query(query: str) -> list[float]:
        """
        Embed a single query string using the configured embedding model.

        Runs synchronously — always call via run_in_executor.

        Args:
            query: Query text.

        Returns:
            Normalised dense vector as a Python list of floats.
        """
        from src.embeddings.embedder import _get_model
        model = _get_model()
        if model is None:
            raise RuntimeError("Embedding model not available")
        vec = model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()

    @staticmethod
    def _build_filter(
        raw_filter: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Convert a simple key→value filter into a ChromaDB-compatible where clause.

        ChromaDB requires the $eq operator for scalar comparisons and
        $in for list comparisons.

        Examples:
            {"ticker": "AAPL"}
            → {"ticker": {"$eq": "AAPL"}}

            {"ticker": "AAPL", "section_type": {"$in": ["guidance","mda"]}}
            → passed through as-is

        Args:
            raw_filter: Simple filter dict or None.

        Returns:
            ChromaDB where-clause dict or None.
        """
        if not raw_filter or not settings.METADATA_FILTER_ENABLED:
            return None

        where: dict[str, Any] = {}
        conditions: list[dict[str, Any]] = []

        for key, value in raw_filter.items():
            if value is None:
                continue
            # Already a ChromaDB operator dict — pass through
            if isinstance(value, dict):
                conditions.append({key: value})
            # List value → $in operator
            elif isinstance(value, list):
                conditions.append({key: {"$in": [str(v) for v in value]}})
            # Scalar value → $eq operator
            else:
                conditions.append({key: {"$eq": str(value)}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        # Multiple conditions → $and
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

retriever = Retriever()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio

    async def _demo() -> None:
        from src.embeddings.vector_store import vector_store

        info = vector_store.collection_info()
        print(f"ChromaDB: {info}")

        count = info.get("count", 0)
        if count == 0:
            print("Collection is empty — run `make ingest TICKER=AAPL && make embed TICKER=AAPL` first")
            return

        results = await retriever.retrieve(
            query="revenue guidance next quarter",
            metadata_filter={"ticker": "AAPL"},
            top_k=3,
        )
        print(f"\nRetrieved {len(results)} chunks for AAPL:")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r.get('section_type','?')} | {r.get('quarter','?')} {r.get('year','?')}")
            print(f"       {r.get('text','')[:120]}...")
            print(f"       distance={r.get('distance', 0):.4f}")

    _asyncio.run(_demo())