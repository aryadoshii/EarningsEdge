"""
ChromaDB vector store wrapper.

Provides a clean interface for upsert, query, and delete operations
against a persistent local ChromaDB instance.

Usage:
    from src.embeddings.vector_store import vector_store
    vector_store.upsert(ids, embeddings, documents, metadatas)
    results = vector_store.query(query_embedding, n_results=10)
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.settings import settings

# Lazy ChromaDB client + collection
_client: Any = None
_collection: Any = None
COLLECTION_NAME = "earningsedge"


def _get_collection() -> Any:
    """Lazy-initialise ChromaDB client and collection."""
    global _client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb  # type: ignore
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_path))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{COLLECTION_NAME}' ready "
            f"({_collection.count()} existing chunks)"
        )
    except Exception as exc:
        logger.error(f"ChromaDB init failed: {exc}")
    return _collection


class VectorStore:
    """ChromaDB CRUD wrapper for EarningsEdge chunks."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """
        Upsert chunks into ChromaDB (insert or update by ID).

        Args:
            ids:         Unique chunk IDs.
            embeddings:  Dense vectors (list of floats per chunk).
            documents:   Raw text of each chunk.
            metadatas:   Metadata dicts for filtering.

        Returns:
            Number of chunks upserted.
        """
        col = _get_collection()
        if col is None:
            logger.error("ChromaDB collection unavailable — skipping upsert")
            return 0

        # ChromaDB has a max batch size of 5461 — chunk if needed
        batch_size = 500
        total = 0
        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            try:
                col.upsert(
                    ids=ids[sl],
                    embeddings=embeddings[sl],
                    documents=documents[sl],
                    metadatas=metadatas[sl],
                )
                total += len(ids[sl])
            except Exception as exc:
                logger.error(f"ChromaDB upsert batch {i // batch_size} failed: {exc}")

        logger.debug(f"Upserted {total} chunks into ChromaDB")
        return total

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        n_results: int = settings.TOP_K_RETRIEVAL,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query ChromaDB for the nearest neighbours to a query vector.

        Args:
            query_embedding: Dense query vector (must match stored dim).
            n_results:       Number of results to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            List of result dicts with keys: id, text, metadata, distance.
        """
        col = _get_collection()
        if col is None:
            return []

        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(n_results, col.count() or 1),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            raw = col.query(**kwargs)

            results: list[dict[str, Any]] = []
            ids       = raw.get("ids", [[]])[0]
            docs      = raw.get("documents", [[]])[0]
            metas     = raw.get("metadatas", [[]])[0]
            distances = raw.get("distances", [[]])[0]

            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                results.append({
                    "chunk_id": cid,
                    "text":     doc,
                    "metadata": meta,
                    "distance": dist,
                    # Flatten metadata fields for convenience
                    **meta,
                })
            return results

        except Exception as exc:
            logger.error(f"ChromaDB query failed: {exc}")
            return []

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Retrieve specific chunks by their IDs."""
        col = _get_collection()
        if col is None:
            return []
        try:
            raw = col.get(ids=ids, include=["documents", "metadatas"])
            results = []
            for cid, doc, meta in zip(
                raw.get("ids", []),
                raw.get("documents", []),
                raw.get("metadatas", []),
            ):
                results.append({"chunk_id": cid, "text": doc, **meta})
            return results
        except Exception as exc:
            logger.error(f"ChromaDB get failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_ticker(self, ticker: str) -> int:
        """Delete all chunks for a given ticker."""
        col = _get_collection()
        if col is None:
            return 0
        try:
            col.delete(where={"ticker": ticker.upper()})
            logger.info(f"Deleted all chunks for {ticker}")
            return 1
        except Exception as exc:
            logger.error(f"Delete failed for {ticker}: {exc}")
            return 0

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def collection_info(self) -> dict[str, Any]:
        """Return basic stats about the collection."""
        col = _get_collection()
        if col is None:
            return {"status": "unavailable"}
        return {
            "name":  COLLECTION_NAME,
            "count": col.count(),
            "path":  str(settings.chroma_path),
        }

    def count(self) -> int:
        """Return total number of chunks stored."""
        col = _get_collection()
        return col.count() if col else 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

vector_store = VectorStore()