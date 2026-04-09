"""
Disk-based response cache using diskcache.

Prevents redundant API calls and web scrapes by caching serialised responses
to disk with a configurable TTL (default 24 hours).  Cache keys are derived
from the function name and its arguments, making it transparent to callers.

Provides:
    DiskCache   — low-level key/value cache class
    cached      — async function decorator

Usage:
    cache = DiskCache()

    # Decorator form:
    @cached(ttl=3600)
    async def fetch_filing(ticker: str, form: str) -> dict:
        ...

    # Direct form:
    key = cache.make_key("sec", ticker, form, date)
    result = cache.get(key)
    if result is None:
        result = await fetch(...)
        cache.set(key, result, ttl=86400)
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

import diskcache
from loguru import logger

from config.settings import settings

P = ParamSpec("P")
T = TypeVar("T")

# Default TTL: 24 hours
_DEFAULT_TTL = getattr(settings, "CACHE_TTL_SECONDS", 86_400)


class DiskCache:
    """
    Thin wrapper around diskcache.Cache with JSON serialisation.

    All values are JSON-serialised before storage and deserialised on
    retrieval, ensuring compatibility with Pydantic models (via .model_dump())
    and plain Python dicts/lists.

    Attributes:
        _cache: Underlying diskcache.Cache instance.
    """

    def __init__(self, directory: str | None = None) -> None:
        cache_dir = directory or str(settings.cache_path)
        self._cache: diskcache.Cache = diskcache.Cache(
            directory=cache_dir,
            size_limit=2 * 1024 ** 3,  # 2 GB max cache size
        )
        logger.debug(f"[Cache] Initialised disk cache at {cache_dir}")

    def get(self, key: str) -> Any | None:
        """
        Retrieve a cached value by key.

        Args:
            key: Cache key string.

        Returns:
            Deserialised value, or None if key is missing or expired.
        """
        try:
            raw = self._cache.get(key, default=None)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.debug(f"[Cache] Get failed for key '{key}': {exc}")
            return None

    def set(self, key: str, value: Any, ttl: int = _DEFAULT_TTL) -> bool:
        """
        Store a value under the given key with an expiry TTL.

        Args:
            key:   Cache key string.
            value: JSON-serialisable value.
            ttl:   Time-to-live in seconds (default 86400 = 24 hours).

        Returns:
            True if stored successfully, False otherwise.
        """
        try:
            serialised = json.dumps(value, default=str)
            self._cache.set(key, serialised, expire=ttl)
            logger.debug(f"[Cache] Stored key '{key}' (ttl={ttl}s)")
            return True
        except Exception as exc:
            logger.warning(f"[Cache] Set failed for key '{key}': {exc}")
            return False

    def delete(self, key: str) -> bool:
        """
        Evict a single key from the cache.

        Args:
            key: Cache key to evict.

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        try:
            return bool(self._cache.delete(key))
        except Exception:
            return False

    def clear(self) -> None:
        """Evict all entries from the cache."""
        self._cache.clear()
        logger.info("[Cache] Cache cleared")

    def make_key(self, *parts: Any) -> str:
        """
        Build a deterministic cache key from an arbitrary set of parts.

        Parts are JSON-serialised and MD5-hashed to produce a short,
        filesystem-safe key.

        Args:
            *parts: Any JSON-serialisable values.

        Returns:
            32-character hex MD5 digest string.
        """
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        """Flush and close the underlying cache."""
        self._cache.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_cache: DiskCache | None = None


def get_cache() -> DiskCache:
    """Return the module-level singleton DiskCache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = DiskCache()
    return _default_cache


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def cached(
    ttl: int = _DEFAULT_TTL,
    prefix: str = "",
    skip_none: bool = True,
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """
    Async function decorator that caches return values to disk.

    The cache key is built from the function's qualified name + all positional
    and keyword arguments.  Keyword arguments are sorted for consistency.

    Args:
        ttl:       Time-to-live in seconds.
        prefix:    Optional string prefix added to the cache key.
        skip_none: If True, None return values are not cached (re-fetched next time).

    Returns:
        Decorator that transparently adds disk caching to the async function.

    Example:
        @cached(ttl=3600, prefix="sec")
        async def fetch_filing(ticker: str, form: str) -> dict:
            ...
    """
    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache = get_cache()
            key_parts = [prefix or func.__qualname__, args, sorted(kwargs.items())]
            key = cache.make_key(*key_parts)

            # Attempt cache hit
            cached_value = cache.get(key)
            if cached_value is not None:
                logger.debug(f"[Cache] HIT  {func.__name__}({key[:8]}…)")
                return cached_value  # type: ignore[return-value]

            logger.debug(f"[Cache] MISS {func.__name__}({key[:8]}…)")
            result = await func(*args, **kwargs)

            if result is not None or not skip_none:
                cache.set(key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio
    import tempfile, os

    async def _demo() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            c = DiskCache(directory=tmpdir)

            # Basic set/get
            key = c.make_key("test", "AAPL", "10-K")
            c.set(key, {"ticker": "AAPL", "revenue": 394300}, ttl=60)
            result = c.get(key)
            print(f"GET {key[:8]}…: {result}")
            assert result == {"ticker": "AAPL", "revenue": 394300}

            # Missing key
            assert c.get("nonexistent_key") is None

            # Decorator
            call_count = 0

            @cached(ttl=60, prefix="demo")
            async def expensive_call(ticker: str) -> str:
                nonlocal call_count
                call_count += 1
                return f"result_for_{ticker}"

            r1 = await expensive_call("MSFT")
            r2 = await expensive_call("MSFT")  # should hit cache
            print(f"Call count: {call_count} (expected 1), r1={r1}, r2={r2}")
            assert call_count == 1

            print("Cache smoke test passed ✓")

    _asyncio.run(_demo())
