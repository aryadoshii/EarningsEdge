"""
Async token-bucket rate limiter.

Provides the RateLimiter class used throughout the ingestion layer to enforce
per-service request rate limits (e.g. SEC EDGAR: ≤10 req/s).

Also provides the @with_retry decorator for wrapping coroutines with
exponential backoff on transient failures.

Usage:
    limiter = RateLimiter(max_calls=8.0, period=1.0)
    await limiter.acquire()
    response = await client.get(url)

    @with_retry(max_attempts=3, base_delay=1.0)
    async def fetch_data(url: str) -> dict:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
T = TypeVar("T")


class RateLimiter:
    """
    Async token-bucket rate limiter.

    Allows at most `max_calls` calls per `period` seconds.  Callers that
    exceed the budget are suspended (not rejected) until a token is available.

    Attributes:
        max_calls: Maximum calls allowed per period.
        period:    Time window in seconds.

    Example:
        limiter = RateLimiter(max_calls=10, period=1.0)
        async with aiohttp.ClientSession() as session:
            for url in urls:
                await limiter.acquire()
                response = await session.get(url)
    """

    def __init__(self, max_calls: float, period: float = 1.0) -> None:
        if max_calls <= 0:
            raise ValueError(f"max_calls must be positive, got {max_calls}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")

        self.max_calls = max_calls
        self.period = period
        self._lock = asyncio.Lock()
        self._call_times: list[float] = []

    async def acquire(self) -> None:
        """
        Wait until a token is available, then consume it.

        Suspends the calling coroutine if the rate limit is exceeded.
        Thread-safe via asyncio.Lock.
        """
        async with self._lock:
            now = time.monotonic()
            window_start = now - self.period

            # Evict timestamps outside the current window
            self._call_times = [t for t in self._call_times if t > window_start]

            if len(self._call_times) >= self.max_calls:
                # Sleep until the oldest token expires
                wait_until = self._call_times[0] + self.period
                sleep_time = wait_until - now
                if sleep_time > 0:
                    logger.debug(f"[RateLimiter] Rate limit reached — sleeping {sleep_time:.3f}s")
                    await asyncio.sleep(sleep_time)
                # Recompute after sleep
                now = time.monotonic()
                window_start = now - self.period
                self._call_times = [t for t in self._call_times if t > window_start]

            self._call_times.append(time.monotonic())

    @property
    def current_usage(self) -> int:
        """Number of calls made within the current time window."""
        now = time.monotonic()
        return sum(1 for t in self._call_times if t > now - self.period)

    def reset(self) -> None:
        """Clear all recorded call times (useful in tests)."""
        self._call_times = []


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """
    Decorator that wraps an async function with exponential backoff retry.

    Args:
        max_attempts:         Total attempts before re-raising the exception.
        base_delay:           Initial delay between retries (seconds).
        max_delay:            Maximum delay cap (seconds).
        backoff_factor:       Multiplier applied to delay after each failure.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        Decorator that adds retry logic to the decorated async function.

    Example:
        @with_retry(max_attempts=4, base_delay=2.0)
        async def call_api(url: str) -> dict:
            return await fetch(url)
    """
    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = base_delay
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            f"[Retry] {func.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    jitter = delay * 0.1 * (asyncio.get_event_loop().time() % 1)
                    sleep_time = min(delay + jitter, max_delay)
                    logger.warning(
                        f"[Retry] {func.__name__} attempt {attempt}/{max_attempts} failed: {exc}. "
                        f"Retrying in {sleep_time:.1f}s"
                    )
                    await asyncio.sleep(sleep_time)
                    delay = min(delay * backoff_factor, max_delay)

            # Should be unreachable, but satisfies type checker
            raise last_exc or RuntimeError("Retry exhausted")

        return wrapper
    return decorator


if __name__ == "__main__":
    import asyncio as _asyncio

    async def _demo() -> None:
        # Demonstrate rate limiting: 5 calls should take ~0.5s at max_calls=10
        limiter = RateLimiter(max_calls=10.0, period=1.0)
        start = time.monotonic()
        for i in range(5):
            await limiter.acquire()
            print(f"Call {i+1} at t={time.monotonic()-start:.3f}s, usage={limiter.current_usage}")
        elapsed = time.monotonic() - start
        print(f"Total elapsed: {elapsed:.3f}s (expected < 0.1s for 5 calls under 10/s limit)")

        # Demonstrate retry decorator
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.1)
        async def flaky_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Simulated failure on attempt {call_count}")
            return "success"

        result = await flaky_function()
        print(f"Flaky function succeeded on attempt {call_count}: '{result}'")

    _asyncio.run(_demo())
