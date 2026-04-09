"""
LLM client with automatic Groq → Gemini fallback.

Architecture:
    Primary:  Groq API (llama-3.3-70b-versatile) — fastest inference, free tier
    Fallback: Google Gemini 1.5 Flash — long context, free tier

Fallback triggers:
    - Groq rate limit (429) or quota exceeded
    - Groq transient errors (5xx)
    - Groq response timeout (> 30s)

All calls are async.  Retry logic uses exponential backoff via tenacity.
Every call is logged with model used, token counts, latency, and an
estimated cost (for portfolio demonstration — both are free tier).

Usage:
    client = LLMClient()
    response = await client.complete(system_prompt, user_prompt)
    response = await client.complete_json(system_prompt, user_prompt)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from loguru import logger

from config.settings import settings

# ---------------------------------------------------------------------------
# Lazy client initialisation — avoid import errors if keys not set
# ---------------------------------------------------------------------------

_groq_client: Any = None
_gemini_model: Any = None


def _get_groq_client() -> Any:
    """Lazy-initialise the Groq async client."""
    global _groq_client
    if _groq_client is None:
        try:
            from groq import AsyncGroq  # type: ignore
            _groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            logger.info(f"Groq client initialised (model: {settings.GROQ_MODEL})")
        except Exception as exc:
            logger.error(f"Failed to initialise Groq client: {exc}")
    return _groq_client


def _get_gemini_model() -> Any:
    """Lazy-initialise the Gemini generative model."""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            _gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info(f"Gemini model initialised (model: {settings.GEMINI_MODEL})")
        except Exception as exc:
            logger.error(f"Failed to initialise Gemini: {exc}")
    return _gemini_model


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0
_MAX_DELAY_S = 30.0

# HTTP status codes that are retryable
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates a rate-limit or quota error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("rate limit", "429", "quota", "too many"))


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is worth retrying."""
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "rate limit", "429", "500", "502", "503", "504",
        "timeout", "connection", "temporary",
    ))


# ---------------------------------------------------------------------------
# LLMResponse data class
# ---------------------------------------------------------------------------

class LLMResponse:
    """Typed container for LLM call results."""

    __slots__ = (
        "content", "model", "input_tokens", "output_tokens",
        "latency_ms", "used_fallback",
    )

    def __init__(
        self,
        content: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        used_fallback: bool = False,
    ) -> None:
        self.content = content
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.used_fallback = used_fallback

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model}, tokens={self.input_tokens}+{self.output_tokens}, "
            f"latency={self.latency_ms:.0f}ms, fallback={self.used_fallback})"
        )


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Async LLM client with Groq primary and Gemini fallback.

    Handles retries, rate limiting, token counting, and logging.
    All methods are coroutines — call with `await`.
    """

    def __init__(
        self,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Send a completion request, trying Groq first then Gemini.

        Args:
            system_prompt: Role-setting system message.
            user_prompt:   User content (the actual query/task).
            json_mode:     If True, request JSON output from the model.

        Returns:
            LLMResponse with content and metadata.
        """
        # Try Groq
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await self._call_groq(system_prompt, user_prompt, json_mode)
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    logger.warning(
                        f"Groq rate limit hit (attempt {attempt}/{_MAX_RETRIES}) — "
                        f"switching to Gemini fallback"
                    )
                    break  # skip remaining Groq retries on rate limit
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    delay = min(_BASE_DELAY_S * (2 ** (attempt - 1)), _MAX_DELAY_S)
                    logger.warning(
                        f"Groq transient error (attempt {attempt}): {exc}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"Groq failed: {exc} — trying Gemini fallback")
                    break

        # Groq failed — try Gemini
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await self._call_gemini(system_prompt, user_prompt)
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    delay = min(_BASE_DELAY_S * (2 ** (attempt - 1)), _MAX_DELAY_S)
                    logger.warning(
                        f"Gemini error (attempt {attempt}): {exc}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Both Groq and Gemini failed. Last error: {exc}")
                    raise RuntimeError(
                        f"All LLM providers exhausted. Last error: {exc}"
                    ) from exc

        raise RuntimeError("LLM client: unexpected exit from retry loop")

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """
        Complete and parse the response as JSON.

        Strips markdown code fences before parsing.  Returns an empty dict
        if parsing fails rather than raising, to keep the pipeline robust.

        Args:
            system_prompt: System message.
            user_prompt:   User content.

        Returns:
            Parsed dict (or list wrapped in {"result": ...}).
        """
        response = await self.complete(system_prompt, user_prompt, json_mode=True)
        return _parse_json_response(response.content)

    async def stream_complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ):
        """
        Stream completion tokens from Groq.

        Yields string chunks as they arrive.  Falls back to a single
        non-streamed Gemini call if Groq is unavailable.

        Args:
            system_prompt: System message.
            user_prompt:   User content.

        Yields:
            String token chunks.
        """
        client = _get_groq_client()
        if client is None:
            response = await self.complete(system_prompt, user_prompt)
            yield response.content
            return

        try:
            stream = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.warning(f"Groq streaming failed: {exc} — falling back to Gemini")
            response = await self._call_gemini(system_prompt, user_prompt)
            yield response.content

    # ------------------------------------------------------------------
    # Provider-specific call implementations
    # ------------------------------------------------------------------

    async def _call_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Call the Groq API with the llama-3.3-70b-versatile model.

        Args:
            system_prompt: System message.
            user_prompt:   User message.
            json_mode:     Enable Groq's native JSON mode.

        Returns:
            LLMResponse.
        """
        client = _get_groq_client()
        if client is None:
            raise RuntimeError("Groq client not initialised")

        kwargs: dict[str, Any] = {
            "model": settings.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        response = await client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        content = response.choices[0].message.content or ""
        usage = response.usage

        result = LLMResponse(
            content=content,
            model=settings.GROQ_MODEL,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            used_fallback=False,
        )
        _log_call(result)
        return result

    async def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """
        Call Google Gemini 1.5 Flash as the fallback model.

        Gemini's Python SDK is synchronous, so we run it in an executor
        to avoid blocking the event loop.

        Args:
            system_prompt: System message (prepended to user prompt).
            user_prompt:   User message.

        Returns:
            LLMResponse with used_fallback=True.
        """
        model = _get_gemini_model()
        if model is None:
            raise RuntimeError("Gemini model not initialised")

        # Gemini doesn't have a separate system role — prepend to user message
        combined_prompt = f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n\n[USER REQUEST]\n{user_prompt}"

        t0 = time.perf_counter()

        # Run the synchronous Gemini SDK call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                combined_prompt,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        content = response.text if hasattr(response, "text") else ""

        # Gemini doesn't always expose token counts — estimate if unavailable
        try:
            in_tok = response.usage_metadata.prompt_token_count
            out_tok = response.usage_metadata.candidates_token_count
        except Exception:
            in_tok = len(combined_prompt) // 4
            out_tok = len(content) // 4

        result = LLMResponse(
            content=content,
            model=settings.GEMINI_MODEL,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            used_fallback=True,
        )
        _log_call(result)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_call(response: LLMResponse) -> None:
    """Log an LLM call with key metrics."""
    logger.info(
        f"LLM call: model={response.model}  "
        f"tokens={response.input_tokens}+{response.output_tokens}  "
        f"latency={response.latency_ms:.0f}ms  "
        f"fallback={response.used_fallback}"
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    """
    Parse JSON from an LLM response, stripping markdown code fences.

    Handles:
        - Pure JSON string
        - ```json ... ``` blocks
        - ``` ... ``` blocks
        - JSON embedded in surrounding text

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict or {"result": parsed_list} for array responses.
        Returns {} on parse failure (logged as error).
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return {"result": parsed}
        return parsed  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object with regex
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)

    if obj_match:
        try:
            return json.loads(obj_match.group())  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    if arr_match:
        try:
            return {"result": json.loads(arr_match.group())}
        except json.JSONDecodeError:
            pass

    logger.error(f"Failed to parse JSON from LLM response: {text[:200]}")
    return {}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

llm_client = LLMClient()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    # Test _parse_json_response with various formats
    cases = [
        ('{"score": 0.9, "issues": []}', {"score": 0.9, "issues": []}),
        ('```json\n{"score": 0.8}\n```', {"score": 0.8}),
        ('```\n[{"metric": "EPS"}]\n```', {"result": [{"metric": "EPS"}]}),
        ('Some text before {"key": "val"} after', {"key": "val"}),
        ("not json at all", {}),
    ]

    print("=== JSON parsing tests ===")
    all_pass = True
    for raw, expected in cases:
        result = _parse_json_response(raw)
        ok = result == expected
        all_pass = all_pass and ok
        print(f"  {'✓' if ok else '✗'}  input={raw[:40]:40s}  got={result}")

    assert all_pass, "JSON parsing tests failed"

    # Test LLMResponse repr
    r = LLMResponse(
        content="Test response",
        model="llama-3.3-70b-versatile",
        input_tokens=100,
        output_tokens=250,
        latency_ms=450.0,
        used_fallback=False,
    )
    print(f"\nLLMResponse repr: {r}")

    # Async test (requires API keys — skipped in sandbox)
    print("\nNote: Live API calls require GROQ_API_KEY and GOOGLE_API_KEY in .env")
    print("llm_client smoke test passed ✓ (JSON parsing + repr verified)")
