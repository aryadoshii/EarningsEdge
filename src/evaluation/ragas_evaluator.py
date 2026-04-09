"""
RAGAS-based RAG evaluation pipeline.

Evaluates the quality of EarningsEdge RAG responses across four dimensions:

    Faithfulness       — Are all answer claims supported by the context?
    Answer Relevance   — Does the answer address the question?
    Context Precision  — Are the retrieved contexts relevant to the query?
    Context Recall     — Does the context contain the ground truth?

RAGAS uses the LLM itself as a judge, so results depend on the LLM quality.
If native ragas cannot run in the current environment, we fall back to the
same Groq/Gemini client used by the main pipeline for consistency.

All scores are in [0, 1].  Higher is better.  Industry benchmarks:
    Faithfulness     > 0.80  acceptable;  > 0.90  excellent
    Answer Relevance > 0.75  acceptable;  > 0.85  excellent
    Context Precision > 0.70 acceptable;  > 0.80  excellent
    Context Recall   > 0.65  acceptable;  > 0.80  excellent

Usage:
    evaluator = RAGASEvaluator()
    result = await evaluator.evaluate(query, answer, contexts, ground_truth)
    batch  = await evaluator.evaluate_batch(ragas_inputs)
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from loguru import logger

from src.ingestion.data_validator import RAGEvaluation

# RAGAS lazy import — graceful fallback if not installed
_ragas_available = False
try:
    from ragas import evaluate as ragas_evaluate                    # type: ignore
    from ragas.metrics import (                                     # type: ignore
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from datasets import Dataset                                     # type: ignore
    _ragas_available = True
except ImportError:
    logger.warning(
        "ragas not installed — evaluator will use LLM-based fallback scoring"
    )


# ---------------------------------------------------------------------------
# LLM-based fallback scorer (used when ragas is unavailable)
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
Rate whether the following ANSWER is factually supported by the CONTEXT.
Every claim in the answer must be traceable to the context.

CONTEXT:
{context}

ANSWER:
{answer}

Score from 0.0 (answer contradicts or ignores context) to 1.0 (fully grounded).
Output only a JSON object: {{"score": <float>}}"""

_RELEVANCE_PROMPT = """\
Rate whether the following ANSWER directly addresses the QUESTION.

QUESTION: {question}
ANSWER: {answer}

Score from 0.0 (completely off-topic) to 1.0 (precisely addresses the question).
Output only a JSON object: {{"score": <float>}}"""


async def _llm_score(prompt: str) -> float:
    """Run a scoring prompt through the LLM and extract the float score."""
    try:
        from src.rag.llm_client import llm_client
        result = await llm_client.complete_json(
            system_prompt="You are an objective evaluator. Output only JSON.",
            user_prompt=prompt,
        )
        return float(result.get("score", 0.5))
    except Exception as exc:
        logger.debug(f"LLM scoring failed: {exc}")
        return 0.5


# ---------------------------------------------------------------------------
# RAGASEvaluator
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS metrics.

    Falls back to LLM-based scoring for faithfulness and answer relevance
    when the ragas library is not available.
    """

    def __init__(self, model_used: str = "") -> None:
        self.model_used = model_used

        # Avoid repeating the same environment warning on every evaluation.
        self._warned_native_ragas_unavailable = False

    @staticmethod
    def _fmt_score(score: float | None) -> str:
        """Format optional score values for logging."""
        return "N/A" if score is None else f"{score:.3f}"

    def _native_ragas_ready(self) -> tuple[bool, str | None]:
        """
        Determine whether native ragas can run in this environment.

        ragas.evaluate() defaults to an OpenAI-backed judge unless a custom
        evaluator LLM is wired in. This project does not currently configure
        that path, so require OPENAI_API_KEY before attempting native ragas.
        """
        if not _ragas_available:
            return False, None
        if os.getenv("OPENAI_API_KEY"):
            return True, None
        return (
            False,
            "OPENAI_API_KEY not set; using built-in RAG evaluation fallback instead of native ragas.",
        )

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> RAGEvaluation:
        """
        Evaluate one RAG query-answer-context triple.

        Args:
            query:        The original user question.
            answer:       The LLM-generated answer.
            contexts:     List of retrieved context strings.
            ground_truth: Expected correct answer (used for recall/precision).

        Returns:
            RAGEvaluation with all four RAGAS scores populated.
        """
        t0 = time.perf_counter()

        ragas_ready, reason = self._native_ragas_ready()
        if ragas_ready:
            scores = await self._evaluate_with_ragas(
                query, answer, contexts, ground_truth
            )
        else:
            if reason and not self._warned_native_ragas_unavailable:
                logger.warning(reason)
                self._warned_native_ragas_unavailable = True
            scores = await self._evaluate_with_llm_fallback(
                query, answer, contexts
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        result = RAGEvaluation(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            faithfulness=scores.get("faithfulness"),
            answer_relevance=scores.get("answer_relevance"),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
            model_used=self.model_used,
            latency_ms=latency_ms,
        )

        logger.info(
            f"RAGAS evaluation: "
            f"faithfulness={self._fmt_score(result.faithfulness)}  "
            f"relevance={self._fmt_score(result.answer_relevance)}  "
            f"precision={self._fmt_score(result.context_precision)}  "
            f"recall={self._fmt_score(result.context_recall)}  "
            f"mean={result.mean_score:.3f}  "
            f"latency={latency_ms:.0f}ms"
        )
        return result

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    async def evaluate_batch(
        self,
        ragas_inputs: list[dict[str, Any]],
        concurrency: int = 3,
    ) -> list[RAGEvaluation]:
        """
        Evaluate multiple RAG outputs concurrently.

        Args:
            ragas_inputs: List of dicts, each with keys:
                          query, answer, contexts, ground_truth (optional).
            concurrency:  Max simultaneous evaluations.

        Returns:
            List of RAGEvaluation objects in the same order as inputs.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _eval_one(inp: dict[str, Any]) -> RAGEvaluation:
            async with sem:
                return await self.evaluate(
                    query=inp.get("query", ""),
                    answer=inp.get("answer", ""),
                    contexts=inp.get("contexts", []),
                    ground_truth=inp.get("ground_truth", ""),
                )

        results = await asyncio.gather(
            *[_eval_one(inp) for inp in ragas_inputs],
            return_exceptions=True,
        )

        final: list[RAGEvaluation] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"RAGAS batch item {i} failed: {r}")
                final.append(RAGEvaluation(
                    query=ragas_inputs[i].get("query", ""),
                    answer="",
                    contexts=[],
                ))
            else:
                final.append(r)  # type: ignore[arg-type]

        mean_scores = [r.mean_score for r in final if r.mean_score > 0]
        if mean_scores:
            logger.info(
                f"RAGAS batch: {len(final)} evaluations  "
                f"mean_overall={sum(mean_scores)/len(mean_scores):.3f}"
            )
        return final

    # ------------------------------------------------------------------
    # RAGAS library path
    # ------------------------------------------------------------------

    async def _evaluate_with_ragas(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict[str, float | None]:
        """
        Run evaluation using the ragas library.

        Builds a HuggingFace Dataset from the inputs and calls ragas.evaluate.
        Runs in a thread pool since ragas is synchronous.

        Args:
            query, answer, contexts, ground_truth: Evaluation inputs.

        Returns:
            Dict with metric scores.
        """
        loop = asyncio.get_event_loop()

        def _run() -> dict[str, float | None]:
            try:
                dataset = Dataset.from_dict({
                    "question":   [query],
                    "answer":     [answer],
                    "contexts":   [contexts],
                    "ground_truth": [ground_truth] if ground_truth else [""],
                })
                metrics = [faithfulness, answer_relevancy, context_precision]
                if ground_truth:
                    metrics.append(context_recall)

                result = ragas_evaluate(dataset, metrics=metrics)
                scores: dict[str, float | None] = {}

                # ragas returns a Dataset with metric columns
                result_dict = result.to_pandas().iloc[0].to_dict()
                scores["faithfulness"]      = float(result_dict.get("faithfulness", 0.5))
                scores["answer_relevance"]  = float(result_dict.get("answer_relevancy", 0.5))
                scores["context_precision"] = float(result_dict.get("context_precision", 0.5))
                if ground_truth:
                    scores["context_recall"] = float(result_dict.get("context_recall", 0.5))
                return scores
            except Exception as exc:
                logger.error(f"ragas.evaluate failed: {exc}")
                return {}

        scores = await loop.run_in_executor(None, _run)
        return scores if scores else await self._evaluate_with_llm_fallback(
            query, answer, contexts
        )

    # ------------------------------------------------------------------
    # LLM fallback path
    # ------------------------------------------------------------------

    async def _evaluate_with_llm_fallback(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float | None]:
        """
        Approximate RAGAS metrics using direct LLM scoring.

        Runs faithfulness and answer relevance in parallel.
        Context precision is approximated by answer relevance.
        Context recall defaults to None (needs ground truth).

        Args:
            query, answer, contexts: Evaluation inputs.

        Returns:
            Dict with partial metric scores.
        """
        context_text = "\n\n".join(contexts[:5])[:3000]

        faith_prompt = _FAITHFULNESS_PROMPT.format(
            context=context_text, answer=answer[:2000]
        )
        relev_prompt = _RELEVANCE_PROMPT.format(
            question=query, answer=answer[:2000]
        )

        faith_score, relev_score = await asyncio.gather(
            _llm_score(faith_prompt),
            _llm_score(relev_prompt),
        )

        return {
            "faithfulness":      float(max(0.0, min(1.0, faith_score))),
            "answer_relevance":  float(max(0.0, min(1.0, relev_score))),
            "context_precision": float(max(0.0, min(1.0, relev_score * 0.9))),
            "context_recall":    None,  # cannot compute without ground truth
        }

    # ------------------------------------------------------------------
    # Summary helpers for dashboard
    # ------------------------------------------------------------------

    @staticmethod
    def summarise_batch(evaluations: list[RAGEvaluation]) -> dict[str, Any]:
        """
        Compute aggregate statistics from a list of RAGAS evaluations.

        Args:
            evaluations: List of RAGEvaluation objects.

        Returns:
            Dict with mean and std for each metric.
        """
        import statistics

        def _stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": round(statistics.mean(values), 4),
                "std":  round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
                "min":  round(min(values), 4),
                "max":  round(max(values), 4),
            }

        return {
            "faithfulness":      _stats([e.faithfulness for e in evaluations if e.faithfulness is not None]),
            "answer_relevance":  _stats([e.answer_relevance for e in evaluations if e.answer_relevance is not None]),
            "context_precision": _stats([e.context_precision for e in evaluations if e.context_precision is not None]),
            "context_recall":    _stats([e.context_recall for e in evaluations if e.context_recall is not None]),
            "mean_overall":      round(sum(e.mean_score for e in evaluations) / len(evaluations), 4) if evaluations else 0.0,
            "total_evaluations": len(evaluations),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

ragas_evaluator = RAGASEvaluator()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    evaluator = RAGASEvaluator(model_used="llama-3.3-70b-versatile")

    # Test RAGEvaluation mean_score property
    from src.ingestion.data_validator import RAGEvaluation
    eval_obj = RAGEvaluation(
        query="What was AAPL Q3 2024 revenue?",
        answer="AAPL Q3 2024 revenue was $94.9 billion, up 6% year over year.",
        contexts=["Revenue of $94.9B in Q3 2024, +6% YoY."],
        ground_truth="Apple reported $94.9B in Q3 2024.",
        faithfulness=0.95,
        answer_relevance=0.90,
        context_precision=0.85,
        context_recall=0.88,
    )
    assert abs(eval_obj.mean_score - (0.95 + 0.90 + 0.85 + 0.88) / 4) < 1e-6
    print(f"RAGEvaluation.mean_score: {eval_obj.mean_score:.4f} ✓")

    # Test summarise_batch
    evals = [eval_obj, eval_obj]
    summary = RAGASEvaluator.summarise_batch(evals)
    assert summary["faithfulness"]["mean"] == 0.95
    assert summary["total_evaluations"] == 2
    print(f"summarise_batch: mean_overall={summary['mean_overall']:.4f} ✓")

    print(f"\nragas library available: {_ragas_available}")
    print("ragas_evaluator smoke test passed ✓")
