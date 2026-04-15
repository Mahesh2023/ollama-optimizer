"""
Benchmark module for ollama-optimizer.

Runs performance benchmarks on Ollama models to measure throughput,
latency, memory usage, and other metrics before and after optimization.
Supports single-prompt benchmarks, full suites, and model-vs-model comparisons.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from ollama_optimizer.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Stores the result of a single benchmark run against one prompt."""

    model_name: str
    timestamp: str  # ISO 8601 format
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_to_first_token_ms: float
    tokens_per_second: float
    total_time_seconds: float
    prompt_eval_rate: float  # tokens/sec for prompt processing
    eval_rate: float  # tokens/sec for generation
    memory_usage_mb: float
    load_time_seconds: float
    prompt_used: str
    raw_response: dict


@dataclass
class BenchmarkComparison:
    """Side-by-side comparison of an original model versus its optimized variant."""

    model_name: str
    original_model: str
    optimized_model: str
    original_result: BenchmarkResult
    optimized_result: BenchmarkResult
    speedup_ratio: float
    memory_reduction_pct: float
    ttft_improvement_pct: float  # time-to-first-token improvement
    throughput_improvement_pct: float
    quality_note: str


@dataclass
class BenchmarkSuite:
    """Aggregates results and comparisons from a full benchmark session."""

    results: List[BenchmarkResult]
    comparisons: List[BenchmarkComparison]
    system_info: dict
    timestamp: str  # ISO 8601 format
    total_duration_seconds: float


# ---------------------------------------------------------------------------
# Default benchmark prompts
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "simple_qa",
        "prompt": "What is the capital of France? Answer in one sentence.",
        "category": "short",
        "expected_tokens": 20,
    },
    {
        "name": "reasoning",
        "prompt": (
            "Explain step by step: if a train travels at 60 mph for 2.5 hours, "
            "then at 80 mph for 1.5 hours, what is the total distance covered?"
        ),
        "category": "reasoning",
        "expected_tokens": 150,
    },
    {
        "name": "creative",
        "prompt": "Write a short poem about the beauty of mathematics in exactly 8 lines.",
        "category": "creative",
        "expected_tokens": 100,
    },
    {
        "name": "code_gen",
        "prompt": (
            "Write a Python function that finds the longest palindromic substring "
            "in a given string. Include docstring and type hints."
        ),
        "category": "code",
        "expected_tokens": 200,
    },
    {
        "name": "summarization",
        "prompt": (
            "Explain quantum computing to a 10-year-old in 3 paragraphs. "
            "Use simple analogies."
        ),
        "category": "long",
        "expected_tokens": 200,
    },
]


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrates benchmark runs, comparisons, and result persistence."""

    def __init__(self, client: Optional[OllamaClient] = None, timeout: int = 300) -> None:
        self.client: OllamaClient = client or OllamaClient()
        self.default_timeout = timeout

    # ------------------------------------------------------------------
    # Single benchmark
    # ------------------------------------------------------------------

    def run_single_benchmark(
        self,
        model_name: str,
        prompt: str,
        num_ctx: int = 2048,
    ) -> BenchmarkResult:
        """Run a single prompt against a model and collect performance metrics.

        Ollama's ``/api/generate`` response includes (all durations in **nanoseconds**):
        - ``total_duration``
        - ``load_duration``
        - ``prompt_eval_count`` / ``prompt_eval_duration``
        - ``eval_count`` / ``eval_duration``

        We derive:
        - ``tokens_per_second = eval_count / (eval_duration / 1e9)``
        - ``prompt_eval_rate  = prompt_eval_count / (prompt_eval_duration / 1e9)``
        - ``time_to_first_token_ms = (load_duration + prompt_eval_duration) / 1e6``

        Parameters
        ----------
        model_name:
            Name of the Ollama model (e.g. ``"llama3:8b-q4_0"``).
        prompt:
            The text prompt to send to the model.
        num_ctx:
            Context window size passed to Ollama.

        Returns
        -------
        BenchmarkResult
            A populated result dataclass.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        memory_before = self.get_memory_usage()

        logger.info("Benchmarking model=%s  prompt_len=%d  num_ctx=%d", model_name, len(prompt), num_ctx)

        # Use a longer timeout for the generate call – large models
        # (70B+) can take several minutes to load and respond.
        saved_timeout = self.client.timeout
        self.client.timeout = max(self.default_timeout, saved_timeout)

        try:
            wall_start = time.perf_counter()
            response: dict = self.client.generate(
                model=model_name,
                prompt=prompt,
                options={"num_ctx": num_ctx},
            )
            wall_end = time.perf_counter()
        except Exception as exc:
            logger.error("Model %s failed to respond: %s", model_name, exc)
            return BenchmarkResult(
                model_name=model_name,
                timestamp=timestamp,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                time_to_first_token_ms=0.0,
                tokens_per_second=0.0,
                total_time_seconds=0.0,
                prompt_eval_rate=0.0,
                eval_rate=0.0,
                memory_usage_mb=0.0,
                load_time_seconds=0.0,
                prompt_used=prompt,
                raw_response={"error": str(exc)},
            )
        finally:
            self.client.timeout = saved_timeout

        memory_after = self.get_memory_usage()
        memory_usage = max(memory_after, memory_before)

        # Extract Ollama timing fields (nanoseconds) -----------------------
        total_duration_ns: int = response.get("total_duration", 0)
        load_duration_ns: int = response.get("load_duration", 0)
        prompt_eval_count: int = response.get("prompt_eval_count", 0)
        prompt_eval_duration_ns: int = response.get("prompt_eval_duration", 0)
        eval_count: int = response.get("eval_count", 0)
        eval_duration_ns: int = response.get("eval_duration", 0)

        # Derived metrics ---------------------------------------------------
        total_time_seconds = (wall_end - wall_start)
        load_time_seconds = load_duration_ns / 1e9

        if eval_duration_ns > 0:
            tokens_per_second = eval_count / (eval_duration_ns / 1e9)
            eval_rate = tokens_per_second
        else:
            tokens_per_second = 0.0
            eval_rate = 0.0

        if prompt_eval_duration_ns > 0:
            prompt_eval_rate = prompt_eval_count / (prompt_eval_duration_ns / 1e9)
        else:
            prompt_eval_rate = 0.0

        time_to_first_token_ms = (load_duration_ns + prompt_eval_duration_ns) / 1e6

        total_tokens = prompt_eval_count + eval_count

        result = BenchmarkResult(
            model_name=model_name,
            timestamp=timestamp,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=total_tokens,
            time_to_first_token_ms=time_to_first_token_ms,
            tokens_per_second=tokens_per_second,
            total_time_seconds=total_time_seconds,
            prompt_eval_rate=prompt_eval_rate,
            eval_rate=eval_rate,
            memory_usage_mb=memory_usage,
            load_time_seconds=load_time_seconds,
            prompt_used=prompt,
            raw_response=response,
        )

        logger.info(
            "Benchmark complete: model=%s  tps=%.1f  ttft=%.1fms  total=%.2fs",
            model_name,
            tokens_per_second,
            time_to_first_token_ms,
            total_time_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # Full suite
    # ------------------------------------------------------------------

    def run_benchmark_suite(
        self,
        model_name: str,
        prompts: Optional[List[Dict[str, Any]]] = None,
        warmup: bool = True,
        runs_per_prompt: int = 2,
    ) -> List[BenchmarkResult]:
        """Run the complete benchmark suite on a model.

        Parameters
        ----------
        model_name:
            Name of the Ollama model.
        prompts:
            List of prompt dicts (``name``, ``prompt``, ``category``,
            ``expected_tokens``).  Defaults to :data:`BENCHMARK_PROMPTS`.
        warmup:
            If *True*, send a throwaway prompt first so the model is loaded
            and any one-time costs are excluded from real measurements.
        runs_per_prompt:
            How many times each prompt is run.  The **median** result
            (by ``tokens_per_second``) is kept.

        Returns
        -------
        list[BenchmarkResult]
            One result per prompt (the median run).
        """
        if prompts is None:
            prompts = BENCHMARK_PROMPTS

        # Warmup -----------------------------------------------------------
        if warmup:
            logger.info("Warming up model %s ...", model_name)
            try:
                self.client.generate(
                    model=model_name,
                    prompt="Hello",
                    options={"num_ctx": 512},
                )
                logger.info("Warmup complete for %s", model_name)
            except Exception as exc:
                logger.warning("Warmup failed for %s: %s", model_name, exc)

        # Benchmark each prompt --------------------------------------------
        suite_results: List[BenchmarkResult] = []

        for prompt_info in prompts:
            prompt_name = prompt_info["name"]
            prompt_text = prompt_info["prompt"]
            logger.info("Running prompt '%s' (%d runs) ...", prompt_name, runs_per_prompt)

            run_results: List[BenchmarkResult] = []
            for run_idx in range(runs_per_prompt):
                logger.debug("  run %d/%d for '%s'", run_idx + 1, runs_per_prompt, prompt_name)
                result = self.run_single_benchmark(model_name, prompt_text)
                run_results.append(result)

            # Pick the median result by tokens_per_second ------------------
            run_results.sort(key=lambda r: r.tokens_per_second)
            median_idx = len(run_results) // 2
            suite_results.append(run_results[median_idx])

        logger.info(
            "Suite complete for %s: %d prompts benchmarked",
            model_name,
            len(suite_results),
        )
        return suite_results

    # ------------------------------------------------------------------
    # Memory measurement
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> float:
        """Return the RSS memory usage of the *ollama* process in MB.

        Scans all running processes for one whose name contains ``'ollama'``.
        If no matching process is found (or *psutil* raises an error), returns
        ``0.0``.
        """
        try:
            for proc in psutil.process_iter(["name", "memory_info"]):
                try:
                    proc_name = (proc.info.get("name") or "").lower()
                    if "ollama" in proc_name:
                        mem_info = proc.info.get("memory_info")
                        if mem_info is not None:
                            return mem_info.rss / (1024 * 1024)  # bytes -> MB
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as exc:
            logger.warning("Failed to read memory usage: %s", exc)

        return 0.0

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        original_model: str,
        optimized_model: str,
        prompts: Optional[List[Dict[str, Any]]] = None,
    ) -> BenchmarkComparison:
        """Run benchmarks on both models and produce a comparison.

        Aggregates metrics by computing the **mean** across all prompt results
        for each model, then derives:

        - ``speedup_ratio = optimized_tps / original_tps``
        - ``memory_reduction_pct = (1 - optimized_mem / original_mem) * 100``
        - ``ttft_improvement_pct = (1 - optimized_ttft / original_ttft) * 100``
        - ``throughput_improvement_pct = (speedup_ratio - 1) * 100``

        Parameters
        ----------
        original_model:
            Name of the baseline (un-optimized) model.
        optimized_model:
            Name of the optimized model variant.
        prompts:
            Optional list of prompt dicts; defaults to :data:`BENCHMARK_PROMPTS`.

        Returns
        -------
        BenchmarkComparison
        """
        logger.info("Comparing %s (original) vs %s (optimized)", original_model, optimized_model)

        original_results = self.run_benchmark_suite(original_model, prompts=prompts)
        optimized_results = self.run_benchmark_suite(optimized_model, prompts=prompts)

        # Aggregate means --------------------------------------------------
        def _mean(values: List[float]) -> float:
            return statistics.mean(values) if values else 0.0

        orig_tps = _mean([r.tokens_per_second for r in original_results])
        opt_tps = _mean([r.tokens_per_second for r in optimized_results])

        orig_mem = _mean([r.memory_usage_mb for r in original_results])
        opt_mem = _mean([r.memory_usage_mb for r in optimized_results])

        orig_ttft = _mean([r.time_to_first_token_ms for r in original_results])
        opt_ttft = _mean([r.time_to_first_token_ms for r in optimized_results])

        # Derived comparison metrics ---------------------------------------
        speedup_ratio = (opt_tps / orig_tps) if orig_tps > 0 else 0.0
        memory_reduction_pct = ((1 - opt_mem / orig_mem) * 100) if orig_mem > 0 else 0.0
        ttft_improvement_pct = ((1 - opt_ttft / orig_ttft) * 100) if orig_ttft > 0 else 0.0
        throughput_improvement_pct = (speedup_ratio - 1) * 100

        # Build a short quality note
        quality_parts: List[str] = []
        if speedup_ratio > 1.0:
            quality_parts.append(f"{throughput_improvement_pct:.1f}% faster throughput")
        elif speedup_ratio < 1.0:
            quality_parts.append(f"{abs(throughput_improvement_pct):.1f}% slower throughput")
        else:
            quality_parts.append("no throughput change")

        if memory_reduction_pct > 0:
            quality_parts.append(f"{memory_reduction_pct:.1f}% less memory")
        elif memory_reduction_pct < 0:
            quality_parts.append(f"{abs(memory_reduction_pct):.1f}% more memory")

        quality_note = "; ".join(quality_parts) if quality_parts else "no notable difference"

        # Use mean-aggregated pseudo-results for the comparison fields
        orig_aggregate = self._aggregate_result(original_model, original_results)
        opt_aggregate = self._aggregate_result(optimized_model, optimized_results)

        comparison = BenchmarkComparison(
            model_name=original_model,
            original_model=original_model,
            optimized_model=optimized_model,
            original_result=orig_aggregate,
            optimized_result=opt_aggregate,
            speedup_ratio=speedup_ratio,
            memory_reduction_pct=memory_reduction_pct,
            ttft_improvement_pct=ttft_improvement_pct,
            throughput_improvement_pct=throughput_improvement_pct,
            quality_note=quality_note,
        )

        logger.info(
            "Comparison done: speedup=%.2fx  mem_reduction=%.1f%%  ttft_improvement=%.1f%%",
            speedup_ratio,
            memory_reduction_pct,
            ttft_improvement_pct,
        )
        return comparison

    # ------------------------------------------------------------------
    # Full benchmark across model pairs
    # ------------------------------------------------------------------

    def run_full_benchmark(
        self,
        model_pairs: List[Tuple[str, str]],
    ) -> BenchmarkSuite:
        """Run benchmarks and comparisons for every (original, optimized) pair.

        Parameters
        ----------
        model_pairs:
            List of ``(original_name, optimized_name)`` tuples.

        Returns
        -------
        BenchmarkSuite
        """
        suite_start = time.perf_counter()
        all_results: List[BenchmarkResult] = []
        all_comparisons: List[BenchmarkComparison] = []

        for original, optimized in model_pairs:
            logger.info("=== Benchmarking pair: %s -> %s ===", original, optimized)
            try:
                comparison = self.compare_models(original, optimized)
                all_comparisons.append(comparison)
                all_results.append(comparison.original_result)
                all_results.append(comparison.optimized_result)
            except Exception as exc:
                logger.error(
                    "Failed to benchmark pair (%s, %s): %s",
                    original,
                    optimized,
                    exc,
                )

        suite_end = time.perf_counter()

        suite = BenchmarkSuite(
            results=all_results,
            comparisons=all_comparisons,
            system_info=self._collect_system_info(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_duration_seconds=suite_end - suite_start,
        )

        logger.info(
            "Full benchmark finished: %d pairs, %.1fs total",
            len(model_pairs),
            suite.total_duration_seconds,
        )
        return suite

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        suite: BenchmarkSuite,
        filepath: Optional[str] = None,
    ) -> str:
        """Serialize a :class:`BenchmarkSuite` to a JSON file.

        Parameters
        ----------
        suite:
            The suite to save.
        filepath:
            Destination path.  When *None* a timestamped file is created
            under ``~/.ollama-optimizer/benchmarks/``.

        Returns
        -------
        str
            The absolute path of the written file.
        """
        if filepath is None:
            base_dir = Path.home() / ".ollama-optimizer" / "benchmarks"
            base_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filepath = str(base_dir / f"benchmark_{ts}.json")
        else:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = self._suite_to_dict(suite)

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

        logger.info("Benchmark results saved to %s", filepath)
        return filepath

    def load_results(self, filepath: str) -> BenchmarkSuite:
        """Deserialize a :class:`BenchmarkSuite` from a JSON file.

        Parameters
        ----------
        filepath:
            Path to the JSON file previously written by :meth:`save_results`.

        Returns
        -------
        BenchmarkSuite
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            data: dict = json.load(fh)

        results = [self._dict_to_benchmark_result(r) for r in data.get("results", [])]

        comparisons: List[BenchmarkComparison] = []
        for c in data.get("comparisons", []):
            comparisons.append(
                BenchmarkComparison(
                    model_name=c["model_name"],
                    original_model=c["original_model"],
                    optimized_model=c["optimized_model"],
                    original_result=self._dict_to_benchmark_result(c["original_result"]),
                    optimized_result=self._dict_to_benchmark_result(c["optimized_result"]),
                    speedup_ratio=c["speedup_ratio"],
                    memory_reduction_pct=c["memory_reduction_pct"],
                    ttft_improvement_pct=c["ttft_improvement_pct"],
                    throughput_improvement_pct=c["throughput_improvement_pct"],
                    quality_note=c["quality_note"],
                )
            )

        suite = BenchmarkSuite(
            results=results,
            comparisons=comparisons,
            system_info=data.get("system_info", {}),
            timestamp=data.get("timestamp", ""),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
        )

        logger.info("Loaded benchmark results from %s", filepath)
        return suite

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_result(
        model_name: str,
        results: List[BenchmarkResult],
    ) -> BenchmarkResult:
        """Create a single aggregated :class:`BenchmarkResult` from a list.

        Numeric fields are averaged; the first prompt text is kept.
        """
        if not results:
            return BenchmarkResult(
                model_name=model_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                time_to_first_token_ms=0.0,
                tokens_per_second=0.0,
                total_time_seconds=0.0,
                prompt_eval_rate=0.0,
                eval_rate=0.0,
                memory_usage_mb=0.0,
                load_time_seconds=0.0,
                prompt_used="",
                raw_response={},
            )

        n = len(results)
        return BenchmarkResult(
            model_name=model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_tokens=round(sum(r.prompt_tokens for r in results) / n),
            completion_tokens=round(sum(r.completion_tokens for r in results) / n),
            total_tokens=round(sum(r.total_tokens for r in results) / n),
            time_to_first_token_ms=statistics.mean(
                r.time_to_first_token_ms for r in results
            ),
            tokens_per_second=statistics.mean(
                r.tokens_per_second for r in results
            ),
            total_time_seconds=statistics.mean(
                r.total_time_seconds for r in results
            ),
            prompt_eval_rate=statistics.mean(
                r.prompt_eval_rate for r in results
            ),
            eval_rate=statistics.mean(r.eval_rate for r in results),
            memory_usage_mb=statistics.mean(
                r.memory_usage_mb for r in results
            ),
            load_time_seconds=statistics.mean(
                r.load_time_seconds for r in results
            ),
            prompt_used="(aggregated)",
            raw_response={},
        )

    @staticmethod
    def _collect_system_info() -> dict:
        """Gather basic system information for the benchmark report."""
        info: Dict[str, Any] = {}
        try:
            import platform

            info["platform"] = platform.platform()
            info["processor"] = platform.processor()
            info["python_version"] = platform.python_version()
        except Exception:
            pass

        try:
            vm = psutil.virtual_memory()
            info["total_memory_gb"] = round(vm.total / (1024 ** 3), 2)
            info["available_memory_gb"] = round(vm.available / (1024 ** 3), 2)
        except Exception:
            pass

        try:
            info["cpu_count"] = psutil.cpu_count(logical=True)
            info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        except Exception:
            pass

        return info

    # Serialization helpers ------------------------------------------------

    @staticmethod
    def _benchmark_result_to_dict(result: BenchmarkResult) -> dict:
        return asdict(result)

    @staticmethod
    def _dict_to_benchmark_result(d: dict) -> BenchmarkResult:
        return BenchmarkResult(
            model_name=d.get("model_name", ""),
            timestamp=d.get("timestamp", ""),
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            time_to_first_token_ms=d.get("time_to_first_token_ms", 0.0),
            tokens_per_second=d.get("tokens_per_second", 0.0),
            total_time_seconds=d.get("total_time_seconds", 0.0),
            prompt_eval_rate=d.get("prompt_eval_rate", 0.0),
            eval_rate=d.get("eval_rate", 0.0),
            memory_usage_mb=d.get("memory_usage_mb", 0.0),
            load_time_seconds=d.get("load_time_seconds", 0.0),
            prompt_used=d.get("prompt_used", ""),
            raw_response=d.get("raw_response", {}),
        )

    def _suite_to_dict(self, suite: BenchmarkSuite) -> dict:
        return {
            "results": [self._benchmark_result_to_dict(r) for r in suite.results],
            "comparisons": [
                {
                    "model_name": c.model_name,
                    "original_model": c.original_model,
                    "optimized_model": c.optimized_model,
                    "original_result": self._benchmark_result_to_dict(c.original_result),
                    "optimized_result": self._benchmark_result_to_dict(c.optimized_result),
                    "speedup_ratio": c.speedup_ratio,
                    "memory_reduction_pct": c.memory_reduction_pct,
                    "ttft_improvement_pct": c.ttft_improvement_pct,
                    "throughput_improvement_pct": c.throughput_improvement_pct,
                    "quality_note": c.quality_note,
                }
                for c in suite.comparisons
            ],
            "system_info": suite.system_info,
            "timestamp": suite.timestamp,
            "total_duration_seconds": suite.total_duration_seconds,
        }
