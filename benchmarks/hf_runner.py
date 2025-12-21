#!/usr/bin/env python3
# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Delia Benchmark Runner with HuggingFace Datasets.

Runs standard LLM benchmarks through Delia's orchestration layer using real
datasets from HuggingFace. Supports train/test splits, comparison modes,
and YAML-based benchmark configurations.

Usage:
    # List available benchmarks
    uv run python benchmarks/hf_runner.py list

    # Run a benchmark with Delia
    uv run python benchmarks/hf_runner.py arc_challenge --limit 50

    # Compare baseline vs Delia
    uv run python benchmarks/hf_runner.py mmlu --limit 100 --compare

    # Run with voting mode
    uv run python benchmarks/hf_runner.py gsm8k --limit 50 --mode voting --voting-k 3

    # Run baseline only (no Delia orchestration)
    uv run python benchmarks/hf_runner.py hellaswag --limit 50 --mode baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TASKS_DIR = Path(__file__).parent / "tasks"


@dataclass
class BenchmarkResult:
    """Result for a single benchmark problem."""
    problem_id: str
    correct: bool
    score: float
    latency_ms: float
    tokens_used: int
    response: str
    expected: str
    mode: str


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    benchmark: str
    mode: str
    results: list[BenchmarkResult]
    split: str = "test"

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.results)


def load_benchmark_config(name: str) -> dict:
    """Load benchmark configuration from YAML."""
    config_path = TASKS_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Benchmark '{name}' not found. Available: {list_benchmarks()}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_benchmarks() -> list[str]:
    """List available benchmarks."""
    if not TASKS_DIR.exists():
        return []
    return sorted([p.stem for p in TASKS_DIR.glob("*.yaml")])


def format_choices(choices: list[str], labels: list[str] | None = None) -> str:
    """Format multiple choice options."""
    if labels is None:
        labels = [chr(65 + i) for i in range(len(choices))]  # A, B, C, D...
    return "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def extract_answer_letter(response: str) -> str | None:
    """Extract answer letter from response."""
    response = response.strip().upper()
    # Check for letter at start
    if response and response[0] in "ABCD":
        return response[0]
    # Check for "Answer: X" pattern
    match = re.search(r"(?:answer|option)[:\s]*([A-D])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Check for standalone letter
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return match.group(1).upper()
    return None


def extract_number(response: str) -> float | None:
    """Extract final number from response."""
    # Look for "answer is X" or "= X" patterns first
    patterns = [
        r"(?:answer|result|total|equals?)[:\s]+\$?([\d,]+(?:\.\d+)?)",
        r"=\s*\$?([\d,]+(?:\.\d+)?)\s*$",
        r"\$?([\d,]+(?:\.\d+)?)\s*(?:dollars?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue
    # Fallback: last number in response
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", response)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


class DeliaClient:
    """Client for calling LLMs through Delia or directly."""

    def __init__(self):
        self._initialized = False
        self._backend_manager = None

    async def _ensure_initialized(self):
        """Lazy initialization of Delia components."""
        if self._initialized:
            return

        from delia.backend_manager import get_backend_manager
        from delia.llm import init_llm_module
        from delia.melons import MelonTracker
        from delia.queue import ModelQueue

        self._backend_manager = get_backend_manager()
        self._tracker = MelonTracker()

        # Initialize LLM module
        model_queue = ModelQueue()
        init_llm_module(
            stats_callback=lambda *args, **kwargs: None,
            save_stats_callback=lambda: None,
            model_queue=model_queue,
        )

        # Probe backends
        for backend in self._backend_manager.get_enabled_backends():
            try:
                probed = await self._backend_manager.probe_backend(backend.id)
                if probed:
                    print(f"  Probed {backend.id}: {list(backend.models.keys())}")
            except Exception as e:
                print(f"  Warning: Failed to probe {backend.id}: {e}")

        self._initialized = True

    async def call_delia(self, prompt: str, tier: str = "quick") -> dict:
        """Call through Delia's orchestration."""
        await self._ensure_initialized()

        from delia.llm import call_llm
        from delia.routing import get_router

        router = get_router()
        _, backend = await router.select_optimal_backend(
            content=prompt,
            file_path=None,
            task_type=tier,
            backend_type=None,
        )

        model_list = backend.models.get(tier) or backend.models.get("coder") or []
        if not model_list:
            return {"response": "", "error": "No model", "tokens": 0}
        model_name = model_list[0]

        response = await call_llm(
            model=model_name,
            prompt=prompt,
            backend_obj=backend,
        )

        return {
            "response": response.get("response", "") if isinstance(response, dict) else "",
            "tokens": response.get("tokens", 0) if isinstance(response, dict) else 0,
            "mode": "delia",
        }

    async def call_baseline(self, prompt: str, tier: str = "quick") -> dict:
        """Call Ollama directly, bypassing Delia."""
        import httpx

        await self._ensure_initialized()

        backends = self._backend_manager.get_enabled_backends()
        if not backends:
            return {"response": "", "error": "No backends", "tokens": 0}

        backend = backends[0]
        model_list = backend.models.get(tier) or backend.models.get("coder") or []
        if not model_list:
            return {"response": "", "error": "No model", "tokens": 0}
        model_name = model_list[0]

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{backend.url}/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "response": data.get("response", ""),
                        "tokens": data.get("eval_count", 0),
                        "mode": "baseline",
                    }
                return {"response": "", "error": f"HTTP {response.status_code}", "tokens": 0}
        except Exception as e:
            return {"response": "", "error": str(e), "tokens": 0}

    async def call_voting(self, prompt: str, tier: str = "quick", k: int = 3) -> dict:
        """Call with voting consensus."""
        await self._ensure_initialized()

        from delia.llm import call_llm

        backends = self._backend_manager.get_enabled_backends()
        if not backends:
            return {"response": "", "error": "No backends", "tokens": 0}

        async def single_call(idx: int):
            backend = backends[idx % len(backends)]
            model_list = backend.models.get(tier) or backend.models.get("coder") or []
            if not model_list:
                return None
            model_name = model_list[0]
            try:
                return await call_llm(model=model_name, prompt=prompt, backend_obj=backend)
            except Exception:
                return None

        results = await asyncio.gather(*[single_call(i) for i in range(k)])
        valid = [r for r in results if r and r.get("response")]

        if not valid:
            return {"response": "", "error": "No valid responses", "tokens": 0}

        # Simple majority: return most common answer letter or first response
        responses = [r.get("response", "") for r in valid]
        # Try to extract answer letters for voting
        letters = [extract_answer_letter(r) for r in responses]
        letters = [l for l in letters if l]
        if letters:
            from collections import Counter
            most_common = Counter(letters).most_common(1)[0][0]
            # Return first response that has this letter
            for r in responses:
                if extract_answer_letter(r) == most_common:
                    return {"response": r, "tokens": sum(v.get("tokens", 0) for v in valid), "mode": "voting"}

        return {"response": valid[0].get("response", ""), "tokens": sum(v.get("tokens", 0) for v in valid), "mode": "voting"}


class HFBenchmarkRunner:
    """Runs benchmarks using HuggingFace datasets."""

    def __init__(
        self,
        mode: str = "delia",
        voting_k: int = 3,
        tier: str | None = None,
        split_ratio: float = 0.8,
    ):
        self.mode = mode
        self.voting_k = voting_k
        self.tier = tier
        self.split_ratio = split_ratio
        self.client = DeliaClient()

    async def run_benchmark(
        self,
        name: str,
        limit: int | None = None,
        quiet: bool = False,
    ) -> BenchmarkSummary:
        """Run a benchmark by name."""
        config = load_benchmark_config(name)

        # Load dataset
        from datasets import load_dataset

        data_config = config["data"]
        dataset_kwargs = {"path": data_config["dataset_path"]}
        if "dataset_name" in data_config:
            dataset_kwargs["name"] = data_config["dataset_name"]

        if not quiet:
            print(f"Loading dataset {data_config['dataset_path']}...")

        try:
            dataset = load_dataset(**dataset_kwargs, split=data_config.get("split", "test"))
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return BenchmarkSummary(name, self.mode, [])

        # Apply limit
        actual_limit = limit or data_config.get("limit", 100)
        if len(dataset) > actual_limit:
            # Random sample for variety
            indices = random.sample(range(len(dataset)), actual_limit)
            dataset = dataset.select(indices)

        if not quiet:
            print(f"Running {len(dataset)} samples...")

        # Get recommended tier from config
        tier = self.tier or config.get("metadata", {}).get("tier", "quick")

        results = []
        for i, sample in enumerate(dataset):
            prompt = self._format_prompt(config, sample)
            expected = self._get_expected(config, sample)

            start = time.perf_counter()
            response = await self._run_prompt(prompt, tier)
            elapsed_ms = (time.perf_counter() - start) * 1000

            correct, score = self._check_answer(config, response, expected, sample)

            results.append(BenchmarkResult(
                problem_id=f"{name}_{i}",
                correct=correct,
                score=score,
                latency_ms=elapsed_ms,
                tokens_used=response.get("tokens", 0),
                response=response.get("response", "")[:200],
                expected=str(expected)[:50],
                mode=self.mode,
            ))

            if not quiet:
                status = "✓" if correct else "✗"
                print(f"  {name} {i+1}/{len(dataset)}: {status}")

        return BenchmarkSummary(name, self.mode, results)

    def _format_prompt(self, config: dict, sample: dict) -> str:
        """Format prompt from config template and sample data."""
        template = config["prompt"]["template"]

        # Handle different dataset formats
        if "choices" in sample:
            choices = sample["choices"]
            if isinstance(choices, dict):
                # ARC format: {"text": [...], "label": [...]}
                formatted = format_choices(choices.get("text", []), choices.get("label", []))
            else:
                formatted = format_choices(choices)
            sample = {**sample, "choices": formatted}

        if "endings" in sample:
            # HellaSwag format
            for j, ending in enumerate(sample["endings"]):
                sample[f"ending{j}"] = ending

        # Format template
        try:
            return template.format(**sample)
        except KeyError as e:
            # Fallback for missing fields
            return template.format(**{k: sample.get(k, "") for k in config["prompt"].get("input_fields", [])})

    def _get_expected(self, config: dict, sample: dict) -> str:
        """Get expected answer from sample."""
        answer_field = config["prompt"].get("answer_field", "answer")
        answer = sample.get(answer_field, "")

        # Handle GSM8K format: "#### 18"
        if isinstance(answer, str) and "####" in answer:
            answer = answer.split("####")[-1].strip()

        return str(answer)

    def _check_answer(self, config: dict, response: dict, expected: str, sample: dict) -> tuple[bool, float]:
        """Check if response matches expected answer."""
        resp_text = response.get("response", "")
        metrics = config.get("metrics", [{"name": "exact_match"}])
        metric = metrics[0]["name"]

        if metric == "exact_match":
            # Letter matching for multiple choice
            resp_letter = extract_answer_letter(resp_text)
            exp_letter = expected.strip().upper()
            # Handle numeric labels (HellaSwag uses 0,1,2,3)
            if exp_letter.isdigit():
                exp_letter = chr(65 + int(exp_letter))  # 0->A, 1->B, etc.
            correct = resp_letter == exp_letter
            return correct, 1.0 if correct else 0.0

        elif metric == "numeric_match":
            resp_num = extract_number(resp_text)
            try:
                exp_num = float(expected.replace(",", ""))
                if resp_num is not None:
                    correct = abs(resp_num - exp_num) < 0.01
                    return correct, 1.0 if correct else 0.0
            except ValueError:
                pass
            return False, 0.0

        elif metric == "pass_at_1":
            # Code execution test
            test_code = sample.get(config["prompt"].get("test_field", "test"), "")
            return self._run_code_test(resp_text, test_code)

        elif metric == "truthfulness":
            # Check against correct/incorrect answers
            correct_answers = sample.get("correct_answers", [])
            incorrect_answers = sample.get("incorrect_answers", [])
            resp_lower = resp_text.lower()
            for correct in correct_answers:
                if correct.lower() in resp_lower:
                    return True, 1.0
            for incorrect in incorrect_answers:
                if incorrect.lower() in resp_lower:
                    return False, 0.0
            return False, 0.5  # Uncertain

        return False, 0.0

    def _run_code_test(self, code: str, test: str) -> tuple[bool, float]:
        """Run code and test it."""
        try:
            exec_globals: dict = {}
            exec(code, exec_globals)
            exec(test, exec_globals)
            exec_globals["check"](exec_globals.get(list(exec_globals.keys())[-1]))
            return True, 1.0
        except Exception:
            return False, 0.0

    async def _run_prompt(self, prompt: str, tier: str) -> dict:
        """Run prompt with configured mode."""
        if self.mode == "baseline":
            return await self.client.call_baseline(prompt, tier)
        elif self.mode == "voting":
            return await self.client.call_voting(prompt, tier, self.voting_k)
        else:  # delia
            return await self.client.call_delia(prompt, tier)


async def run_comparison(
    benchmark: str,
    limit: int,
    tier: str | None,
    voting_k: int,
    quiet: bool,
) -> None:
    """Run baseline vs Delia comparison."""
    print(f"\n{'='*70}")
    print(f"Comparison: {benchmark.upper()}")
    print(f"{'='*70}")

    results = {}

    for mode in ["baseline", "delia", "voting"]:
        print(f"\n--- Running {mode.upper()} mode ---")
        runner = HFBenchmarkRunner(mode=mode, voting_k=voting_k, tier=tier)
        summary = await runner.run_benchmark(benchmark, limit=limit, quiet=quiet)
        results[mode] = summary
        print(f"Accuracy: {summary.accuracy*100:.1f}%, Latency: {summary.avg_latency_ms:.0f}ms")

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Mode':<12} {'Accuracy':>10} {'Latency':>12} {'vs Baseline':>12}")
    print("-" * 50)

    baseline_acc = results["baseline"].accuracy
    for mode in ["baseline", "delia", "voting"]:
        s = results[mode]
        delta = (s.accuracy - baseline_acc) * 100
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%" if delta < 0 else "---"
        print(f"{mode:<12} {s.accuracy*100:>9.1f}% {s.avg_latency_ms:>10.0f}ms {delta_str:>12}")


async def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks through Delia")
    parser.add_argument(
        "benchmark",
        nargs="?",
        help="Benchmark to run (or 'list' to show available)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Override sample limit",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["baseline", "delia", "voting"],
        default="delia",
        help="Execution mode",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare baseline vs Delia vs voting",
    )
    parser.add_argument(
        "--voting-k", "-k",
        type=int,
        default=3,
        help="Number of votes for voting mode",
    )
    parser.add_argument(
        "--tier", "-t",
        choices=["quick", "coder", "moe"],
        help="Force model tier (default: from benchmark config)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-sample output",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file",
    )

    args = parser.parse_args()

    # Handle 'list' command
    if args.benchmark == "list" or not args.benchmark:
        print("\nAvailable benchmarks:")
        print("-" * 60)
        for name in list_benchmarks():
            config = load_benchmark_config(name)
            meta = config.get("metadata", {})
            desc = meta.get("description", "")[:40]
            tier = meta.get("tier", "quick")
            diff = meta.get("difficulty", "?")
            print(f"  {name:<16} {tier:<8} {diff:<8} {desc}")
        return

    # Validate benchmark
    if args.benchmark not in list_benchmarks():
        print(f"Unknown benchmark: {args.benchmark}")
        print(f"Available: {', '.join(list_benchmarks())}")
        return

    # Run comparison or single mode
    if args.compare:
        await run_comparison(
            args.benchmark,
            args.limit or 50,
            args.tier,
            args.voting_k,
            args.quiet,
        )
    else:
        runner = HFBenchmarkRunner(
            mode=args.mode,
            voting_k=args.voting_k,
            tier=args.tier,
        )

        print(f"\n{'='*60}")
        print(f"Delia Benchmark: {args.benchmark}")
        print(f"Mode: {args.mode}" + (f" (k={args.voting_k})" if args.mode == "voting" else ""))
        print(f"{'='*60}\n")

        summary = await runner.run_benchmark(
            args.benchmark,
            limit=args.limit,
            quiet=args.quiet,
        )

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {summary.accuracy*100:.1f}%")
        print(f"Avg Latency: {summary.avg_latency_ms:.0f}ms")
        print(f"Total Tokens: {summary.total_tokens}")

        # Save if requested
        if args.output:
            output = {
                "benchmark": args.benchmark,
                "mode": args.mode,
                "accuracy": summary.accuracy,
                "avg_latency_ms": summary.avg_latency_ms,
                "total_tokens": summary.total_tokens,
                "sample_count": len(summary.results),
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
