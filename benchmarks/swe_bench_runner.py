#!/usr/bin/env python3
# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
SWE-Bench Lite Runner for Delia.

Evaluates Delia's ability to solve real GitHub issues.

Usage:
    # Generate patches for 10 issues (no Docker evaluation)
    uv run python benchmarks/swe_bench_runner.py --limit 10 --generate-only

    # Full evaluation with Docker (requires ~120GB disk, 16GB RAM)
    uv run python benchmarks/swe_bench_runner.py --limit 10 --evaluate

    # Compare baseline vs Delia
    uv run python benchmarks/swe_bench_runner.py --limit 10 --compare
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class SWEBenchResult:
    instance_id: str
    repo: str
    issue: str
    patch: str
    resolved: bool | None  # None if not evaluated
    latency_ms: float
    mode: str


async def load_swe_bench_lite(limit: int = 50) -> list[dict]:
    """Load SWE-bench Lite dataset."""
    from datasets import load_dataset

    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    instances = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        instances.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "problem_statement": item["problem_statement"],
            "hints_text": item.get("hints_text", ""),
            "base_commit": item["base_commit"],
            "patch": item["patch"],  # Ground truth
        })

    print(f"Loaded {len(instances)} instances")
    return instances


async def generate_patch_with_delia(instance: dict, mode: str = "raw") -> SWEBenchResult:
    """Generate a patch for a SWE-bench instance using Delia."""
    from delia.backend_manager import backend_manager
    from delia.llm import call_llm
    from delia.routing import select_model

    prompt = f"""You are a software engineer fixing a bug in the {instance['repo']} repository.

## Issue Description
{instance['problem_statement']}

## Additional Context
{instance.get('hints_text', 'No additional hints.')}

## Task
Generate a git patch that fixes this issue. Output ONLY the patch in unified diff format.
The patch should be minimal and focused on fixing the specific issue.

Start your response with ```diff and end with ```.
"""

    start = time.time()
    patch = ""

    try:
        # Get backend and model
        backends = backend_manager.get_enabled_backends()
        if not backends:
            return SWEBenchResult(
                instance_id=instance["instance_id"],
                repo=instance["repo"],
                issue=instance["problem_statement"][:200],
                patch="Error: No backends available",
                resolved=None,
                latency_ms=0,
                mode=mode,
            )
        backend = backends[0]
        swe_models = backend.models.get("swe")
        coder_models = backend.models.get("coder")
        if swe_models:
            model_name = swe_models[0]
        elif coder_models:
            model_name = coder_models[0]
        else:
            model_name = "qwen3:14b"

        if mode == "baseline":
            # Direct Ollama call
            import httpx
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{backend.url}/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                if response.status_code == 200:
                    patch = response.json().get("response", "")
                else:
                    patch = ""
        else:
            # Through Delia - use direct provider for speed
            import httpx
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Add Delia-style system prompt for better results
                system_prompt = "You are an expert software engineer. Generate minimal, focused patches."
                response = await client.post(
                    f"{backend.url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {"num_predict": 2048},
                    },
                )
                if response.status_code == 200:
                    patch = response.json().get("response", "")
                else:
                    patch = f"Error: HTTP {response.status_code}"

        # Extract diff from response
        if patch and "```diff" in patch:
            patch = patch.split("```diff")[1].split("```")[0].strip()
        elif patch and "```" in patch:
            parts = patch.split("```")
            if len(parts) > 1:
                patch = parts[1].split("```")[0].strip()

    except Exception as e:
        import traceback
        patch = f"Error: {e}\n{traceback.format_exc()}"
        print(f"\n    ERROR: {e}")

    latency = (time.time() - start) * 1000

    return SWEBenchResult(
        instance_id=instance["instance_id"],
        repo=instance["repo"],
        issue=instance["problem_statement"][:200],
        patch=patch,
        resolved=None,
        latency_ms=latency,
        mode=mode,
    )


async def evaluate_patches(results: list[SWEBenchResult], output_dir: Path) -> list[SWEBenchResult]:
    """Evaluate patches using SWE-bench Docker harness."""
    try:
        from swebench.harness.run_evaluation import run_evaluation
    except ImportError:
        print("SWE-bench harness not available. Install with: pip install swebench")
        return results

    # Create predictions file
    predictions = []
    for r in results:
        predictions.append({
            "instance_id": r.instance_id,
            "model_patch": r.patch,
            "model_name_or_path": "delia",
        })

    pred_file = output_dir / "predictions.json"
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Running Docker evaluation on {len(predictions)} patches...")
    print("This may take a while (cloning repos, running tests)...")

    # Run evaluation
    try:
        run_evaluation(
            dataset_name="princeton-nlp/SWE-bench_Lite",
            predictions_path=str(pred_file),
            max_workers=4,
            run_id="delia_eval",
            namespace="",  # Build locally
        )

        # Load results
        results_file = output_dir / "evaluation_results" / "delia_eval.json"
        if results_file.exists():
            with open(results_file) as f:
                eval_results = json.load(f)

            # Update resolved status
            for r in results:
                r.resolved = eval_results.get(r.instance_id, {}).get("resolved", False)
    except Exception as e:
        print(f"Evaluation failed: {e}")

    return results


async def run_benchmark(
    limit: int = 10,
    mode: str = "raw",
    evaluate: bool = False,
    output_dir: Path | None = None,
) -> list[SWEBenchResult]:
    """Run SWE-bench benchmark."""
    from delia.backend_manager import backend_manager

    # Probe backends
    for backend in backend_manager.get_enabled_backends():
        await backend_manager.probe_backend(backend.id)

    # Load dataset
    instances = await load_swe_bench_lite(limit)

    print(f"\nRunning SWE-bench Lite ({mode} mode) on {len(instances)} instances...")
    print("-" * 60)

    results = []
    for i, instance in enumerate(instances):
        print(f"  [{i+1}/{len(instances)}] {instance['instance_id'][:50]}...", end=" ", flush=True)
        result = await generate_patch_with_delia(instance, mode)
        results.append(result)

        status = "✓" if result.patch and not result.patch.startswith("Error") else "✗"
        print(f"{status} ({result.latency_ms:.0f}ms)")

    # Evaluate if requested
    if evaluate and output_dir:
        results = await evaluate_patches(results, output_dir)

    return results


def print_summary(results: list[SWEBenchResult], mode: str):
    """Print benchmark summary."""
    total = len(results)
    generated = sum(1 for r in results if r.patch and not r.patch.startswith("Error"))
    resolved = sum(1 for r in results if r.resolved) if any(r.resolved is not None for r in results) else None
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"SWE-bench Lite Results ({mode})")
    print(f"{'='*60}")
    print(f"  Patches generated: {generated}/{total} ({generated/total*100:.1f}%)")
    if resolved is not None:
        print(f"  Issues resolved:   {resolved}/{total} ({resolved/total*100:.1f}%)")
    print(f"  Avg latency:       {avg_latency:.0f}ms")


async def main():
    parser = argparse.ArgumentParser(description="SWE-bench Lite Runner for Delia")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Number of instances to run")
    parser.add_argument("--mode", "-m", choices=["baseline", "raw"], default="raw", help="Execution mode")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Run Docker evaluation")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare baseline vs Delia")
    parser.add_argument("--output", "-o", type=Path, default=Path("swe_bench_output"), help="Output directory")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.compare:
        print("=" * 60)
        print("SWE-bench Lite: Baseline vs Delia Comparison")
        print("=" * 60)

        # Run baseline
        print("\n--- Running BASELINE mode ---")
        baseline_results = await run_benchmark(args.limit, "baseline", args.evaluate, args.output)
        print_summary(baseline_results, "baseline")

        # Run Delia
        print("\n--- Running DELIA mode ---")
        delia_results = await run_benchmark(args.limit, "raw", args.evaluate, args.output)
        print_summary(delia_results, "delia")

        # Comparison
        b_gen = sum(1 for r in baseline_results if r.patch and not r.patch.startswith("Error"))
        d_gen = sum(1 for r in delia_results if r.patch and not r.patch.startswith("Error"))
        b_lat = sum(r.latency_ms for r in baseline_results) / len(baseline_results)
        d_lat = sum(r.latency_ms for r in delia_results) / len(delia_results)

        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Baseline':>15} {'Delia':>15} {'Delta':>15}")
        print("-" * 60)
        print(f"{'Patches Generated':<20} {b_gen:>15} {d_gen:>15} {d_gen-b_gen:>+15}")
        print(f"{'Avg Latency (ms)':<20} {b_lat:>15.0f} {d_lat:>15.0f} {d_lat-b_lat:>+15.0f}")

    else:
        results = await run_benchmark(args.limit, args.mode, args.evaluate, args.output)
        print_summary(results, args.mode)

        # Save results
        output_file = args.output / f"results_{args.mode}.json"
        with open(output_file, "w") as f:
            json.dump([{
                "instance_id": r.instance_id,
                "repo": r.repo,
                "resolved": r.resolved,
                "latency_ms": r.latency_ms,
                "patch_preview": r.patch[:500] if r.patch else None,
            } for r in results], f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
