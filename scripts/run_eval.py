"""
CLI entry point for the evaluation framework.

Usage:
  python scripts/run_eval.py               # k=5, summary table only
  python scripts/run_eval.py --k 10        # k=10
  python scripts/run_eval.py --verbose     # include per-query breakdown
  python scripts/run_eval.py --mode hybrid+rerank  # run a single mode

Prerequisites:
  1. Run docker-compose up and wait for the worker to index articles.
  2. Open evaluation/test_queries.json.
  3. Find relevant URLs for each query in the database:
        SELECT url, title FROM articles WHERE title ILIKE '%elden ring%';
  4. Populate "relevant_urls" in the JSON file.
  5. Run this script.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path to ensure lore/ and evaluation/ modules can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.runner import EvalRunner, print_results
from retrieval_lab.config import settings


async def main(k: int, verbose: bool, mode: str | None) -> None:
    runner = EvalRunner()
    modes = runner.get_modes()

    if mode:
        if mode not in modes:
            print(f"❌ Unknown mode '{mode}'. Available modes: {', '.join(modes)}")
            sys.exit(1)

        # Run evaluation for a single specified mode
        metrics = await runner.run_single_mode(mode=mode, k=k)
        if metrics:
            print_results([metrics], verbose=verbose)
    else:
        # Run evaluation across all modes
        all_metrics = await runner.run(k=k)
        print_results(all_metrics, verbose=verbose)

AVAILABLE_MODES = ["vector", "keyword", "hybrid", "hybrid+rerank"]
if __name__ == "__main__":
    # Temporary holder to fetch modes for help text generation
    temp_runner = EvalRunner()
    available_modes = list(temp_runner.get_modes().keys())

    parser = argparse.ArgumentParser(description="Lore retrieval evaluation runner.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff rank k (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-query metrics breakdown")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help=f"Evaluate a single specific mode: {', '.join(available_modes)}"
    )
    args = parser.parse_args()

    asyncio.run(main(k=args.k, verbose=args.verbose, mode=args.mode))