#!/usr/bin/env python3
"""Graph RAG POC — entry point."""

import argparse
import json

from config import Config
from src.pipeline import GraphRAGPipeline
from benchmark.runner import BenchmarkRunner


def run_benchmark(config: Config, dataset_path: str, subset_size: int | None = None) -> None:
    pipeline = GraphRAGPipeline(config)
    runner = BenchmarkRunner(pipeline, config)
    runner.run(dataset_path, subset_size=subset_size)


def run_query(config: Config, dataset_path: str, question: str) -> None:
    with open(dataset_path) as f:
        dataset = json.load(f)

    pipeline = GraphRAGPipeline(config)
    runner = BenchmarkRunner(pipeline, config)
    passages = runner.prepare_passages(dataset[:10])
    pipeline.build(passages)

    result = pipeline.query(question)
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"\nRetrieved {result['num_passages_retrieved']} supporting passages:")
    for i, p in enumerate(result["context"], 1):
        print(f"  [{i}] {p.get('title', 'Unknown')}: {p['text'][:120]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph RAG POC with Benchmark Dataset")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "query"],
        default="benchmark",
        help="Run mode: benchmark evaluation or single query",
    )
    parser.add_argument(
        "--dataset",
        default="data/benchmark/hotpotqa_sample.json",
        help="Path to benchmark dataset JSON",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: config.benchmark_subset_size)",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to answer (query mode only)",
    )
    args = parser.parse_args()

    config = Config()

    if args.mode == "benchmark":
        run_benchmark(config, args.dataset, subset_size=args.subset)
    elif args.mode == "query":
        if not args.question:
            parser.error("--question is required in query mode")
        run_query(config, args.dataset, args.question)


if __name__ == "__main__":
    main()
