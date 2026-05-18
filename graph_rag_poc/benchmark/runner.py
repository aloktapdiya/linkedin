from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.pipeline import GraphRAGPipeline
from .evaluator import BenchmarkEvaluator


class BenchmarkRunner:
    def __init__(self, pipeline: GraphRAGPipeline, config) -> None:
        self.pipeline = pipeline
        self.config = config
        self.evaluator = BenchmarkEvaluator(config)

    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        with open(path) as f:
            return json.load(f)

    def prepare_passages(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten HotpotQA-style context dicts into a passage list."""
        passages: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in dataset:
            for title, sentences in item.get("context", {}).items():
                text = " ".join(sentences)
                key = f"{title}::{text[:60]}"
                if key not in seen:
                    seen.add(key)
                    passages.append(
                        {
                            "id": f"doc_{len(passages)}",
                            "title": title,
                            "text": text,
                            "source": "hotpotqa",
                        }
                    )
        return passages

    def run(
        self, dataset_path: str, subset_size: Optional[int] = None
    ) -> Dict[str, Any]:
        print(f"\n[Runner] Loading dataset from {dataset_path} ...")
        dataset = self.load_dataset(dataset_path)

        n = subset_size if subset_size is not None else self.config.benchmark_subset_size
        dataset = dataset[:n]
        print(f"[Runner] Using {len(dataset)} examples.")

        passages = self.prepare_passages(dataset)
        self.pipeline.build(passages)

        print(f"\n[Runner] Running inference on {len(dataset)} questions ...")
        results: List[Dict[str, Any]] = []
        for item in tqdm(dataset, desc="Evaluating", unit="q"):
            out = self.pipeline.query(item["question"])
            out["ground_truth"] = item["answer"]
            results.append(out)

        evaluation = self.evaluator.evaluate(results)
        self.evaluator.print_report(evaluation)

        os.makedirs(self.config.results_dir, exist_ok=True)
        out_path = os.path.join(self.config.results_dir, "benchmark_results.json")
        with open(out_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        print(f"[Runner] Results saved -> {out_path}\n")

        return evaluation
