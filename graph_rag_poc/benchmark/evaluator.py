from __future__ import annotations

from typing import Any, Dict, List

from .metrics import compute_metrics


class BenchmarkEvaluator:
    def __init__(self, config) -> None:
        self.config = config

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        predictions = [r["answer"] for r in results]
        ground_truths = [r["ground_truth"] for r in results]
        metrics = compute_metrics(predictions, ground_truths)

        detailed = [
            {
                "id": i,
                "question": r["question"],
                "predicted": r["answer"],
                "ground_truth": r["ground_truth"],
                "exact_match": em,
                "f1": f1,
            }
            for i, (r, em, f1) in enumerate(
                zip(results, metrics["em_per_example"], metrics["f1_per_example"])
            )
        ]
        return {
            "summary": {
                "exact_match": round(float(metrics["exact_match"]) * 100, 2),
                "f1": round(float(metrics["f1"]) * 100, 2),
                "num_examples": metrics["num_examples"],
            },
            "detailed": detailed,
        }

    def print_report(self, evaluation: Dict[str, Any]) -> None:
        s = evaluation["summary"]
        print("\n" + "=" * 54)
        print("   GRAPH RAG BENCHMARK EVALUATION RESULTS")
        print("=" * 54)
        print(f"   Examples evaluated : {s['num_examples']}")
        print(f"   Exact Match (EM)   : {s['exact_match']:.2f}%")
        print(f"   F1 Score           : {s['f1']:.2f}%")
        print("=" * 54 + "\n")
