from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def _tokens(s: str) -> List[str]:
    return normalize(s).split()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: List[str], ground_truths: List[str]
) -> Dict[str, object]:
    assert len(predictions) == len(ground_truths)
    em_list = [exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    f1_list = [f1_score(p, g) for p, g in zip(predictions, ground_truths)]
    return {
        "exact_match": sum(em_list) / len(em_list) if em_list else 0.0,
        "f1": sum(f1_list) / len(f1_list) if f1_list else 0.0,
        "num_examples": len(predictions),
        "em_per_example": em_list,
        "f1_per_example": f1_list,
    }
