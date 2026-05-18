import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from benchmark.metrics import normalize, exact_match, f1_score, compute_metrics

def test_normalize_lowercases():
    assert normalize("Hello World") == "hello world"

def test_normalize_removes_articles():
    result = normalize("The quick brown fox")
    assert "the" not in result.split()

def test_normalize_removes_punctuation():
    assert normalize("Hello, World!") == "hello world"

def test_exact_match_identical():
    assert exact_match("Paris", "Paris") == 1.0

def test_exact_match_normalized():
    assert exact_match("the paris", "Paris") == 1.0

def test_exact_match_different():
    assert exact_match("London", "Paris") == 0.0

def test_f1_perfect():
    assert f1_score("Neil Armstrong", "Neil Armstrong") == 1.0

def test_f1_partial():
    score = f1_score("Neil Armstrong", "Armstrong")
    assert 0.0 < score < 1.0

def test_f1_zero():
    assert f1_score("London", "Paris") == 0.0

def test_compute_metrics_basic():
    preds = ["Paris", "Neil Armstrong", "1945"]
    golds = ["Paris", "Neil Armstrong", "1944"]
    metrics = compute_metrics(preds, golds)
    assert metrics["num_examples"] == 3
    assert metrics["exact_match"] == pytest.approx(2 / 3)
