import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch
from config import Config
from src.pipeline import GraphRAGPipeline

SAMPLE_PASSAGES = [
    {"id": "doc_1", "title": "Apollo 11", "text": "Neil Armstrong and Buzz Aldrin landed on the Moon in 1969.", "source": "test"},
    {"id": "doc_2", "title": "Neil Armstrong", "text": "Neil Armstrong was an American astronaut who commanded Apollo 11.", "source": "test"},
    {"id": "doc_3", "title": "Moon", "text": "The Moon is Earth's only natural satellite. Apollo missions landed humans there.", "source": "test"},
]

@pytest.fixture
def config():
    cfg = Config()
    cfg.top_k_passages = 2
    cfg.top_k_nodes = 3
    cfg.max_hops = 1
    return cfg

@pytest.fixture
def pipeline(config):
    return GraphRAGPipeline(config)

def test_build_does_not_raise(pipeline):
    pipeline.build(SAMPLE_PASSAGES)
    assert pipeline._retriever is not None

def test_query_raises_before_build(pipeline):
    with pytest.raises(RuntimeError, match="build"):
        pipeline.query("Who walked on the Moon?")

def test_query_returns_expected_keys(pipeline):
    pipeline.build(SAMPLE_PASSAGES)
    with patch.object(pipeline.generator, "generate", return_value="Neil Armstrong"):
        result = pipeline.query("Who walked on the Moon?")
    for key in ("question", "answer", "context", "num_passages_retrieved"):
        assert key in result

def test_passage_store_fully_populated(pipeline):
    pipeline.build(SAMPLE_PASSAGES)
    assert len(pipeline.passage_store) == len(SAMPLE_PASSAGES)

def test_entity_store_is_populated(pipeline):
    pipeline.build(SAMPLE_PASSAGES)
    assert len(pipeline.entity_store) > 0
