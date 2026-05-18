import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from config import Config
from src.graph.builder import GraphBuilder

SAMPLE_PASSAGES = [
    {"id": "doc_1", "title": "Apollo 11", "text": "Neil Armstrong and Buzz Aldrin landed on the Moon on July 20, 1969. Armstrong was the first person to walk on the Moon.", "source": "test"},
    {"id": "doc_2", "title": "Neil Armstrong", "text": "Neil Armstrong was an American astronaut and the commander of Apollo 11. He was born in Wapakoneta, Ohio.", "source": "test"},
]

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def builder(config):
    return GraphBuilder(config)

def test_build_creates_correct_number_of_passage_nodes(builder):
    builder.build_from_passages(SAMPLE_PASSAGES)
    assert len(builder.get_passage_nodes()) == len(SAMPLE_PASSAGES)

def test_build_creates_entity_nodes(builder):
    builder.build_from_passages(SAMPLE_PASSAGES)
    assert len(builder.get_entity_nodes()) > 0

def test_graph_has_edges(builder):
    graph = builder.build_from_passages(SAMPLE_PASSAGES)
    assert graph.number_of_edges() > 0

def test_entity_deduplication(builder):
    builder.build_from_passages(SAMPLE_PASSAGES)
    texts_lower = [n.text.lower() for n in builder.get_entity_nodes()]
    assert texts_lower.count("neil armstrong") == 1

def test_all_nodes_have_correct_type(builder):
    builder.build_from_passages(SAMPLE_PASSAGES)
    for node in builder.get_passage_nodes():
        assert node.node_type == "passage"
    for node in builder.get_entity_nodes():
        assert node.node_type == "entity"
