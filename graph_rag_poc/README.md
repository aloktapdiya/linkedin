# Graph RAG POC with Benchmark Dataset

A proof-of-concept implementation of **Graph Retrieval-Augmented Generation (Graph RAG)** evaluated against a HotpotQA-style benchmark dataset.

## Architecture

```
Documents
   |
   v
Entity Extraction  -->  Knowledge Graph (NetworkX)
                               |
                    +----------+----------+
                    |                     |
               Passage Nodes         Entity Nodes
               (vector index)        (vector index)
                    |                     |
                    +----------+----------+
                               |
                    Query  -->  Vector Similarity (seed)
                               |
                        BFS Graph Traversal
                               |
                     Retrieved Passages
                               |
                     Claude LLM  -->  Answer
```

### Module Overview

| Path | Description |
|------|-------------|
| `src/graph/builder.py` | Builds knowledge graph from passages via regex-based entity extraction |
| `src/indexing/embedder.py` | Sentence embeddings via `sentence-transformers` |
| `src/indexing/vector_store.py` | FAISS-backed cosine-similarity store (numpy fallback) |
| `src/retrieval/graph_retriever.py` | Hybrid retrieval: vector seed + BFS graph expansion |
| `src/generation/generator.py` | Answer generation via Anthropic Claude API |
| `src/pipeline.py` | End-to-end pipeline orchestrator |
| `benchmark/runner.py` | Benchmark runner for HotpotQA |
| `benchmark/evaluator.py` | Result evaluation and reporting |
| `benchmark/metrics.py` | Exact Match and F1 metric computation |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

### Run full benchmark evaluation

```bash
python main.py --mode benchmark --dataset data/benchmark/hotpotqa_sample.json
```

### Evaluate on a subset

```bash
python main.py --mode benchmark --subset 10
```

### Single question query

```bash
python main.py --mode query \
  --question "Who was the first astronaut to walk on the Moon?" \
  --dataset data/benchmark/hotpotqa_sample.json
```

### Run tests

```bash
pytest tests/ -v
```

## Benchmark Dataset

`data/benchmark/hotpotqa_sample.json` contains **20 multi-hop questions** in HotpotQA format:

- **Bridge** questions: require chaining two facts
- **Comparison** questions: compare attributes of two entities
- **Factoid** questions: single-hop factual questions

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | % of predictions that exactly match the gold answer after normalization |
| **F1 Score** | Token-level F1 between prediction and gold answer |

## Configuration (`config.py`)

```python
embedding_model = "all-MiniLM-L6-v2"  # sentence-transformers model
embedding_dim   = 384
top_k_passages  = 3                    # seed passages from vector search
top_k_nodes     = 5                    # seed entity nodes from vector search
max_hops        = 2                    # BFS depth for graph expansion
benchmark_subset_size = 20
```

## Requirements

- Python 3.10+
- `ANTHROPIC_API_KEY` environment variable
