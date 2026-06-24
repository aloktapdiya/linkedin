# Graph RAG POC with Benchmark Dataset

A proof-of-concept implementation of **Graph Retrieval-Augmented Generation (Graph RAG)**
with a smart multi-provider inference engine and a HotpotQA benchmark.

## Inference Engine — Priority Order

```
1. Ollama  (local, default)   ──► auto-detected via HTTP ping to localhost:11434
2. Gemini  (remote fallback)  ──► requires GEMINI_API_KEY or GOOGLE_API_KEY
3. Claude  (last resort)      ──► requires ANTHROPIC_API_KEY
```

The engine is selected **automatically** at runtime — no code changes needed.
If Ollama is running locally it is always preferred (free, private, no rate limits).

## Architecture

```
Documents
   │
   ▼
Entity Extraction  ─────►  LLM prompt via Ollama (when running locally)
   │                          Regex fallback (when Ollama is absent)
   ▼
Knowledge Graph (NetworkX)
   ├── Passage Nodes  ──►  FAISS vector index (sentence-transformers)
   └── Entity Nodes   ──►  FAISS vector index
          │
   Query ► Vector Similarity (seed nodes)
          │
     BFS Graph Traversal (max_hops)
          │
   Retrieved Passages
          │
   LLM Answer
     ├─ 1. Ollama  (local, preferred)
     ├─ 2. Gemini  (remote fallback)
     └─ 3. Claude  (last resort)
```

## Setup

```bash
pip install -r requirements.txt
```

Set at least one inference backend:

```bash
# Option 1 — Ollama (recommended, runs everything locally)
brew install ollama          # macOS; see https://ollama.com for Linux/Windows
ollama pull llama3.2         # or: mistral, phi3, qwen2.5, gemma2, etc.
ollama serve                 # listens on localhost:11434 by default

# Option 2 — Gemini (remote fallback)
export GEMINI_API_KEY="your-key"

# Option 3 — Claude (last resort)
export ANTHROPIC_API_KEY="your-key"
```

## Configuration (`config.py`)

| Setting | Default | Env var override |
|---------|---------|------------------|
| `ollama_base_url` | `http://localhost:11434` | `OLLAMA_BASE_URL` |
| `ollama_model` | `llama3.2` | `OLLAMA_MODEL` |
| `gemini_model` | `gemini-1.5-flash` | `GEMINI_MODEL` |
| `use_llm_entity_extraction` | `True` | — |
| `embedding_model` | `all-MiniLM-L6-v2` | — |
| `top_k_passages` | `3` | — |
| `max_hops` | `2` | — |

## Usage

```bash
# Full benchmark (20 HotpotQA examples)
python main.py --mode benchmark

# Smaller subset
python main.py --mode benchmark --subset 5

# Single question
python main.py --mode query --question "Who painted the Mona Lisa?"

# Run tests (providers are mocked — no Ollama / API key needed)
pytest tests/ -v
```

## Module Overview

| Path | Description |
|------|-------------|
| `src/generation/providers/factory.py` | Auto-detect Ollama + fallback chain |
| `src/generation/providers/ollama.py` | Ollama REST API provider |
| `src/generation/providers/gemini.py` | Google Gemini provider |
| `src/generation/providers/claude.py` | Anthropic Claude provider |
| `src/generation/generator.py` | QA generator using active provider |
| `src/graph/builder.py` | Graph builder — LLM NER or regex fallback |
| `src/indexing/vector_store.py` | FAISS cosine-similarity store |
| `src/retrieval/graph_retriever.py` | Vector seed → BFS expansion |
| `src/pipeline.py` | End-to-end orchestrator |
| `benchmark/runner.py` | HotpotQA benchmark runner |
| `benchmark/metrics.py` | Exact Match + F1 |

## Entity Extraction

When **Ollama is running locally**, graph construction uses an LLM prompt for NER:

```
Extract all named entities from the text below.
Named entities include: people, places, organizations, products, and events.
Return ONLY a comma-separated list of entity names.

Text: {passage}
Entities:
```

When Ollama is **absent**, a regex extractor silently takes over — no configuration change needed.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | % matching gold answer after normalization |
| **F1 Score** | Token-level F1 between prediction and gold |

## Requirements

- Python 3.10+
- At least one of: Ollama running locally, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`
