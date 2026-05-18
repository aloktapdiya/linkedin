import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Anthropic API
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model_id: str = "claude-sonnet-4-6"

    # Embedding model (sentence-transformers)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Graph construction
    min_edge_weight: float = 0.3

    # Retrieval
    top_k_nodes: int = 5
    top_k_passages: int = 3
    max_hops: int = 2

    # Generation
    max_tokens: int = 512
    temperature: float = 0.0

    # Benchmark
    benchmark_subset_size: int = 20

    # Paths
    data_dir: str = "data"
    benchmark_dir: str = "data/benchmark"
    results_dir: str = "results"
