import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # -------------------------------------------------------------------
    # Inference engine priority: Ollama (local) -> Gemini -> Claude
    # -------------------------------------------------------------------

    # 1. Ollama — default, local, free
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    )

    # 2. Gemini — first remote fallback
    gemini_api_key: str = field(
        default_factory=lambda: (
            os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        )
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    )

    # 3. Claude (Anthropic) — second remote fallback
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    claude_model: str = "claude-sonnet-4-6"

    # -------------------------------------------------------------------
    # Embedding (always local via sentence-transformers, unaffected by LLM)
    # -------------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # -------------------------------------------------------------------
    # Graph construction
    # -------------------------------------------------------------------
    min_edge_weight: float = 0.3
    use_llm_entity_extraction: bool = True

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------
    top_k_nodes: int = 5
    top_k_passages: int = 3
    max_hops: int = 2

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------
    max_tokens: int = 512
    temperature: float = 0.0

    # -------------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------------
    benchmark_subset_size: int = 20

    # -------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------
    data_dir: str = "data"
    benchmark_dir: str = "data/benchmark"
    results_dir: str = "results"
