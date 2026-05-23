from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .graph.builder import GraphBuilder
from .generation.generator import Generator
from .generation.providers.factory import create_provider, is_ollama_running
from .indexing.embedder import Embedder
from .indexing.vector_store import VectorStore
from .retrieval.graph_retriever import GraphRetriever

_NER_PROMPT = (
    "Extract all named entities from the text below. "
    "Named entities include: people, places, organizations, products, and events.\n"
    "Return ONLY a comma-separated list of entity names — no explanations, no numbering.\n\n"
    "Text: {text}\n\n"
    "Entities:"
)


class GraphRAGPipeline:
    """End-to-end Graph RAG pipeline: build index, then query."""

    def __init__(self, config) -> None:
        self.config = config
        self.embedder = Embedder(model_name=config.embedding_model)
        self.passage_store = VectorStore(dim=config.embedding_dim)
        self.entity_store = VectorStore(dim=config.embedding_dim)
        self.generator = Generator(config)
        self._retriever: Optional[GraphRetriever] = None

    def _make_llm_extractor(self, provider) -> Callable[[str], List[str]]:
        """Build an entity-extraction callable backed by an LLM prompt."""

        def extract(text: str) -> List[str]:
            prompt = _NER_PROMPT.format(text=text)
            raw = provider.generate(prompt, max_tokens=128, temperature=0.0)
            entities = [
                e.strip().strip('"').strip("'").strip("-").strip()
                for e in raw.split(",")
            ]
            return [e for e in entities if 2 < len(e) < 80]

        return extract

    def build(self, passages: List[Dict[str, Any]]) -> None:
        """Index passages into the knowledge graph and vector stores."""
        print(f"[Pipeline] Building graph index from {len(passages)} passages ...")

        llm_extractor: Optional[Callable[[str], List[str]]] = None
        if self.config.use_llm_entity_extraction and is_ollama_running(
            self.config.ollama_base_url
        ):
            provider = self.generator._get_provider()
            print(f"[Pipeline] LLM entity extraction via {provider.name}")
            llm_extractor = self._make_llm_extractor(provider)
        else:
            print("[Pipeline] Regex entity extraction (Ollama not detected locally)")

        graph_builder = GraphBuilder(self.config, llm_extractor=llm_extractor)
        graph = graph_builder.build_from_passages(passages)
        nodes = graph_builder.nodes
        print(
            f"[Pipeline]   Graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        passage_nodes = graph_builder.get_passage_nodes()
        if passage_nodes:
            embs = self.embedder.embed([n.text for n in passage_nodes])
            for node, emb in zip(passage_nodes, embs):
                node.embedding = emb.tolist()
                self.passage_store.add(node.id, emb, metadata=node.metadata)
        print(f"[Pipeline]   Indexed {len(passage_nodes)} passage nodes.")

        entity_nodes = graph_builder.get_entity_nodes()
        if entity_nodes:
            embs = self.embedder.embed([n.text for n in entity_nodes])
            for node, emb in zip(entity_nodes, embs):
                node.embedding = emb.tolist()
                self.entity_store.add(node.id, emb, metadata=node.metadata)
        print(f"[Pipeline]   Indexed {len(entity_nodes)} entity nodes.")

        self._retriever = GraphRetriever(
            graph=graph,
            nodes=nodes,
            embedder=self.embedder,
            passage_store=self.passage_store,
            entity_store=self.entity_store,
            config=self.config,
        )
        print("[Pipeline] Build complete.")

    def query(self, question: str) -> Dict[str, Any]:
        if self._retriever is None:
            raise RuntimeError("Pipeline not built. Call build() first.")
        context = self._retriever.retrieve(question)
        answer = self.generator.generate(question, context)
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "num_passages_retrieved": len(context),
        }
