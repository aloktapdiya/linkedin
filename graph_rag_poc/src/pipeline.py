from __future__ import annotations

from typing import Any, Dict, List, Optional

from .graph.builder import GraphBuilder
from .generation.generator import Generator
from .indexing.embedder import Embedder
from .indexing.vector_store import VectorStore
from .retrieval.graph_retriever import GraphRetriever


class GraphRAGPipeline:
    """End-to-end Graph RAG pipeline: build index, then query."""

    def __init__(self, config) -> None:
        self.config = config
        self.embedder = Embedder(model_name=config.embedding_model)
        self.passage_store = VectorStore(dim=config.embedding_dim)
        self.entity_store = VectorStore(dim=config.embedding_dim)
        self.graph_builder = GraphBuilder(config)
        self.generator = Generator(config)
        self._retriever: Optional[GraphRetriever] = None

    def build(self, passages: List[Dict[str, Any]]) -> None:
        """Index passages into the knowledge graph and vector store."""
        print(f"[Pipeline] Building graph index from {len(passages)} passages ...")
        graph = self.graph_builder.build_from_passages(passages)
        nodes = self.graph_builder.nodes
        print(
            f"[Pipeline]   Graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        passage_nodes = self.graph_builder.get_passage_nodes()
        if passage_nodes:
            embs = self.embedder.embed([n.text for n in passage_nodes])
            for node, emb in zip(passage_nodes, embs):
                node.embedding = emb.tolist()
                self.passage_store.add(node.id, emb, metadata=node.metadata)
        print(f"[Pipeline]   Indexed {len(passage_nodes)} passage nodes.")

        entity_nodes = self.graph_builder.get_entity_nodes()
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
