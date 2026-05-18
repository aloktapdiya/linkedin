from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx

from ..graph.node import Node
from ..indexing.embedder import Embedder
from ..indexing.vector_store import VectorStore


class GraphRetriever:
    """Hybrid retriever: vector-similarity seed selection + BFS graph expansion."""

    def __init__(
        self,
        graph: nx.Graph,
        nodes: Dict[str, Node],
        embedder: Embedder,
        passage_store: VectorStore,
        entity_store: VectorStore,
        config,
    ) -> None:
        self.graph = graph
        self.nodes = nodes
        self.embedder = embedder
        self.passage_store = passage_store
        self.entity_store = entity_store
        self.config = config

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed_single(query)

        seed_passage_ids = [
            nid
            for nid, _ in self.passage_store.search(q_emb, top_k=self.config.top_k_passages)
        ]
        seed_entity_ids = [
            nid
            for nid, _ in self.entity_store.search(q_emb, top_k=self.config.top_k_nodes)
        ]
        all_seeds = list(set(seed_passage_ids + seed_entity_ids))

        expanded = self._bfs_expand(all_seeds, self.config.max_hops)

        passage_ids: set[str] = set(seed_passage_ids)
        for nid in expanded:
            if nid in self.nodes and self.nodes[nid].node_type == "passage":
                passage_ids.add(nid)

        results = [
            {
                "id": nid,
                "text": self.nodes[nid].text,
                "title": self.nodes[nid].metadata.get("title", ""),
                "doc_id": self.nodes[nid].doc_id,
            }
            for nid in passage_ids
            if nid in self.nodes
        ]
        return results[: self.config.top_k_passages * 2]

    def _bfs_expand(self, seeds: List[str], max_hops: int) -> List[str]:
        visited: set[str] = set(seeds)
        frontier = [s for s in seeds if s in self.graph]
        for _ in range(max_hops):
            next_frontier: List[str] = []
            for nid in frontier:
                for neighbor in self.graph.neighbors(nid):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return list(visited)
