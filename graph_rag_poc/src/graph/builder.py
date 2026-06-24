from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

import networkx as nx

from .node import Node

_STOP_WORDS: frozenset[str] = frozenset({
    "The", "A", "An", "In", "On", "At", "To", "For", "Of", "And", "Or", "But",
    "Is", "Was", "Are", "Were", "Be", "Been", "Being", "Have", "Has", "Had",
    "Do", "Does", "Did", "Will", "Would", "Could", "Should", "May", "Might",
    "Must", "Can", "It", "He", "She", "They", "We", "You", "I", "That", "This",
    "These", "Those", "His", "Her", "Their", "Its", "Our", "Your", "My",
    "Which", "Who", "Whom", "Whose", "What", "Where", "When", "Why", "How",
    "Also", "However", "Although", "While", "During", "After", "Before",
    "Between", "Through", "Among", "Against", "Including",
})


class GraphBuilder:
    """
    Builds a NetworkX knowledge graph from passages.

    Entity extraction strategy:
      - LLM prompt via *llm_extractor* when Ollama is running (richer NER)
      - Regex-based capitalized phrase extraction as silent fallback
    """

    def __init__(
        self,
        config,
        llm_extractor: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.config = config
        self.llm_extractor = llm_extractor
        self.graph: nx.Graph = nx.Graph()
        self.nodes: Dict[str, Node] = {}
        self._entity_to_node_id: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_from_passages(self, passages: List[Dict]) -> nx.Graph:
        passage_nodes: List[Node] = []
        for p in passages:
            node = Node(
                text=p["text"],
                node_type="passage",
                doc_id=p.get("id"),
                metadata={"title": p.get("title", ""), "source": p.get("source", "")},
            )
            self.nodes[node.id] = node
            self.graph.add_node(node.id, text=node.text, node_type="passage")
            passage_nodes.append(node)

        for pnode in passage_nodes:
            entities = self._extract_entities(pnode.text)
            entity_ids: List[str] = []
            for ent in entities:
                ent_key = ent.lower()
                if ent_key not in self._entity_to_node_id:
                    enode = Node(
                        text=ent,
                        node_type="entity",
                        metadata={"normalized": ent_key},
                    )
                    self.nodes[enode.id] = enode
                    self.graph.add_node(enode.id, text=ent, node_type="entity")
                    self._entity_to_node_id[ent_key] = enode.id
                eid = self._entity_to_node_id[ent_key]
                entity_ids.append(eid)
                if not self.graph.has_edge(pnode.id, eid):
                    self.graph.add_edge(pnode.id, eid, relation="contains_entity", weight=1.0)

            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    u, v = entity_ids[i], entity_ids[j]
                    if u == v:
                        continue
                    if self.graph.has_edge(u, v):
                        self.graph[u][v]["weight"] = min(
                            self.graph[u][v]["weight"] + 0.1, 2.0
                        )
                    else:
                        self.graph.add_edge(u, v, relation="co_occurs", weight=0.5)

        return self.graph

    def get_passage_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == "passage"]

    def get_entity_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == "entity"]

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    # ------------------------------------------------------------------ #
    # Entity extraction
    # ------------------------------------------------------------------ #

    def _extract_entities(self, text: str) -> List[str]:
        """Try LLM extraction first; fall back silently to regex on any error."""
        if self.llm_extractor is not None:
            try:
                entities = self.llm_extractor(text)
                if entities:
                    return entities
            except Exception:
                pass
        return self._regex_extract(text)

    def _regex_extract(self, text: str) -> List[str]:
        """Lightweight capitalized-phrase NER — no external dependencies."""
        pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        matches = re.findall(pattern, text)
        return list({m for m in matches if m not in _STOP_WORDS and len(m) > 2})
