from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Node:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    node_type: str = "entity"  # "entity" | "passage"
    doc_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.id == other.id
