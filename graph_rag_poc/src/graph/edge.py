from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Edge:
    source_id: str = ""
    target_id: str = ""
    relation: str = "related_to"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
