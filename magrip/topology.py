"""Data structures describing prunable FFN topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FFNTopologyKind(str, Enum):
    """Supported FFN topology categories."""

    DENSE = "dense"
    GATED = "gated"
    BRANCHED = "branched"
    MOE = "moe"
    UNKNOWN = "unknown"


@dataclass
class FFNTarget:
    """A structured FFN pruning target inside a transformer block."""

    block_index: int
    block_path: str
    ffn_path: str
    topology: FFNTopologyKind
    expand_module_paths: tuple[str, ...] = field(default_factory=tuple)
    contract_module_paths: tuple[str, ...] = field(default_factory=tuple)
    intermediate_size: int | None = None
    hidden_size: int | None = None
