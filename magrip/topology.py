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
class FFNTopology:
    """Static description of an FFN topology pattern."""

    name: str
    kind: FFNTopologyKind
    expand_names: tuple[str, ...]
    contract_names: tuple[str, ...]
    description: str


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
    registry_name: str | None = None


@dataclass
class DiscoveryIssue:
    """A warning or skipped module found during discovery."""

    path: str
    reason: str
    severity: str = "warning"


@dataclass
class DiscoveryReport:
    """Discovery output with targets plus skipped/diagnostic information."""

    targets: list[FFNTarget] = field(default_factory=list)
    issues: list[DiscoveryIssue] = field(default_factory=list)

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def issue_count(self) -> int:
        return len(self.issues)
