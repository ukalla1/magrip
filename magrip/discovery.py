"""Model inspection and FFN target discovery."""

from __future__ import annotations

from collections.abc import Sequence

from magrip.module_utils import get_module_by_path
from magrip.topology import DiscoveryIssue, DiscoveryReport, FFNTarget
from magrip.topology_registry import (
    iter_transformer_block_stacks,
    looks_like_moe,
    match_registered_topology,
)


def discover_ffn_targets(model: object) -> Sequence[FFNTarget]:
    """Return prunable FFN targets for a model."""

    return discover_ffn_topology(model).targets


def discover_ffn_topology(model: object) -> DiscoveryReport:
    """Discover FFN targets and topology issues for a model.

    Discovery is intentionally restricted to repeated transformer block stacks. This avoids
    pruning embeddings, LM heads, classifiers, and other non-transformer modules.
    """

    report = DiscoveryReport()
    seen_paths: set[str] = set()
    for stack_path, layers in iter_transformer_block_stacks(model):
        for block_index, block in enumerate(layers):
            block_path = f"{stack_path}.{block_index}"
            if block_path in seen_paths:
                continue
            seen_paths.add(block_path)
            _inspect_block(block=block, block_index=block_index, block_path=block_path, report=report)
    return report


def _inspect_block(
    block: object,
    block_index: int,
    block_path: str,
    report: DiscoveryReport,
) -> None:
    mlp = _find_ffn_module(block)
    if mlp is None:
        report.issues.append(
            DiscoveryIssue(path=block_path, reason="No FFN/MLP child was found in block.")
        )
        return

    ffn_path = f"{block_path}.{_ffn_child_name(block)}"
    if looks_like_moe(mlp):
        report.issues.append(
            DiscoveryIssue(
                path=ffn_path,
                reason="MoE-style FFN detected and skipped in M2.",
            )
        )
        return

    topology = match_registered_topology(mlp)
    if topology is None:
        report.issues.append(
            DiscoveryIssue(
                path=ffn_path,
                reason=f"Unsupported FFN topology on module type {type(mlp).__name__}.",
            )
        )
        return

    expand_paths = tuple(f"{ffn_path}.{name}" for name in topology.expand_names)
    contract_paths = tuple(f"{ffn_path}.{name}" for name in topology.contract_names)
    expand_modules = [get_module_by_path(mlp, name) for name in topology.expand_names]
    contract_modules = [get_module_by_path(mlp, name) for name in topology.contract_names]
    intermediate_size = _shared_output_features(expand_modules)
    hidden_size = _first_output_features(contract_modules)

    report.targets.append(
        FFNTarget(
            block_index=block_index,
            block_path=block_path,
            ffn_path=ffn_path,
            topology=topology.kind,
            expand_module_paths=expand_paths,
            contract_module_paths=contract_paths,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            registry_name=topology.name,
        )
    )


def _find_ffn_module(block: object) -> object | None:
    child_name = _ffn_child_name(block)
    if child_name:
        return getattr(block, child_name)
    return None


def _ffn_child_name(block: object) -> str | None:
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(block, name):
            return name
    return None


def _shared_output_features(modules: Sequence[object]) -> int | None:
    features = [_output_features(module) for module in modules]
    if not features or any(value is None for value in features):
        return None
    if len(set(features)) != 1:
        return None
    return features[0]


def _first_output_features(modules: Sequence[object]) -> int | None:
    if not modules:
        return None
    return _output_features(modules[0])


def _output_features(module: object) -> int | None:
    if hasattr(module, "out_features"):
        return int(module.out_features)
    if hasattr(module, "nf"):
        return int(module.nf)
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[-1])
