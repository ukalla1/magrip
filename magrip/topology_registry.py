"""Registry of FFN topology patterns understood by MaGRIP discovery."""

from __future__ import annotations

from collections.abc import Iterable

from magrip.module_utils import get_module_by_path
from magrip.topology import FFNTopology, FFNTopologyKind


TRANSFORMER_BLOCK_STACK_PATHS: tuple[str, ...] = (
    "transformer.h",
    "model.layers",
    "language_model.model.layers",
    "transformer.layers",
    "decoder.layers",
    "gpt_neox.layers",
)

TOPOLOGY_REGISTRY: tuple[FFNTopology, ...] = (
    FFNTopology(
        name="gpt2_dense_c_fc",
        kind=FFNTopologyKind.DENSE,
        expand_names=("c_fc",),
        contract_names=("c_proj",),
        description="GPT-2 style dense FFN with c_fc expansion and c_proj contraction.",
    ),
    FFNTopology(
        name="dense_fc_in",
        kind=FFNTopologyKind.DENSE,
        expand_names=("fc_in",),
        contract_names=("fc_out",),
        description="Dense FFN with fc_in/fc_out naming.",
    ),
    FFNTopology(
        name="dense_h_to_4h",
        kind=FFNTopologyKind.DENSE,
        expand_names=("dense_h_to_4h",),
        contract_names=("dense_4h_to_h",),
        description="GPT-NeoX style dense FFN naming.",
    ),
    FFNTopology(
        name="gated_proj",
        kind=FFNTopologyKind.GATED,
        expand_names=("gate_proj", "up_proj"),
        contract_names=("down_proj",),
        description="Gemma/LLaMA/Qwen style gated FFN.",
    ),
    FFNTopology(
        name="gated_w_names",
        kind=FFNTopologyKind.GATED,
        expand_names=("w1", "w3"),
        contract_names=("w2",),
        description="SwiGLU FFN using w1/w3 expansion and w2 contraction names.",
    ),
)

MOE_MARKERS: tuple[str, ...] = (
    "experts",
    "router",
    "block_sparse_moe",
    "moe",
)

MOE_TYPE_MARKERS: tuple[str, ...] = (
    "moe",
    "mixture",
    "expert",
    "router",
)


def iter_transformer_block_stacks(model: object) -> Iterable[tuple[str, object]]:
    """Yield repeated transformer block stacks from known model layouts."""

    for path in TRANSFORMER_BLOCK_STACK_PATHS:
        try:
            stack = get_module_by_path(model, path)
        except (AttributeError, IndexError, TypeError):
            continue
        if _looks_like_stack(stack):
            yield path, stack


def match_registered_topology(mlp: object) -> FFNTopology | None:
    """Return the first registry topology matching ``mlp``."""

    for topology in TOPOLOGY_REGISTRY:
        if all(hasattr(mlp, name) for name in topology.expand_names + topology.contract_names):
            return topology
    return None


def looks_like_moe(module: object) -> bool:
    """Return whether an FFN-like module appears to be MoE."""

    names = set(dir(module))
    lowered_type = type(module).__name__.lower()
    if any(marker in lowered_type for marker in MOE_TYPE_MARKERS):
        return True
    return any(name in names for name in MOE_MARKERS)


def _looks_like_stack(value: object) -> bool:
    return hasattr(value, "__iter__") and hasattr(value, "__len__") and len(value) > 0
