"""Model inspection and FFN target discovery."""

from __future__ import annotations

from collections.abc import Sequence

from magrip.module_utils import get_module_by_path
from magrip.topology import FFNTarget, FFNTopologyKind


def discover_ffn_targets(model: object) -> Sequence[FFNTarget]:
    """Return prunable FFN targets for a model.

    M1 intentionally supports the GPT-2 dense-FFN path first. M2 will generalize this
    into a richer topology registry.
    """

    return list(_discover_gpt2_dense_targets(model))


def _discover_gpt2_dense_targets(model: object) -> Sequence[FFNTarget]:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        return []

    targets: list[FFNTarget] = []
    for block_index, block in enumerate(model.transformer.h):
        mlp = getattr(block, "mlp", None)
        if mlp is None or not hasattr(mlp, "c_fc") or not hasattr(mlp, "c_proj"):
            continue

        block_path = f"transformer.h.{block_index}"
        ffn_path = f"{block_path}.mlp"
        c_fc_path = f"{ffn_path}.c_fc"
        c_proj_path = f"{ffn_path}.c_proj"

        c_fc = get_module_by_path(model, c_fc_path)
        c_proj = get_module_by_path(model, c_proj_path)
        targets.append(
            FFNTarget(
                block_index=block_index,
                block_path=block_path,
                ffn_path=ffn_path,
                topology=FFNTopologyKind.DENSE,
                expand_module_paths=(c_fc_path,),
                contract_module_paths=(c_proj_path,),
                intermediate_size=_output_features(c_fc),
                hidden_size=_output_features(c_proj),
            )
        )
    return targets


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
