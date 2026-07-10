"""Model inspection and FFN target discovery."""

from __future__ import annotations

from collections.abc import Sequence

from magrip.module_utils import get_module_by_path
from magrip.topology import FFNTarget, FFNTopologyKind


def discover_ffn_targets(model: object) -> Sequence[FFNTarget]:
    """Return prunable FFN targets for a model.

    M1 supports GPT-2 dense FFNs and Gemma/LLaMA/Qwen-style gated FFNs. M2 will
    generalize this into a richer topology registry.
    """

    return [
        *list(_discover_gpt2_dense_targets(model)),
        *list(_discover_decoder_gated_targets(model)),
    ]


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


def _discover_decoder_gated_targets(model: object) -> Sequence[FFNTarget]:
    layers_path = _find_decoder_layers_path(model)
    if layers_path is None:
        return []

    layers = get_module_by_path(model, layers_path)
    targets: list[FFNTarget] = []
    for block_index, block in enumerate(layers):
        mlp = getattr(block, "mlp", None)
        if (
            mlp is None
            or not hasattr(mlp, "gate_proj")
            or not hasattr(mlp, "up_proj")
            or not hasattr(mlp, "down_proj")
        ):
            continue

        block_path = f"{layers_path}.{block_index}"
        ffn_path = f"{block_path}.mlp"
        gate_path = f"{ffn_path}.gate_proj"
        up_path = f"{ffn_path}.up_proj"
        down_path = f"{ffn_path}.down_proj"

        gate_proj = get_module_by_path(model, gate_path)
        down_proj = get_module_by_path(model, down_path)
        targets.append(
            FFNTarget(
                block_index=block_index,
                block_path=block_path,
                ffn_path=ffn_path,
                topology=FFNTopologyKind.GATED,
                expand_module_paths=(gate_path, up_path),
                contract_module_paths=(down_path,),
                intermediate_size=_output_features(gate_proj),
                hidden_size=_output_features(down_proj),
            )
        )
    return targets


def _find_decoder_layers_path(model: object) -> str | None:
    candidate_paths = (
        "model.layers",
        "language_model.model.layers",
        "transformer.layers",
        "decoder.layers",
    )
    for path in candidate_paths:
        try:
            layers = get_module_by_path(model, path)
        except (AttributeError, IndexError, TypeError):
            continue
        if hasattr(layers, "__iter__"):
            return path
    return None


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
