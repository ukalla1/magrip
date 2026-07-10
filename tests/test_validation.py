from magrip.topology import FFNTarget, FFNTopologyKind
from magrip.validation import validate_targets


def test_validate_dense_target_ok():
    result = validate_targets(
        [
            FFNTarget(
                block_index=0,
                block_path="transformer.h.0",
                ffn_path="transformer.h.0.mlp",
                topology=FFNTopologyKind.DENSE,
                expand_module_paths=("transformer.h.0.mlp.c_fc",),
                contract_module_paths=("transformer.h.0.mlp.c_proj",),
                intermediate_size=16,
                hidden_size=4,
            )
        ]
    )
    assert result.ok


def test_validate_gated_target_requires_multiple_branches():
    result = validate_targets(
        [
            FFNTarget(
                block_index=0,
                block_path="model.layers.0",
                ffn_path="model.layers.0.mlp",
                topology=FFNTopologyKind.GATED,
                expand_module_paths=("model.layers.0.mlp.gate_proj",),
                contract_module_paths=("model.layers.0.mlp.down_proj",),
                intermediate_size=32,
                hidden_size=8,
            )
        ]
    )
    assert not result.ok
    assert "too few expansion branches" in result.errors[0]
