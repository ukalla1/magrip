import torch
from torch import nn

from magrip.config import OptimizerConfig
from magrip.optim import build_apollo_parameter_groups


def test_build_apollo_parameter_groups_separates_matrix_and_vector_parameters() -> None:
    model = nn.Sequential(nn.Linear(4, 6), nn.LayerNorm(6))
    config = OptimizerConfig(use_apollo=True, apollo_rank=8, apollo_scale=2.0)

    groups, stats = build_apollo_parameter_groups(model.named_parameters(), config)

    assert len(groups) == 2
    assert groups[1]["rank"] == 8
    assert groups[1]["scale"] == 2.0
    assert groups[1]["scale_type"] == "channel"
    assert stats.lowrank_tensors == 1
    assert stats.regular_tensors == 3
    assert stats.adapted_parameters == sum(parameter.numel() for parameter in model.parameters())
    assert stats.lowrank_auxiliary_elements == 2 * 8 * 4
    assert stats.lowrank_projection_elements == 8 * 6
    assert stats.regular_optimizer_state_elements == 2 * 18
    assert stats.estimated_optimizer_state_elements == 64 + 48 + 36
    assert stats.estimated_optimizer_state_bytes_fp32 == 4 * (64 + 48 + 36)
    assert stats.estimated_state_ratio_vs_adamw > 0.0


def test_apollo_mini_uses_tensor_rank_one_group_defaults() -> None:
    model = nn.Linear(4, 6)
    config = OptimizerConfig(
        use_apollo=True,
        apollo_variant="apollo-mini",
        apollo_rank=256,
        apollo_scale=1.0,
        apollo_scale_type="channel",
    )

    groups, _stats = build_apollo_parameter_groups(model.named_parameters(), config)

    lowrank_group = groups[-1]
    assert lowrank_group["rank"] == 1
    assert lowrank_group["scale"] == 128.0
    assert lowrank_group["scale_type"] == "tensor"
    assert _stats.lowrank_auxiliary_elements == 2 * 1 * 4
    assert _stats.lowrank_projection_elements == 1 * 6
