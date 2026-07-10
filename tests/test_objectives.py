import torch

from magrip.config import ObjectiveConfig
from magrip.masks import StructuredMask
from magrip.objectives import (
    budget_penalty_schedule,
    compute_magrip_objective,
    differentiable_retained_cost_ratio,
    retained_ratio_schedule,
)
from magrip.topology import FFNTarget, FFNTopologyKind


def target() -> FFNTarget:
    return FFNTarget(
        block_index=0,
        block_path="block",
        ffn_path="block.mlp",
        topology=FFNTopologyKind.DENSE,
        expand_module_paths=("block.mlp.fc_in",),
        contract_module_paths=("block.mlp.fc_out",),
        intermediate_size=4,
        hidden_size=2,
        registry_name="toy",
    )


def test_retained_ratio_schedule_reaches_target() -> None:
    config = ObjectiveConfig(
        initial_retained_ratio=1.0,
        target_retained_ratio=0.5,
        budget_warmup_steps=10,
        initial_budget_penalty_weight=0.0,
        budget_penalty_weight=2.0,
        penalty_warmup_steps=10,
    )

    assert retained_ratio_schedule(config, 0) == 1.0
    assert retained_ratio_schedule(config, 10) == 0.5
    assert budget_penalty_schedule(config, 0) == 0.0
    assert budget_penalty_schedule(config, 10) == 2.0


def test_budget_objective_backpropagates_to_mask_logits() -> None:
    mask = StructuredMask(
        target=target(),
        logits=torch.zeros(4),
        hard=False,
        trainable=True,
        cost_per_channel=torch.ones(4),
    )
    task_loss = torch.tensor(1.0, requires_grad=True)
    config = ObjectiveConfig(target_retained_ratio=0.25, budget_penalty_weight=1.0)

    objective = compute_magrip_objective(task_loss, {mask.target.ffn_path: mask}, config)
    objective.total_loss.backward()

    assert mask.logits.grad is not None
    assert objective.retained_cost_ratio.item() == 0.5


def test_differentiable_retained_ratio_uses_channel_costs() -> None:
    mask = StructuredMask(
        target=target(),
        logits=torch.tensor([10.0, -10.0, 10.0, -10.0]),
        hard=False,
        trainable=True,
        cost_per_channel=torch.tensor([1.0, 1.0, 3.0, 3.0]),
    )

    ratio = differentiable_retained_cost_ratio({mask.target.ffn_path: mask})

    assert 0.49 < ratio.item() < 0.51
