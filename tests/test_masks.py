import tempfile
from pathlib import Path

import torch
from torch import nn

from magrip.masks import (
    annealed_temperature,
    apply_structured_masks,
    infer_channel_costs,
    infer_channel_flop_costs,
    load_mask_state,
    masks_from_saliency,
    save_mask_state,
    set_mask_temperature,
    total_mask_cost,
    trainable_masks_from_saliency,
)
from magrip.topology import FFNTarget, FFNTopologyKind


def dense_target() -> FFNTarget:
    return FFNTarget(
        block_index=0,
        block_path="block",
        ffn_path="block.mlp",
        topology=FFNTopologyKind.DENSE,
        expand_module_paths=("block.mlp.c_fc",),
        contract_module_paths=("block.mlp.c_proj",),
        intermediate_size=6,
        hidden_size=4,
        registry_name="test_dense",
    )


def gated_target() -> FFNTarget:
    return FFNTarget(
        block_index=0,
        block_path="block",
        ffn_path="block.mlp",
        topology=FFNTopologyKind.GATED,
        expand_module_paths=("block.mlp.gate_proj", "block.mlp.up_proj"),
        contract_module_paths=("block.mlp.down_proj",),
        intermediate_size=6,
        hidden_size=4,
        registry_name="test_gated",
    )


class DenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(4, 6)
        self.c_proj = nn.Linear(6, 4)


class GatedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6)
        self.up_proj = nn.Linear(4, 6)
        self.down_proj = nn.Linear(6, 4)


class Block(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp


class Model(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.block = Block(mlp)


def test_masks_from_saliency_uses_target_channel_count() -> None:
    target = dense_target()
    saliency = {target.ffn_path: torch.arange(6, dtype=torch.float32)}

    masks = masks_from_saliency(saliency, [target], retained_ratio=0.5)
    mask = masks[target.ffn_path]

    assert mask.total_channels == 6
    assert mask.active_channels == 3
    assert torch.equal(mask.binary_values, torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32))


def test_dense_channel_cost_is_inferred_from_module_shapes() -> None:
    model = Model(DenseMLP())
    target = dense_target()

    costs = infer_channel_costs(model, target)
    flop_costs = infer_channel_flop_costs(model, target)

    assert costs.shape == (6,)
    assert torch.allclose(costs, torch.full((6,), 9.0))
    assert torch.allclose(flop_costs, torch.full((6,), 8.0))


def test_gated_channel_cost_includes_all_expansion_branches() -> None:
    model = Model(GatedMLP())
    target = gated_target()

    costs = infer_channel_costs(model, target)
    flop_costs = infer_channel_flop_costs(model, target)

    assert costs.shape == (6,)
    assert torch.allclose(costs, torch.full((6,), 14.0))
    assert torch.allclose(flop_costs, torch.full((6,), 12.0))


def test_apply_structured_masks_reuses_one_gated_mask_for_all_branches() -> None:
    model = Model(GatedMLP())
    target = gated_target()
    saliency = {target.ffn_path: torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)}
    masks = masks_from_saliency(saliency, [target], retained_ratio=0.5)
    x = torch.ones(1, 2, 4)

    with apply_structured_masks(model, masks):
        gate = model.block.mlp.gate_proj(x)
        up = model.block.mlp.up_proj(x)

    expected_zero = torch.zeros_like(gate[..., :3])
    assert torch.allclose(gate[..., :3], expected_zero)
    assert torch.allclose(up[..., :3], expected_zero)


def test_trainable_mask_ste_propagates_to_logits() -> None:
    target = dense_target()
    saliency = {target.ffn_path: torch.arange(6, dtype=torch.float32)}
    collection = trainable_masks_from_saliency(saliency, [target], retained_ratio=0.5)
    mask = collection.as_dict()[target.ffn_path]

    loss = mask.values.sum()
    loss.backward()

    assert mask.logits.grad is not None
    assert mask.logits.grad.shape == mask.logits.shape


def test_temperature_schedule_updates_collection_masks() -> None:
    target = dense_target()
    saliency = {target.ffn_path: torch.arange(6, dtype=torch.float32)}
    collection = trainable_masks_from_saliency(saliency, [target], retained_ratio=0.5)

    temperature = annealed_temperature(
        step=2,
        initial_temperature=1.0,
        min_temperature=0.25,
        decay=0.5,
    )
    set_mask_temperature(collection, temperature)

    assert temperature == 0.25
    assert collection.as_dict()[target.ffn_path].temperature == 0.25


def test_mask_state_round_trip_preserves_cost_and_values() -> None:
    model = Model(DenseMLP())
    target = dense_target()
    saliency = {target.ffn_path: torch.arange(6, dtype=torch.float32)}
    masks = masks_from_saliency(saliency, [target], retained_ratio=0.5, model=model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mask_state.pt"
        save_mask_state(path, masks)
        loaded = load_mask_state(path)

    original = masks[target.ffn_path]
    restored = loaded[target.ffn_path]
    assert torch.equal(original.binary_values, restored.binary_values)
    assert original.cost_summary.full_cost == restored.cost_summary.full_cost
    assert original.cost_summary.full_flop_cost == restored.cost_summary.full_flop_cost
    assert total_mask_cost(loaded).retained_ratio == original.cost_summary.retained_ratio
