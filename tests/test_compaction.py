import copy

import torch
from torch import nn

from magrip.compaction import compact_model_inplace, compaction_available
from magrip.masks import StructuredMask, apply_structured_masks
from magrip.topology import FFNTarget, FFNTopologyKind


class DenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(4, 6)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(6, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class GatedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6)
        self.up_proj = nn.Linear(4, 6)
        self.down_proj = nn.Linear(6, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Model(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.block = Block(mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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


def test_compaction_available() -> None:
    assert compaction_available()


def test_dense_compaction_matches_masked_model() -> None:
    torch.manual_seed(0)
    model = Model(DenseMLP()).eval()
    compacted = copy.deepcopy(model)
    target = dense_target()
    mask = StructuredMask.from_binary_values(
        target,
        torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.float32),
    )
    x = torch.randn(2, 3, 4)

    with apply_structured_masks(model, {target.ffn_path: mask}):
        expected = model(x)
    report = compact_model_inplace(compacted, {target.ffn_path: mask})
    actual = compacted(x)

    assert report.retained_channels == 3
    assert compacted.block.mlp.c_fc.out_features == 3
    assert compacted.block.mlp.c_proj.in_features == 3
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_gated_compaction_matches_masked_model() -> None:
    torch.manual_seed(0)
    model = Model(GatedMLP()).eval()
    compacted = copy.deepcopy(model)
    target = gated_target()
    mask = StructuredMask.from_binary_values(
        target,
        torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.float32),
    )
    x = torch.randn(2, 3, 4)

    with apply_structured_masks(model, {target.ffn_path: mask}):
        expected = model(x)
    report = compact_model_inplace(compacted, {target.ffn_path: mask})
    actual = compacted(x)

    assert report.retained_channels == 3
    assert compacted.block.mlp.gate_proj.out_features == 3
    assert compacted.block.mlp.up_proj.out_features == 3
    assert compacted.block.mlp.down_proj.in_features == 3
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
