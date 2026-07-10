from types import SimpleNamespace

import torch
from torch import nn

from magrip.saliency import (
    SaliencyNormalization,
    SaliencyRefreshSchedule,
    SaliencyTracker,
    channel_first_order_saliency,
    collect_saliency,
)
from magrip.topology import FFNTarget, FFNTopologyKind


def dense_target() -> FFNTarget:
    return FFNTarget(
        block_index=0,
        block_path="block",
        ffn_path="block.mlp",
        topology=FFNTopologyKind.DENSE,
        expand_module_paths=("block.mlp.fc_in",),
        contract_module_paths=("block.mlp.fc_out",),
        intermediate_size=5,
        hidden_size=3,
        registry_name="toy_dense",
    )


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_in = nn.Linear(4, 5)
        self.fc_out = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(torch.relu(self.fc_in(x)))


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = ToyMLP()


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = ToyBlock()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        logits = self.block.mlp(input_ids.float())
        return SimpleNamespace(loss=logits.pow(2).mean(), logits=logits)


def test_first_order_saliency_matches_explicit_mask_gradient() -> None:
    torch.manual_seed(0)
    activation = torch.randn(2, 3, 5, requires_grad=True)
    mask = torch.ones(5, requires_grad=True)
    masked = activation * mask.view(1, 1, -1)
    loss = masked.pow(2).sum()

    loss.backward()
    proxy = channel_first_order_saliency(activation, activation.grad)

    assert torch.allclose(proxy, mask.grad.abs(), atol=1e-6)


def test_collect_saliency_uses_contract_input_channel_shape() -> None:
    torch.manual_seed(0)
    model = ToyModel()
    target = dense_target()
    input_ids = torch.randn(2, 3, 4)

    result = collect_saliency(model, targets=[target], input_ids=input_ids)

    assert result.metadata[target.ffn_path]["source"] == "contract_input"
    assert result.magnitude[target.ffn_path].shape == (5,)
    assert result.gradient[target.ffn_path].shape == (5,)
    assert result.branch_magnitude[target.ffn_path]["block.mlp.fc_in"].shape == (5,)
    combined = result.combined(normalization=SaliencyNormalization.LAYER_MEDIAN)
    assert combined[target.ffn_path].shape == (5,)


def test_saliency_tracker_reports_drift_on_update() -> None:
    torch.manual_seed(0)
    model = ToyModel()
    target = dense_target()
    tracker = SaliencyTracker(SaliencyRefreshSchedule(every_steps=2))

    first = collect_saliency(model, targets=[target], input_ids=torch.randn(2, 3, 4))
    second = collect_saliency(model, targets=[target], input_ids=torch.randn(2, 3, 4))

    assert tracker.should_recompute(0)
    assert tracker.update(first, step=0) is None
    assert not tracker.should_recompute(1)
    assert tracker.should_recompute(2)
    drift = tracker.update(second, step=2)
    assert drift is not None
    assert target.ffn_path in drift.per_target
