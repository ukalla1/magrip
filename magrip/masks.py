"""Mask parameterization and structured FFN mask utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor
from torch.autograd import Function

from magrip.module_utils import get_module_by_path
from magrip.topology import FFNTarget


class BinaryMaskSTE(Function):
    """Straight-through estimator for hard binary masks."""

    @staticmethod
    def forward(ctx: object, scores: Tensor, threshold: Tensor) -> Tensor:
        return torch.where(scores >= threshold, torch.ones_like(scores), torch.zeros_like(scores))

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        return torch.clamp(grad_output, min=-1.0, max=1.0), None


@dataclass
class StructuredMask:
    """A binary channel mask associated with one FFN target."""

    target: FFNTarget
    values: Tensor

    @property
    def active_channels(self) -> int:
        return int(self.values.detach().sum().item())

    @property
    def total_channels(self) -> int:
        return int(self.values.numel())

    @property
    def retained_ratio(self) -> float:
        if self.total_channels == 0:
            return 0.0
        return self.active_channels / self.total_channels


def masks_from_saliency(
    saliency_by_target: Mapping[str, Tensor],
    targets: Sequence[FFNTarget],
    retained_ratio: float,
) -> dict[str, StructuredMask]:
    """Create top-k binary masks from per-channel saliency scores."""

    if not 0.0 < retained_ratio <= 1.0:
        raise ValueError("retained_ratio must be in (0, 1].")

    masks: dict[str, StructuredMask] = {}
    for target in targets:
        scores = saliency_by_target[target.ffn_path].detach()
        channels = int(scores.numel())
        keep = max(1, int(round(channels * retained_ratio)))
        topk = torch.topk(scores, k=keep, largest=True).indices
        values = torch.zeros_like(scores)
        values[topk] = 1.0
        masks[target.ffn_path] = StructuredMask(target=target, values=values)
    return masks


@contextmanager
def apply_structured_masks(
    model: object,
    masks: Mapping[str, StructuredMask],
) -> Iterator[None]:
    """Temporarily apply FFN channel masks to expansion outputs."""

    handles = []

    def make_hook(mask: Tensor):
        def hook(module: object, inputs: tuple[object, ...], output: Tensor) -> Tensor:
            broadcast = _broadcast_mask(mask.to(device=output.device, dtype=output.dtype), output)
            return output * broadcast

        return hook

    try:
        for structured_mask in masks.values():
            for module_path in structured_mask.target.expand_module_paths:
                module = get_module_by_path(model, module_path)
                handles.append(module.register_forward_hook(make_hook(structured_mask.values)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _broadcast_mask(mask: Tensor, output: Tensor) -> Tensor:
    view_shape = [1] * output.ndim
    view_shape[-1] = mask.numel()
    return mask.view(*view_shape)
