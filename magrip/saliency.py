"""Magnitude and gradient-informed saliency collection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from magrip.module_utils import get_module_by_path
from magrip.topology import FFNTarget


@dataclass
class SaliencyResult:
    """Per-target FFN saliency scores."""

    magnitude: dict[str, Tensor]
    gradient: dict[str, Tensor]

    def combined(
        self,
        magnitude_weight: float = 1.0,
        gradient_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        """Return layer-normalized magnitude-plus-gradient saliency."""

        scores: dict[str, Tensor] = {}
        for key in self.magnitude:
            mag = _median_normalize(self.magnitude[key])
            grad = _median_normalize(self.gradient[key])
            scores[key] = magnitude_weight * mag + gradient_weight * grad
        return scores

    def add_(self, other: "SaliencyResult") -> None:
        """Accumulate another saliency result in place."""

        for key in self.magnitude:
            self.magnitude[key] = self.magnitude[key] + other.magnitude[key]
            self.gradient[key] = self.gradient[key] + other.gradient[key]

    def divide_(self, value: float) -> None:
        """Scale saliency values in place."""

        for key in self.magnitude:
            self.magnitude[key] = self.magnitude[key] / value
            self.gradient[key] = self.gradient[key] / value


def collect_saliency(
    model: object,
    targets: list[FFNTarget],
    input_ids: Tensor,
    labels: Tensor | None = None,
) -> SaliencyResult:
    """Collect v1-style activation magnitude and gradient saliency.

    This is a frozen-weight saliency pass: gradients are computed for sensitivity, but no
    optimizer step is taken.
    """

    if labels is None:
        labels = input_ids

    model.train(False)
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(True)

    activations: dict[tuple[str, str], Tensor] = {}
    handles = []

    def make_hook(target_key: str, module_path: str):
        def hook(module: object, inputs: tuple[object, ...], output: Tensor) -> Tensor:
            output.retain_grad()
            activations[(target_key, module_path)] = output
            return output

        return hook

    try:
        for target in targets:
            for module_path in target.expand_module_paths:
                module = get_module_by_path(model, module_path)
                handles.append(
                    module.register_forward_hook(make_hook(target.ffn_path, module_path))
                )

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        magnitude: dict[str, Tensor] = {}
        gradient: dict[str, Tensor] = {}
        for target in targets:
            branch_magnitude = []
            branch_gradient = []
            for module_path in target.expand_module_paths:
                activation = activations[(target.ffn_path, module_path)]
                grad = activation.grad
                if grad is None:
                    raise RuntimeError(
                        f"No activation gradient captured for {target.ffn_path} "
                        f"branch {module_path}."
                    )
                branch_magnitude.append(activation.detach().norm(dim=(0, 1)))
                branch_gradient.append(
                    (activation.detach() * grad.detach()).abs().sum(dim=(0, 1))
                )
            magnitude[target.ffn_path] = torch.stack(branch_magnitude, dim=0).mean(dim=0)
            gradient[target.ffn_path] = torch.stack(branch_gradient, dim=0).mean(dim=0)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for param, requires_grad in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(requires_grad)

    return SaliencyResult(magnitude=magnitude, gradient=gradient)


def _median_normalize(values: Tensor) -> Tensor:
    denominator = torch.median(values.detach()).clamp_min(1e-8)
    return values / denominator
