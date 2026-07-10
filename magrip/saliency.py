"""Topology-aware magnitude and gradient-informed saliency collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import Tensor

from magrip.module_utils import get_module_by_path
from magrip.topology import FFNTarget


class SaliencyNormalization(str, Enum):
    """Normalization modes for combining saliency terms."""

    NONE = "none"
    LAYER_MEDIAN = "layer_median"
    GLOBAL_MEDIAN = "global_median"


@dataclass
class SaliencyConfig:
    """Configuration for saliency collection and score combination."""

    magnitude_weight: float = 1.0
    gradient_weight: float = 1.0
    normalization: SaliencyNormalization | str = SaliencyNormalization.LAYER_MEDIAN
    eps: float = 1e-8
    collect_branch_diagnostics: bool = True


@dataclass
class SaliencyDrift:
    """Change in saliency scores between two model states."""

    mean_relative_l2: float
    max_relative_l2: float
    mean_cosine_distance: float
    per_target: dict[str, dict[str, float]]


@dataclass
class SaliencyResult:
    """Per-target FFN saliency scores.

    ``magnitude`` and ``gradient`` are measured at the theory-level FFN intermediate
    ``u``: the tensor entering the contraction/down projection. For dense FFNs this is
    post-activation; for gated FFNs this is the post-gating product.
    """

    magnitude: dict[str, Tensor]
    gradient: dict[str, Tensor]
    branch_magnitude: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    branch_gradient: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def combined(
        self,
        magnitude_weight: float = 1.0,
        gradient_weight: float = 1.0,
        normalization: SaliencyNormalization | str = SaliencyNormalization.LAYER_MEDIAN,
        eps: float = 1e-8,
    ) -> dict[str, Tensor]:
        """Return the MaGRIP saliency score from ``docs/THEORY.tex``."""

        mode = SaliencyNormalization(normalization)
        magnitude = _normalize_scores(self.magnitude, mode, eps)
        gradient = _normalize_scores(self.gradient, mode, eps)
        return {
            key: magnitude_weight * magnitude[key] + gradient_weight * gradient[key]
            for key in self.magnitude
        }

    def combined_from_config(self, config: SaliencyConfig) -> dict[str, Tensor]:
        """Return combined scores using a config object."""

        return self.combined(
            magnitude_weight=config.magnitude_weight,
            gradient_weight=config.gradient_weight,
            normalization=config.normalization,
            eps=config.eps,
        )

    def add_(self, other: "SaliencyResult") -> None:
        """Accumulate another saliency result in place."""

        for key in self.magnitude:
            self.magnitude[key] = self.magnitude[key] + other.magnitude[key]
            self.gradient[key] = self.gradient[key] + other.gradient[key]
            for module_path in self.branch_magnitude.get(key, {}):
                self.branch_magnitude[key][module_path] = (
                    self.branch_magnitude[key][module_path]
                    + other.branch_magnitude[key][module_path]
                )
                self.branch_gradient[key][module_path] = (
                    self.branch_gradient[key][module_path]
                    + other.branch_gradient[key][module_path]
                )

    def divide_(self, value: float) -> None:
        """Scale saliency values in place."""

        if value <= 0.0:
            raise ValueError("value must be positive.")
        for key in self.magnitude:
            self.magnitude[key] = self.magnitude[key] / value
            self.gradient[key] = self.gradient[key] / value
            for module_path in self.branch_magnitude.get(key, {}):
                self.branch_magnitude[key][module_path] = (
                    self.branch_magnitude[key][module_path] / value
                )
                self.branch_gradient[key][module_path] = (
                    self.branch_gradient[key][module_path] / value
                )

    def drift_from(
        self,
        previous: "SaliencyResult",
        normalization: SaliencyNormalization | str = SaliencyNormalization.LAYER_MEDIAN,
        eps: float = 1e-8,
    ) -> SaliencyDrift:
        """Compute saliency drift against a previous state."""

        current_scores = self.combined(normalization=normalization, eps=eps)
        previous_scores = previous.combined(normalization=normalization, eps=eps)
        per_target: dict[str, dict[str, float]] = {}
        relative_l2_values: list[float] = []
        cosine_distance_values: list[float] = []
        for key, current in current_scores.items():
            prior = previous_scores[key].to(device=current.device, dtype=current.dtype)
            relative_l2 = _relative_l2(current, prior, eps)
            cosine_distance = _cosine_distance(current, prior, eps)
            per_target[key] = {
                "relative_l2": relative_l2,
                "cosine_distance": cosine_distance,
            }
            relative_l2_values.append(relative_l2)
            cosine_distance_values.append(cosine_distance)

        return SaliencyDrift(
            mean_relative_l2=_mean(relative_l2_values),
            max_relative_l2=max(relative_l2_values) if relative_l2_values else 0.0,
            mean_cosine_distance=_mean(cosine_distance_values),
            per_target=per_target,
        )

    def summaries(self) -> dict[str, dict[str, Any]]:
        """Return lightweight metadata useful for run logs."""

        return {
            key: {
                "channels": int(self.magnitude[key].numel()),
                "source": self.metadata.get(key, {}).get("source"),
                "branch_count": len(self.branch_magnitude.get(key, {})),
            }
            for key in self.magnitude
        }


@dataclass
class SaliencyRefreshSchedule:
    """Schedule describing when a training loop should recompute saliency."""

    every_steps: int
    start_step: int = 0

    def should_recompute(self, step: int) -> bool:
        """Return whether saliency should be recomputed at ``step``."""

        if self.every_steps <= 0:
            return False
        if step < self.start_step:
            return False
        return (step - self.start_step) % self.every_steps == 0


@dataclass
class SaliencyTracker:
    """Small state holder for saliency recomputation during joint training."""

    schedule: SaliencyRefreshSchedule
    current: SaliencyResult | None = None
    last_step: int | None = None
    last_drift: SaliencyDrift | None = None

    def should_recompute(self, step: int) -> bool:
        """Return whether a new saliency pass should run."""

        return self.current is None or self.schedule.should_recompute(step)

    def update(self, saliency: SaliencyResult, step: int) -> SaliencyDrift | None:
        """Store a new saliency result and return drift from the previous result."""

        drift = saliency.drift_from(self.current) if self.current is not None else None
        self.current = saliency
        self.last_step = step
        self.last_drift = drift
        return drift


def collect_saliency(
    model: object,
    targets: list[FFNTarget],
    input_ids: Tensor,
    labels: Tensor | None = None,
    config: SaliencyConfig | None = None,
) -> SaliencyResult:
    """Collect activation magnitude and first-order gradient saliency.

    The main saliency signal is measured on the tensor entering each contraction module.
    This matches the structured FFN unit ``u`` used in the theory document.
    """

    if labels is None:
        labels = input_ids
    if config is None:
        config = SaliencyConfig()

    model.train(False)
    original_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(True)

    intermediate_activations: dict[tuple[str, str], Tensor] = {}
    branch_activations: dict[tuple[str, str], Tensor] = {}
    handles = []

    try:
        for target in targets:
            for module_path in target.contract_module_paths:
                module = get_module_by_path(model, module_path)
                handles.append(
                    module.register_forward_pre_hook(
                        _make_contract_pre_hook(
                            target_key=target.ffn_path,
                            module_path=module_path,
                            activations=intermediate_activations,
                        )
                    )
                )
            if config.collect_branch_diagnostics:
                for module_path in target.expand_module_paths:
                    module = get_module_by_path(model, module_path)
                    handles.append(
                        module.register_forward_hook(
                            _make_branch_hook(
                                target_key=target.ffn_path,
                                module_path=module_path,
                                activations=branch_activations,
                            )
                        )
                    )

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        return _build_saliency_result(
            targets=targets,
            intermediate_activations=intermediate_activations,
            branch_activations=branch_activations,
        )
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for param, requires_grad in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(requires_grad)


def channel_magnitude(activation: Tensor) -> Tensor:
    """Compute ``||u_i||_2`` over all non-channel dimensions."""

    return activation.detach().norm(dim=_reduction_dims(activation))


def channel_first_order_saliency(activation: Tensor, gradient: Tensor) -> Tensor:
    """Compute ``|<dL/du_i, u_i>|`` over all non-channel dimensions."""

    return (activation.detach() * gradient.detach()).sum(dim=_reduction_dims(activation)).abs()


def branch_proxy_saliency(activation: Tensor, gradient: Tensor) -> tuple[Tensor, Tensor]:
    """Return branch-level magnitude and first-order diagnostics."""

    return channel_magnitude(activation), channel_first_order_saliency(activation, gradient)


def _make_contract_pre_hook(
    target_key: str,
    module_path: str,
    activations: dict[tuple[str, str], Tensor],
):
    def hook(module: object, inputs: tuple[object, ...]) -> None:
        activation = _first_tensor_input(inputs, module_path)
        activation.retain_grad()
        activations[(target_key, module_path)] = activation

    return hook


def _make_branch_hook(
    target_key: str,
    module_path: str,
    activations: dict[tuple[str, str], Tensor],
):
    def hook(module: object, inputs: tuple[object, ...], output: Tensor) -> Tensor:
        output.retain_grad()
        activations[(target_key, module_path)] = output
        return output

    return hook


def _build_saliency_result(
    targets: list[FFNTarget],
    intermediate_activations: dict[tuple[str, str], Tensor],
    branch_activations: dict[tuple[str, str], Tensor],
) -> SaliencyResult:
    magnitude: dict[str, Tensor] = {}
    gradient: dict[str, Tensor] = {}
    branch_magnitude: dict[str, dict[str, Tensor]] = {}
    branch_gradient: dict[str, dict[str, Tensor]] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for target in targets:
        target_magnitude = []
        target_gradient = []
        sources = []
        for module_path in target.contract_module_paths:
            activation = intermediate_activations[(target.ffn_path, module_path)]
            grad = activation.grad
            if grad is None:
                raise RuntimeError(
                    f"No intermediate gradient captured for {target.ffn_path} "
                    f"at {module_path}."
                )
            target_magnitude.append(channel_magnitude(activation))
            target_gradient.append(channel_first_order_saliency(activation, grad))
            sources.append(module_path)

        magnitude[target.ffn_path] = _mean_stack(target_magnitude)
        gradient[target.ffn_path] = _mean_stack(target_gradient)
        _validate_channel_count(target, magnitude[target.ffn_path])
        metadata[target.ffn_path] = {
            "source": "contract_input",
            "contract_module_paths": sources,
        }

        branch_magnitude[target.ffn_path] = {}
        branch_gradient[target.ffn_path] = {}
        for module_path in target.expand_module_paths:
            key = (target.ffn_path, module_path)
            if key not in branch_activations:
                continue
            activation = branch_activations[key]
            grad = activation.grad
            if grad is None:
                continue
            branch_mag, branch_grad = branch_proxy_saliency(activation, grad)
            branch_magnitude[target.ffn_path][module_path] = branch_mag
            branch_gradient[target.ffn_path][module_path] = branch_grad

    return SaliencyResult(
        magnitude=magnitude,
        gradient=gradient,
        branch_magnitude=branch_magnitude,
        branch_gradient=branch_gradient,
        metadata=metadata,
    )


def _normalize_scores(
    scores: dict[str, Tensor],
    normalization: SaliencyNormalization,
    eps: float,
) -> dict[str, Tensor]:
    if normalization == SaliencyNormalization.NONE:
        return {key: value for key, value in scores.items()}
    if normalization == SaliencyNormalization.LAYER_MEDIAN:
        return {key: _median_normalize(value, eps) for key, value in scores.items()}
    if normalization == SaliencyNormalization.GLOBAL_MEDIAN:
        denominator = torch.cat([value.detach().flatten() for value in scores.values()])
        scale = torch.median(denominator).clamp_min(eps)
        return {
            key: value / scale.to(device=value.device, dtype=value.dtype)
            for key, value in scores.items()
        }
    raise ValueError(f"Unsupported normalization mode: {normalization}.")


def _median_normalize(values: Tensor, eps: float) -> Tensor:
    denominator = torch.median(values.detach()).clamp_min(eps)
    return values / denominator.to(device=values.device, dtype=values.dtype)


def _first_tensor_input(inputs: tuple[object, ...], module_path: str) -> Tensor:
    if not inputs:
        raise RuntimeError(f"No inputs captured for {module_path}.")
    activation = inputs[0]
    if not isinstance(activation, Tensor):
        raise RuntimeError(f"First input to {module_path} is not a tensor.")
    return activation


def _reduction_dims(tensor: Tensor) -> tuple[int, ...]:
    if tensor.ndim <= 1:
        return ()
    return tuple(range(tensor.ndim - 1))


def _mean_stack(values: list[Tensor]) -> Tensor:
    if not values:
        raise RuntimeError("Expected at least one saliency tensor.")
    if len(values) == 1:
        return values[0]
    return torch.stack(values, dim=0).mean(dim=0)


def _validate_channel_count(target: FFNTarget, scores: Tensor) -> None:
    if target.intermediate_size is None:
        return
    if int(scores.numel()) != int(target.intermediate_size):
        raise RuntimeError(
            f"Saliency for {target.ffn_path} has {scores.numel()} channels, "
            f"expected {target.intermediate_size}."
        )


def _relative_l2(current: Tensor, previous: Tensor, eps: float) -> float:
    numerator = torch.linalg.vector_norm(current.detach() - previous.detach())
    denominator = torch.linalg.vector_norm(previous.detach()).clamp_min(eps)
    return float((numerator / denominator).detach().cpu().item())


def _cosine_distance(current: Tensor, previous: Tensor, eps: float) -> float:
    current_flat = current.detach().flatten()
    previous_flat = previous.detach().flatten()
    numerator = torch.dot(current_flat, previous_flat)
    denominator = (
        torch.linalg.vector_norm(current_flat)
        * torch.linalg.vector_norm(previous_flat)
    ).clamp_min(eps)
    cosine = torch.clamp(numerator / denominator, min=-1.0, max=1.0)
    return float((1.0 - cosine).detach().cpu().item())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
