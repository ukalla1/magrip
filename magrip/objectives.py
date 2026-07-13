"""Budget-aware MaGRIP objective terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from magrip.config import ObjectiveConfig
from magrip.masks import MaskCollection, StructuredMask


@dataclass
class ObjectiveBreakdown:
    """Individual terms of the MaGRIP objective."""

    total_loss: Tensor
    task_loss: Tensor
    budget_penalty: Tensor
    mask_regularization: Tensor
    distillation_loss: Tensor
    retained_cost_ratio: Tensor
    budget_error: Tensor
    mask_entropy: Tensor
    target_retained_ratio: float
    budget_penalty_weight: float

    def detached(self) -> dict[str, float]:
        """Return scalar values for logging."""

        return {
            "total_loss": _to_float(self.total_loss),
            "task_loss": _to_float(self.task_loss),
            "budget_penalty": _to_float(self.budget_penalty),
            "mask_regularization": _to_float(self.mask_regularization),
            "distillation_loss": _to_float(self.distillation_loss),
            "retained_cost_ratio": _to_float(self.retained_cost_ratio),
            "budget_error": _to_float(self.budget_error),
            "mask_entropy": _to_float(self.mask_entropy),
            "target_retained_ratio": float(self.target_retained_ratio),
            "budget_penalty_weight": float(self.budget_penalty_weight),
        }


def compute_magrip_objective(
    task_loss: Tensor,
    masks: Mapping[str, StructuredMask] | MaskCollection,
    config: ObjectiveConfig,
    step: int = 0,
    student_logits: Tensor | None = None,
    teacher_logits: Tensor | None = None,
) -> ObjectiveBreakdown:
    """Compute the M5 budget-aware objective from ``docs/THEORY.tex``."""

    target_ratio = retained_ratio_schedule(config, step)
    penalty_weight = budget_penalty_schedule(config, step)
    retained_ratio = differentiable_retained_cost_ratio(masks, like=task_loss)
    budget_error = retained_ratio - target_ratio
    budget_penalty = penalty_weight * budget_error.pow(2)
    entropy = mask_entropy_regularization(
        masks,
        like=task_loss,
    )
    regularization = config.mask_regularization_weight * entropy
    distillation = config.distillation_weight * distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=config.distillation_temperature,
        mode=config.distillation_mode,
        like=task_loss,
    )
    total = task_loss + budget_penalty + regularization + distillation
    return ObjectiveBreakdown(
        total_loss=total,
        task_loss=task_loss,
        budget_penalty=budget_penalty,
        mask_regularization=regularization,
        distillation_loss=distillation,
        retained_cost_ratio=retained_ratio,
        budget_error=budget_error,
        mask_entropy=entropy,
        target_retained_ratio=target_ratio,
        budget_penalty_weight=penalty_weight,
    )


def distillation_is_enabled(config: ObjectiveConfig) -> bool:
    """Return whether optional distillation should be used."""

    return config.distillation_weight > 0.0 and config.distillation_mode != "disabled"


def retained_ratio_schedule(config: ObjectiveConfig, step: int) -> float:
    """Anneal retained budget from initial ratio to target ratio."""

    if config.budget_warmup_steps <= 0:
        return float(config.target_retained_ratio)
    progress = min(1.0, max(0.0, step / float(config.budget_warmup_steps)))
    return float(
        config.initial_retained_ratio
        + progress * (config.target_retained_ratio - config.initial_retained_ratio)
    )


def budget_penalty_schedule(config: ObjectiveConfig, step: int) -> float:
    """Anneal budget pressure from initial to final penalty weight."""

    if config.penalty_warmup_steps <= 0:
        return float(config.budget_penalty_weight)
    progress = min(1.0, max(0.0, step / float(config.penalty_warmup_steps)))
    return float(
        config.initial_budget_penalty_weight
        + progress * (config.budget_penalty_weight - config.initial_budget_penalty_weight)
    )


def differentiable_retained_cost_ratio(
    masks: Mapping[str, StructuredMask] | MaskCollection,
    like: Tensor | None = None,
) -> Tensor:
    """Return ``Cost(q) / Cost(1)`` using relaxed mask probabilities."""

    mask_map = masks.as_dict() if isinstance(masks, MaskCollection) else masks
    if not mask_map:
        raise ValueError("masks must not be empty.")

    retained_terms = []
    full_terms = []
    for mask in mask_map.values():
        probabilities = mask.probabilities
        costs = mask.cost_per_channel.to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        retained_terms.append(torch.sum(probabilities * costs))
        full_terms.append(torch.sum(costs))

    retained = torch.stack(retained_terms).sum()
    full = torch.stack(full_terms).sum().clamp_min(1e-12)
    ratio = retained / full
    if like is not None:
        ratio = ratio.to(device=like.device, dtype=like.dtype)
    return ratio


def mask_entropy_regularization(
    masks: Mapping[str, StructuredMask] | MaskCollection,
    like: Tensor | None = None,
) -> Tensor:
    """Return ``sum q(1-q) / N`` so minimizing encourages binary probabilities."""

    mask_map = masks.as_dict() if isinstance(masks, MaskCollection) else masks
    terms = []
    for mask in mask_map.values():
        probabilities = mask.probabilities
        terms.append(torch.mean(probabilities * (1.0 - probabilities)))
    if not terms:
        raise ValueError("masks must not be empty.")
    regularization = torch.stack(terms).mean()
    if like is not None:
        regularization = regularization.to(device=like.device, dtype=like.dtype)
    return regularization


def distillation_loss(
    student_logits: Tensor | None,
    teacher_logits: Tensor | None,
    temperature: float,
    mode: str,
    like: Tensor,
) -> Tensor:
    """Compute optional distillation loss.

    M5 supports the hardware-friendly disabled mode and cached-logit mode. More elaborate
    hidden-state or EMA-teacher modes remain design space for later experiments.
    """

    if mode == "disabled":
        return like.new_zeros(())
    if mode not in {"cached_logits", "teacher_logits"}:
        raise ValueError(f"Unsupported distillation mode: {mode!r}.")
    if student_logits is None or teacher_logits is None:
        raise ValueError("Distillation requires both student_logits and teacher_logits.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    student = student_logits / temperature
    teacher = teacher_logits.to(device=student.device, dtype=student.dtype) / temperature
    return (
        F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


def _to_float(value: Tensor) -> float:
    return float(value.detach().cpu().item())
