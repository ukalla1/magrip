"""Evaluation helpers for MaGRIP runs."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from magrip.masks import StructuredMask, apply_structured_masks


def perplexity_from_loss(loss: float) -> float:
    """Convert average negative log-likelihood to perplexity."""

    return math.exp(loss)


@torch.no_grad()
def causal_lm_loss(
    model: object,
    input_ids: Tensor,
    labels: Tensor | None = None,
    masks: dict[str, StructuredMask] | None = None,
) -> float:
    """Evaluate causal-LM loss with optional temporary FFN masks."""

    if labels is None:
        labels = input_ids

    model.eval()
    if masks:
        with apply_structured_masks(model, masks):
            outputs = model(input_ids=input_ids, labels=labels)
    else:
        outputs = model(input_ids=input_ids, labels=labels)
    return float(outputs.loss.detach().cpu().item())
