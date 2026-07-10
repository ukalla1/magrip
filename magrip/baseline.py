"""M1 baseline flow preserving the core v1 pruning behavior."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from torch import Tensor

from magrip.discovery import discover_ffn_targets
from magrip.evaluation import causal_lm_loss, perplexity_from_loss
from magrip.masks import StructuredMask, masks_from_saliency
from magrip.saliency import SaliencyResult, collect_saliency
from magrip.topology import FFNTarget


@dataclass
class FrozenPruningResult:
    """Result from a frozen-weight, one-shot MaGRIP baseline run."""

    targets: list[FFNTarget]
    saliency: SaliencyResult
    masks: dict[str, StructuredMask]
    baseline_loss: float
    masked_loss: float
    num_batches: int = 1
    num_tokens: int = 0

    @property
    def baseline_perplexity(self) -> float:
        return perplexity_from_loss(self.baseline_loss)

    @property
    def masked_perplexity(self) -> float:
        return perplexity_from_loss(self.masked_loss)


def run_frozen_pruning_baseline(
    model: object,
    input_ids: Tensor,
    retained_ratio: float,
    labels: Tensor | None = None,
    targets: list[FFNTarget] | None = None,
) -> FrozenPruningResult:
    """Run a v1-style frozen saliency and mask application pass."""

    if labels is None:
        labels = input_ids

    if targets is None:
        targets = list(discover_ffn_targets(model))
    if not targets:
        raise RuntimeError("No prunable FFN targets were discovered.")

    baseline_loss = causal_lm_loss(model, input_ids=input_ids, labels=labels)
    saliency = collect_saliency(model, targets=targets, input_ids=input_ids, labels=labels)
    masks = masks_from_saliency(
        saliency.combined(),
        targets=targets,
        retained_ratio=retained_ratio,
        model=model,
    )
    masked_loss = causal_lm_loss(model, input_ids=input_ids, labels=labels, masks=masks)

    return FrozenPruningResult(
        targets=targets,
        saliency=saliency,
        masks=masks,
        baseline_loss=baseline_loss,
        masked_loss=masked_loss,
        num_batches=1,
        num_tokens=int(input_ids.numel()),
    )


def run_frozen_pruning_baseline_on_batches(
    model: object,
    batches: Sequence[Tensor],
    retained_ratio: float,
    targets: list[FFNTarget] | None = None,
) -> FrozenPruningResult:
    """Run the frozen saliency baseline over multiple calibration batches."""

    if not batches:
        raise ValueError("batches must not be empty.")

    if targets is None:
        targets = list(discover_ffn_targets(model))
    if not targets:
        raise RuntimeError("No prunable FFN targets were discovered.")

    baseline_losses: list[float] = []
    accumulated_saliency: SaliencyResult | None = None
    for batch in batches:
        baseline_losses.append(causal_lm_loss(model, input_ids=batch, labels=batch))
        batch_saliency = collect_saliency(model, targets=targets, input_ids=batch, labels=batch)
        if accumulated_saliency is None:
            accumulated_saliency = batch_saliency
        else:
            accumulated_saliency.add_(batch_saliency)

    if accumulated_saliency is None:
        raise RuntimeError("No saliency was collected.")
    accumulated_saliency.divide_(len(batches))

    masks = masks_from_saliency(
        accumulated_saliency.combined(),
        targets=targets,
        retained_ratio=retained_ratio,
        model=model,
    )
    masked_losses = [
        causal_lm_loss(model, input_ids=batch, labels=batch, masks=masks)
        for batch in batches
    ]

    return FrozenPruningResult(
        targets=targets,
        saliency=accumulated_saliency,
        masks=masks,
        baseline_loss=sum(baseline_losses) / len(baseline_losses),
        masked_loss=sum(masked_losses) / len(masked_losses),
        num_batches=len(batches),
        num_tokens=int(sum(batch.numel() for batch in batches)),
    )
