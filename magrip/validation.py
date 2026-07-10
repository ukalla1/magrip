"""Validation helpers for discovery targets and smoke artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from magrip.topology import FFNTarget, FFNTopologyKind


@dataclass
class ValidationResult:
    """Validation result with errors and warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def validate_targets(targets: list[FFNTarget]) -> ValidationResult:
    """Validate discovered FFN targets."""

    result = ValidationResult()
    if not targets:
        result.errors.append("No FFN targets were discovered.")
        return result

    seen_paths: set[str] = set()
    for target in targets:
        if target.ffn_path in seen_paths:
            result.errors.append(f"Duplicate target path: {target.ffn_path}")
        seen_paths.add(target.ffn_path)

        if target.topology == FFNTopologyKind.DENSE and len(target.expand_module_paths) != 1:
            result.errors.append(f"Dense target has invalid expansion count: {target.ffn_path}")
        if target.topology == FFNTopologyKind.GATED and len(target.expand_module_paths) < 2:
            result.errors.append(f"Gated target has too few expansion branches: {target.ffn_path}")
        if not target.contract_module_paths:
            result.errors.append(f"Target has no contraction path: {target.ffn_path}")
        if target.intermediate_size is None or target.intermediate_size <= 0:
            result.errors.append(f"Target has invalid intermediate size: {target.ffn_path}")
        if target.hidden_size is None or target.hidden_size <= 0:
            result.errors.append(f"Target has invalid hidden size: {target.ffn_path}")
    return result


def validate_smoke_summary(path: str | Path, expected_topology: str | None = None) -> ValidationResult:
    """Validate a MaGRIP smoke-test summary or manifest JSON."""

    summary_path = Path(path)
    summary = json.loads(summary_path.read_text())
    result = ValidationResult()

    targets = summary.get("targets", [])
    masks = summary.get("mask_summaries", {})
    saliency = summary.get("saliency_summaries", {})
    calibration = summary.get("calibration", {})

    if not targets:
        result.errors.append("Artifact contains no discovered targets.")
    if len(masks) != len(targets):
        result.errors.append(f"Mask count {len(masks)} does not match target count {len(targets)}.")
    if len(saliency) != len(targets):
        result.errors.append(
            f"Saliency count {len(saliency)} does not match target count {len(targets)}."
        )
    if calibration.get("num_tokens", 0) <= 0:
        result.errors.append("Calibration token count must be positive.")
    if calibration.get("num_batches", 0) <= 0:
        result.errors.append("Calibration batch count must be positive.")

    topologies = {target.get("topology") for target in targets}
    if expected_topology and topologies != {expected_topology}:
        result.errors.append(
            f"Expected topology {expected_topology!r}, found {sorted(topologies)!r}."
        )

    for target in targets:
        _validate_artifact_target(target, result)

    for key, mask in masks.items():
        _validate_artifact_mask(key, mask, result)

    for key, saliency_stats in saliency.items():
        _validate_saliency_stats(key, saliency_stats, result)

    if "baseline_loss" not in summary or "masked_loss" not in summary:
        result.errors.append("Artifact is missing baseline or masked loss.")
    if "baseline_perplexity" not in summary or "masked_perplexity" not in summary:
        result.errors.append("Artifact is missing baseline or masked perplexity.")

    return result


def _validate_artifact_target(target: dict[str, Any], result: ValidationResult) -> None:
    topology = target.get("topology")
    ffn_path = target.get("ffn_path", "<unknown>")
    expand_paths = target.get("expand_module_paths", [])
    contract_paths = target.get("contract_module_paths", [])

    if topology == FFNTopologyKind.DENSE.value and len(expand_paths) != 1:
        result.errors.append(f"Dense target has invalid expansion count: {ffn_path}")
    if topology == FFNTopologyKind.GATED.value and len(expand_paths) < 2:
        result.errors.append(f"Gated target has too few expansion branches: {ffn_path}")
    if not contract_paths:
        result.errors.append(f"Target has no contraction paths: {ffn_path}")
    if not target.get("intermediate_size"):
        result.errors.append(f"Target has invalid intermediate size: {ffn_path}")
    if not target.get("hidden_size"):
        result.errors.append(f"Target has invalid hidden size: {ffn_path}")


def _validate_artifact_mask(key: str, mask: dict[str, Any], result: ValidationResult) -> None:
    values = mask.get("values", {})
    total = mask.get("total_channels")
    active = mask.get("active_channels")
    retained = mask.get("retained_ratio")
    shape = values.get("shape", [])
    mask_sum = values.get("sum")

    if not total or total <= 0:
        result.errors.append(f"Mask {key} has invalid total channel count.")
    if shape and total and shape[-1] != total:
        result.errors.append(f"Mask {key} shape {shape} does not match total {total}.")
    if retained is None or not 0.0 < retained <= 1.0:
        result.errors.append(f"Mask {key} has invalid retained ratio {retained}.")
    if active is None:
        result.errors.append(f"Mask {key} is missing active channel count.")
    if mask_sum is not None and active is not None and abs(float(mask_sum) - float(active)) > 1.0:
        result.warnings.append(
            f"Mask {key} active_channels={active} differs from tensor sum={mask_sum}. "
            "This can happen in old low-precision artifacts and is fixed for future runs."
        )


def _validate_saliency_stats(
    key: str,
    saliency_stats: dict[str, Any],
    result: ValidationResult,
) -> None:
    for name in ("magnitude", "gradient", "combined"):
        stats = saliency_stats.get(name)
        if not stats:
            result.errors.append(f"Saliency {key} is missing {name} stats.")
            continue
        if stats.get("numel", 0) <= 0:
            result.errors.append(f"Saliency {key}/{name} has no elements.")
        if stats.get("max") is None or stats.get("mean") is None:
            result.errors.append(f"Saliency {key}/{name} is missing numeric stats.")
