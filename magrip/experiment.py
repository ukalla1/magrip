"""Experiment tracking artifacts for MaGRIP runs."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
import statistics
from typing import Any

from magrip.logging import tensor_stats, to_jsonable


def write_training_research_artifacts(
    *,
    run_dir: str | Path,
    summary: dict[str, Any],
    result: Any,
) -> dict[str, str]:
    """Write research-oriented tables and tensors for a training run."""

    run_path = Path(run_dir)
    metrics_dir = run_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "stage_metrics": metrics_dir / "stage_metrics.csv",
        "validation_curve": metrics_dir / "validation_curve.csv",
        "training_windows": metrics_dir / "training_windows.csv",
        "layer_diagnostics": metrics_dir / "layer_diagnostics.csv",
        "channel_diagnostics": metrics_dir / "channel_diagnostics.pkl",
        "channel_diagnostics_manifest": metrics_dir / "channel_diagnostics_manifest.json",
        "research_summary": metrics_dir / "research_summary.json",
        "run_card": run_path / "RUN_CARD.md",
    }

    metrics = _training_metrics(summary)
    _write_stage_metrics(artifact_paths["stage_metrics"], summary)
    _write_validation_curve(artifact_paths["validation_curve"], metrics)
    _write_training_windows(artifact_paths["training_windows"], metrics)
    channel_manifest = _write_channel_diagnostics(
        artifact_paths["layer_diagnostics"],
        artifact_paths["channel_diagnostics"],
        result,
    )
    artifact_paths["channel_diagnostics_manifest"].write_text(
        json.dumps(to_jsonable(channel_manifest), indent=2) + "\n"
    )
    research_summary = build_research_summary(
        summary=summary,
        metrics=metrics,
        channel_manifest=channel_manifest,
        artifact_paths=artifact_paths,
    )
    artifact_paths["research_summary"].write_text(
        json.dumps(to_jsonable(research_summary), indent=2) + "\n"
    )
    artifact_paths["run_card"].write_text(_run_card_text(research_summary) + "\n")
    return {key: str(value) for key, value in artifact_paths.items()}


def write_compaction_research_artifacts(
    *,
    output_dir: str | Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Write research-oriented artifacts for a compaction run."""

    output_path = Path(output_dir)
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "compaction_summary": metrics_dir / "compaction_summary.json",
        "compaction_stage_metrics": metrics_dir / "compaction_stage_metrics.csv",
        "compaction_targets": metrics_dir / "compaction_targets.csv",
        "logit_equivalence": metrics_dir / "logit_equivalence.csv",
    }

    artifact_paths["compaction_summary"].write_text(
        json.dumps(to_jsonable(manifest), indent=2) + "\n"
    )
    _write_compaction_stage_metrics(artifact_paths["compaction_stage_metrics"], manifest)
    _write_compaction_targets(artifact_paths["compaction_targets"], manifest)
    _write_logit_equivalence(artifact_paths["logit_equivalence"], manifest)
    return {key: str(path) for key, path in artifact_paths.items()}


def build_research_summary(
    *,
    summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    channel_manifest: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build compact run-level diagnostics for paper-oriented analysis."""

    validation_points = [
        item for item in metrics if _finite_or_none(item.get("validation_loss")) is not None
    ]
    best_validation = None
    if validation_points:
        best = min(validation_points, key=lambda item: float(item["validation_loss"]))
        best_validation = {
            "step": best.get("step"),
            "loss": best.get("validation_loss"),
            "perplexity": best.get("validation_perplexity"),
        }

    training = summary.get("training", {})
    mask_cost = summary.get("mask_cost", {})
    objective_config = summary.get("config", {}).get("objective", {})
    target_ratio = objective_config.get("target_retained_ratio")
    final_ratio = mask_cost.get("retained_ratio")
    budget_error = None
    if target_ratio is not None and final_ratio is not None:
        budget_error = float(final_ratio) - float(target_ratio)

    return {
        "model_name": summary.get("model_name"),
        "run_dir": summary.get("run_dir"),
        "pruned_dir": summary.get("pruned_dir"),
        "calibration": summary.get("calibration"),
        "objective_config": objective_config,
        "optimizer_config": summary.get("config", {}).get("optimizer", {}),
        "mask_schedule": summary.get("config", {}).get("mask_schedule", {}),
        "training_config": summary.get("config", {}).get("training", {}),
        "losses": {
            "baseline": summary.get("baseline_loss"),
            "initial_masked": training.get("initial_masked_loss"),
            "final_masked": summary.get("masked_loss"),
            "delta_final_vs_baseline": summary.get("loss_delta"),
            "recovered_from_initial": _loss_recovery_fraction(summary),
        },
        "perplexity": {
            "baseline": summary.get("baseline_perplexity"),
            "initial_masked": training.get("initial_masked_perplexity"),
            "final_masked": summary.get("masked_perplexity"),
            "delta_final_vs_baseline": summary.get("perplexity_delta"),
        },
        "validation": {
            "points": len(validation_points),
            "first": _validation_point(validation_points[0]) if validation_points else None,
            "best": best_validation,
            "last": _validation_point(validation_points[-1]) if validation_points else None,
        },
        "updates": {
            "steps": len(metrics),
            "mask_updates": sum(bool(item.get("mask_update")) for item in metrics),
            "weight_updates": sum(bool(item.get("weight_update")) for item in metrics),
            "mask_grad_steps": sum(item.get("mask_grad_norm") is not None for item in metrics),
        },
        "budget": {
            "target_retained_ratio": target_ratio,
            "final_retained_ratio": final_ratio,
            "final_budget_error": budget_error,
            "soft_retained": _range_for_nested(metrics, "objective", "retained_cost_ratio"),
            "hard_retained": _range_for_key(metrics, "hard_retained_cost_ratio"),
        },
        "mask_dynamics": {
            "entropy": _range_for_nested(metrics, "objective", "mask_entropy"),
            "mask_grad_norm": _range_for_key(metrics, "mask_grad_norm"),
            "mask_grad_nonfinite_count": _sum_for_key(metrics, "mask_grad_nonfinite_count"),
            "mask_update_mean_abs": _range_for_key(metrics, "mask_update_mean_abs"),
            "active_to_inactive_count": _sum_for_key(metrics, "active_to_inactive_count"),
            "inactive_to_active_count": _sum_for_key(metrics, "inactive_to_active_count"),
            "mask_flip_count": _sum_for_key(metrics, "mask_flip_count"),
            "initial_to_final_mask_flips": training.get("initial_to_final_mask_flips"),
        },
        "weight_dynamics": {
            "weight_grad_norm": _range_for_key(metrics, "weight_grad_norm"),
            "weight_grad_nonfinite_steps": sum(
                1 for item in metrics if bool(item.get("weight_grad_nonfinite"))
            ),
            "apollo_parameter_stats": training.get("apollo_parameter_stats"),
            "apollo_projected_state_norm": _range_for_nested(
                metrics,
                "apollo_diagnostics",
                "projected_state_tensor_norm",
            ),
            "apollo_update_state_norm": _range_for_nested(
                metrics,
                "apollo_diagnostics",
                "update_state_tensor_norm",
            ),
        },
        "mask_cost": mask_cost,
        "targets": {
            "count": len(summary.get("targets", [])),
            "topology_counts": _topology_counts(summary.get("targets", [])),
        },
        "channel_diagnostics": channel_manifest,
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }


def _write_stage_metrics(path: Path, summary: dict[str, Any]) -> None:
    training = summary.get("training", {})
    mask_cost = summary.get("mask_cost", {})
    rows = [
        {
            "stage": "baseline",
            "loss": summary.get("baseline_loss"),
            "perplexity": summary.get("baseline_perplexity"),
            "retained_ratio": 1.0,
            "flop_retained_ratio": 1.0,
        },
        {
            "stage": "initial_masked",
            "loss": training.get("initial_masked_loss"),
            "perplexity": training.get("initial_masked_perplexity"),
            "retained_ratio": mask_cost.get("retained_ratio"),
            "flop_retained_ratio": mask_cost.get("flop_retained_ratio"),
        },
        {
            "stage": "final_masked",
            "loss": summary.get("masked_loss"),
            "perplexity": summary.get("masked_perplexity"),
            "retained_ratio": mask_cost.get("retained_ratio"),
            "flop_retained_ratio": mask_cost.get("flop_retained_ratio"),
        },
    ]
    _write_csv(path, rows, ["stage", "loss", "perplexity", "retained_ratio", "flop_retained_ratio"])


def _write_compaction_stage_metrics(path: Path, manifest: dict[str, Any]) -> None:
    evaluation = manifest.get("evaluation") or {}
    rows = []
    for stage in ("masked_reference", "compacted"):
        item = evaluation.get(stage) or {}
        rows.append(
            {
                "stage": stage,
                "loss": item.get("loss"),
                "perplexity": item.get("perplexity"),
                "num_batches": item.get("num_batches"),
                "num_tokens": item.get("num_tokens"),
            }
        )
    _write_csv(path, rows, ["stage", "loss", "perplexity", "num_batches", "num_tokens"])


def _write_compaction_targets(path: Path, manifest: dict[str, Any]) -> None:
    targets = manifest.get("compaction", {}).get("targets", [])
    rows = [
        {
            "ffn_path": target.get("ffn_path"),
            "topology": target.get("topology"),
            "original_channels": target.get("original_channels"),
            "retained_channels": target.get("retained_channels"),
            "retained_ratio": target.get("retained_ratio"),
        }
        for target in targets
    ]
    _write_csv(
        path,
        rows,
        ["ffn_path", "topology", "original_channels", "retained_channels", "retained_ratio"],
    )


def _write_logit_equivalence(path: Path, manifest: dict[str, Any]) -> None:
    verification = manifest.get("verification") or {}
    rows = [
        {
            "ok": verification.get("ok"),
            "atol": verification.get("atol"),
            "rtol": verification.get("rtol"),
            "max_abs_error": verification.get("max_abs_error"),
            "mean_abs_error": verification.get("mean_abs_error"),
            "batches": verification.get("batches"),
        }
    ]
    _write_csv(
        path,
        rows,
        ["ok", "atol", "rtol", "max_abs_error", "mean_abs_error", "batches"],
    )


def _write_validation_curve(path: Path, metrics: list[dict[str, Any]]) -> None:
    rows = [
        {
            "step": item.get("step"),
            "validation_loss": item.get("validation_loss"),
            "validation_perplexity": item.get("validation_perplexity"),
            "task_loss": item.get("objective", {}).get("task_loss"),
            "total_loss": item.get("objective", {}).get("total_loss"),
            "soft_retained_ratio": item.get("objective", {}).get("retained_cost_ratio"),
            "hard_retained_ratio": item.get("hard_retained_cost_ratio"),
        }
        for item in metrics
        if item.get("validation_loss") is not None
    ]
    _write_csv(
        path,
        rows,
        [
            "step",
            "validation_loss",
            "validation_perplexity",
            "task_loss",
            "total_loss",
            "soft_retained_ratio",
            "hard_retained_ratio",
        ],
    )


def _write_training_windows(path: Path, metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        _write_csv(path, [], ["window_start", "window_end"])
        return
    window = max(1, min(512, len(metrics) // 8 or 1))
    rows = []
    for start in range(0, len(metrics), window):
        segment = metrics[start : start + window]
        rows.append(
            {
                "window_start": segment[0].get("step", start),
                "window_end": segment[-1].get("step", start + len(segment) - 1),
                "num_steps": len(segment),
                "task_loss_mean": _mean_nested(segment, "objective", "task_loss"),
                "task_loss_median": _median_nested(segment, "objective", "task_loss"),
                "total_loss_mean": _mean_nested(segment, "objective", "total_loss"),
                "validation_loss_last": _last_for_key(segment, "validation_loss"),
                "soft_retained_first": _first_nested(segment, "objective", "retained_cost_ratio"),
                "soft_retained_last": _last_nested(segment, "objective", "retained_cost_ratio"),
                "hard_retained_first": _first_for_key(segment, "hard_retained_cost_ratio"),
                "hard_retained_last": _last_for_key(segment, "hard_retained_cost_ratio"),
                "entropy_first": _first_nested(segment, "objective", "mask_entropy"),
                "entropy_last": _last_nested(segment, "objective", "mask_entropy"),
                "mask_grad_norm_mean": _mean_for_key(segment, "mask_grad_norm"),
                "weight_grad_norm_mean": _mean_for_key(segment, "weight_grad_norm"),
                "mask_flips_total": _sum_for_key(segment, "mask_flip_count"),
                "mask_flip_steps": sum(
                    1 for item in segment if _finite_or_none(item.get("mask_flip_count"))
                ),
            }
        )
    _write_csv(path, rows, list(rows[0].keys()) if rows else ["window_start", "window_end"])


def _write_channel_diagnostics(
    layer_path: Path,
    channel_path: Path,
    result: Any,
) -> dict[str, Any]:
    layer_rows = []
    channel_payload: dict[str, Any] = {}
    masks = result.masks.as_dict()
    saliency = result.initial_saliency
    combined = saliency.combined()
    flips_by_target = result.initial_to_final_mask_flips_by_target or {}
    for key, mask in masks.items():
        probabilities = mask.probabilities.detach().cpu().float()
        logits = mask.logits.detach().cpu().float()
        binary = mask.binary_values.detach().cpu().float()
        magnitude = saliency.magnitude[key].detach().cpu().float()
        gradient = saliency.gradient[key].detach().cpu().float()
        combined_scores = combined[key].detach().cpu().float()
        layer_rows.append(
            {
                "ffn_path": key,
                "block_index": mask.target.block_index,
                "topology": mask.target.topology.value,
                "active_channels": mask.active_channels,
                "total_channels": mask.total_channels,
                "retained_ratio": mask.retained_ratio,
                "cost_retained_ratio": mask.cost_summary.retained_ratio,
                "flop_cost_retained_ratio": mask.cost_summary.flop_retained_ratio,
                "probability_mean": float(probabilities.mean().item()),
                "probability_std": float(probabilities.std(unbiased=False).item()),
                "logit_mean": float(logits.mean().item()),
                "logit_std": float(logits.std(unbiased=False).item()),
                "saliency_magnitude_mean": float(magnitude.mean().item()),
                "saliency_gradient_mean": float(gradient.mean().item()),
                "saliency_combined_mean": float(combined_scores.mean().item()),
                "saliency_combined_std": float(combined_scores.std(unbiased=False).item()),
                "initial_to_final_active_to_inactive": (
                    flips_by_target.get(key, {}).get("active_to_inactive")
                ),
                "initial_to_final_inactive_to_active": (
                    flips_by_target.get(key, {}).get("inactive_to_active")
                ),
                "initial_to_final_total_flips": flips_by_target.get(key, {}).get("total"),
            }
        )
        channel_payload[key] = {
            "target": {
                "block_index": mask.target.block_index,
                "ffn_path": mask.target.ffn_path,
                "topology": mask.target.topology.value,
                "intermediate_size": mask.target.intermediate_size,
                "hidden_size": mask.target.hidden_size,
            },
            "binary_mask": binary,
            "probabilities": probabilities,
            "logits": logits,
            "saliency_magnitude": magnitude,
            "saliency_gradient": gradient,
            "saliency_combined": combined_scores,
            "cost_per_channel": mask.cost_per_channel.detach().cpu().float(),
            "flop_cost_per_channel": mask.flop_cost_per_channel.detach().cpu().float(),
            "stats": {
                "binary_mask": tensor_stats(binary),
                "probabilities": tensor_stats(probabilities),
                "logits": tensor_stats(logits),
                "saliency_combined": tensor_stats(combined_scores),
            },
        }
    _write_csv(layer_path, layer_rows, list(layer_rows[0].keys()) if layer_rows else ["ffn_path"])
    with channel_path.open("wb") as handle:
        pickle.dump(channel_payload, handle)
    return {
        "path": str(channel_path),
        "format": "pickle",
        "target_count": len(channel_payload),
        "total_channels": sum(item["binary_mask"].numel() for item in channel_payload.values()),
        "layers": {
            key: {
                "channels": int(value["binary_mask"].numel()),
                "active_channels": int(value["binary_mask"].sum().item()),
                "retained_ratio": (
                    float(value["binary_mask"].mean().item())
                    if value["binary_mask"].numel()
                    else 0.0
                ),
            }
            for key, value in channel_payload.items()
        },
    }


def _run_card_text(summary: dict[str, Any]) -> str:
    losses = summary.get("losses", {})
    ppl = summary.get("perplexity", {})
    budget = summary.get("budget", {})
    validation = summary.get("validation", {})
    updates = summary.get("updates", {})
    return "\n".join(
        [
            f"# MaGRIP Run Card: {summary.get('model_name')}",
            "",
            f"- Run directory: `{summary.get('run_dir')}`",
            f"- Targets: {summary.get('targets', {}).get('count')} "
            f"{summary.get('targets', {}).get('topology_counts')}",
            f"- Baseline loss / ppl: {losses.get('baseline')} / {ppl.get('baseline')}",
            f"- Initial masked loss / ppl: {losses.get('initial_masked')} / "
            f"{ppl.get('initial_masked')}",
            f"- Final masked loss / ppl: {losses.get('final_masked')} / {ppl.get('final_masked')}",
            f"- Final retained ratio: {budget.get('final_retained_ratio')} "
            f"(target {budget.get('target_retained_ratio')})",
            f"- Validation best: {validation.get('best')}",
            f"- Updates: {updates}",
            "",
            "Research artifacts are listed in `metrics/research_summary.json`.",
        ]
    )


def _training_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    training = summary.get("training", {})
    metrics = training.get("metrics", []) if isinstance(training, dict) else []
    return metrics if isinstance(metrics, list) else []


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def _topology_counts(targets: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        topology = str(target.get("topology", "unknown"))
        counts[topology] = counts.get(topology, 0) + 1
    return counts


def _loss_recovery_fraction(summary: dict[str, Any]) -> float | None:
    training = summary.get("training", {})
    baseline = _finite_or_none(summary.get("baseline_loss"))
    initial = _finite_or_none(training.get("initial_masked_loss"))
    final = _finite_or_none(summary.get("masked_loss"))
    if baseline is None or initial is None or final is None or initial == baseline:
        return None
    return (initial - final) / (initial - baseline)


def _validation_point(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": item.get("step"),
        "loss": item.get("validation_loss"),
        "perplexity": item.get("validation_perplexity"),
    }


def _range_for_key(items: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    return _range([item.get(key) for item in items])


def _range_for_nested(
    items: list[dict[str, Any]],
    parent_key: str,
    child_key: str,
) -> dict[str, float] | None:
    values = []
    for item in items:
        parent = item.get(parent_key)
        if isinstance(parent, dict):
            values.append(parent.get(child_key))
    return _range(values)


def _range(values: list[Any]) -> dict[str, float] | None:
    finite = [_finite_or_none(value) for value in values]
    finite = [value for value in finite if value is not None]
    if not finite:
        return None
    return {
        "min": min(finite),
        "max": max(finite),
        "first": finite[0],
        "last": finite[-1],
    }


def _sum_for_key(items: list[dict[str, Any]], key: str) -> float:
    return sum(value for value in (_finite_or_none(item.get(key)) for item in items) if value is not None)


def _mean_for_key(items: list[dict[str, Any]], key: str) -> float | None:
    values = [value for value in (_finite_or_none(item.get(key)) for item in items) if value is not None]
    return sum(values) / len(values) if values else None


def _mean_nested(items: list[dict[str, Any]], parent_key: str, child_key: str) -> float | None:
    values = _nested_values(items, parent_key, child_key)
    return sum(values) / len(values) if values else None


def _median_nested(items: list[dict[str, Any]], parent_key: str, child_key: str) -> float | None:
    values = _nested_values(items, parent_key, child_key)
    return statistics.median(values) if values else None


def _nested_values(items: list[dict[str, Any]], parent_key: str, child_key: str) -> list[float]:
    values = []
    for item in items:
        parent = item.get(parent_key)
        if isinstance(parent, dict):
            value = _finite_or_none(parent.get(child_key))
            if value is not None:
                values.append(value)
    return values


def _first_for_key(items: list[dict[str, Any]], key: str) -> float | None:
    for item in items:
        value = _finite_or_none(item.get(key))
        if value is not None:
            return value
    return None


def _last_for_key(items: list[dict[str, Any]], key: str) -> float | None:
    for item in reversed(items):
        value = _finite_or_none(item.get(key))
        if value is not None:
            return value
    return None


def _first_nested(items: list[dict[str, Any]], parent_key: str, child_key: str) -> float | None:
    values = _nested_values(items, parent_key, child_key)
    return values[0] if values else None


def _last_nested(items: list[dict[str, Any]], parent_key: str, child_key: str) -> float | None:
    values = _nested_values(items, parent_key, child_key)
    return values[-1] if values else None


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
