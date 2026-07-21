"""Audit and plot MaGRIP training run artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path to summary.json from a MaGRIP run.")
    parser.add_argument(
        "--output-dir",
        help="Directory for plots. Defaults to <run_dir>/plots.",
    )
    parser.add_argument(
        "--mask-state",
        help="Optional mask_state.pt path for final per-layer retention plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only print the textual audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    training = summary.get("training", summary)
    metrics = _load_metrics(summary_path, summary, training)
    run_dir = Path(summary.get("run_dir") or summary_path.parent)

    _print_audit(summary, training, metrics)

    if args.no_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots. Install it with `pip install matplotlib`."
        ) from exc

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = [
        _plot_losses(plt, metrics, output_dir),
        _plot_budget(plt, training, metrics, output_dir),
        _plot_mask_dynamics(plt, metrics, output_dir),
    ]

    mask_state = _resolve_mask_state(args.mask_state, summary_path, summary, run_dir)
    layer_plot = _plot_final_layer_retention(plt, mask_state, output_dir)
    if layer_plot is not None:
        plot_paths.append(layer_plot)

    print("\nplots:")
    for path in plot_paths:
        print(f"  {path}")


def _load_metrics(
    summary_path: Path,
    summary: dict[str, Any],
    training: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = training.get("metrics")
    if isinstance(metrics, list):
        return metrics
    run_dir = Path(summary.get("run_dir") or summary_path.parent)
    metrics_path = run_dir / "metrics" / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text())
        loaded = payload.get("metrics")
        if isinstance(loaded, list):
            return loaded
    raise SystemExit("Could not find metrics in summary.json or metrics/metrics.json.")


def _print_audit(
    summary: dict[str, Any],
    training: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> None:
    calibration = summary.get("calibration", {})
    targets = summary.get("targets", [])
    target_count = len(targets)
    topology_counts = {}
    for target in targets:
        topology = target.get("topology", "unknown")
        topology_counts[topology] = topology_counts.get(topology, 0) + 1

    mask_updates = sum(bool(item.get("mask_update")) for item in metrics)
    weight_updates = sum(bool(item.get("weight_update")) for item in metrics)
    mask_grad_steps = sum(item.get("mask_grad_norm") is not None for item in metrics)

    objectives = [item.get("objective", {}) for item in metrics]
    soft = _series(objectives, "retained_cost_ratio")
    hard = _series(metrics, "hard_retained_cost_ratio")
    entropy = _series(objectives, "mask_entropy")
    task_loss = _series(objectives, "task_loss")
    total_loss = _series(objectives, "total_loss")
    grad_norm = _series(metrics, "mask_grad_norm")
    weight_grad_norm = _series(metrics, "weight_grad_norm")
    apollo_state_norm = _nested_series(metrics, "apollo_diagnostics", "optimizer_state_tensor_norm")
    apollo_projected_norm = _nested_series(
        metrics,
        "apollo_diagnostics",
        "projected_state_tensor_norm",
    )
    validation_loss = _series(metrics, "validation_loss")
    nonfinite = _count_nonfinite(metrics)

    print("audit:")
    print(f"  model: {summary.get('model_name')}")
    print(f"  targets: {target_count} {topology_counts}")
    print(
        "  data: "
        f"train_samples={calibration.get('num_samples')} "
        f"eval_samples={calibration.get('eval_num_samples')} "
        f"max_length={calibration.get('max_length')}"
    )
    print(
        "  losses: "
        f"baseline={_fmt(training.get('baseline_loss'))} "
        f"initial_masked={_fmt(training.get('initial_masked_loss'))} "
        f"final_masked={_fmt(training.get('final_masked_loss'))}"
    )
    print(
        "  perplexity: "
        f"baseline={_fmt(training.get('baseline_perplexity'))} "
        f"initial_masked={_fmt(training.get('initial_masked_perplexity'))} "
        f"final_masked={_fmt(training.get('final_masked_perplexity'))}"
    )
    print(
        "  updates: "
        f"steps={len(metrics)} mask_updates={mask_updates} "
        f"mask_grad_steps={mask_grad_steps} weight_updates={weight_updates}"
    )
    print(f"  soft_budget: {_range_text(soft)}")
    print(f"  hard_budget: {_range_text(hard)}")
    print(f"  entropy: {_range_text(entropy)}")
    print(f"  task_loss: {_range_text(task_loss)}")
    print(f"  total_loss: {_range_text(total_loss)}")
    print(f"  mask_grad_norm: {_range_text(grad_norm)}")
    print(f"  weight_grad_norm: {_range_text(weight_grad_norm)}")
    print(f"  apollo_state_norm: {_range_text(apollo_state_norm)}")
    print(f"  apollo_projected_state_norm: {_range_text(apollo_projected_norm)}")
    print(f"  validation_loss: {_range_text(validation_loss)}")
    print(f"  nonfinite_metrics: {nonfinite}")

    grad_nonzero = _series(metrics, "mask_grad_nonzero_count")
    grad_targets = _series(metrics, "mask_grad_target_count")
    if grad_nonzero and grad_targets:
        print(
            "  per_target_grad_coverage: "
            f"first={int(grad_nonzero[0])}/{int(grad_targets[0])} "
            f"last={int(grad_nonzero[-1])}/{int(grad_targets[-1])}"
        )
    else:
        print("  per_target_grad_coverage: not logged in this run")


def _plot_losses(plt: Any, metrics: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    task_steps, task_loss = _nested_paired_series(metrics, "objective", "task_loss")
    total_steps, total_loss = _nested_paired_series(metrics, "objective", "total_loss")
    if task_loss:
        ax.plot(task_steps, task_loss, label="task loss", linewidth=1.5)
    if total_loss:
        ax.plot(total_steps, total_loss, label="total loss", linewidth=1.5)
    validation_steps, validation_loss = _paired_series(metrics, "step", "validation_loss")
    if validation_loss:
        ax.plot(
            validation_steps,
            validation_loss,
            label="validation loss",
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
    budget_steps, budget = _nested_paired_series(metrics, "objective", "budget_penalty")
    if budget:
        ax.plot(budget_steps, budget, label="budget penalty", linewidth=1.0)
    ax.set_title("Training Objective")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save(fig, output_dir / "loss_curves.png")


def _plot_budget(
    plt: Any,
    training: dict[str, Any],
    metrics: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    objectives = [item.get("objective", {}) for item in metrics]
    target = _first_finite(_series(objectives, "target_retained_ratio"))
    fig, ax = plt.subplots(figsize=(9, 5))
    soft_steps, soft_retained = _nested_paired_series(metrics, "objective", "retained_cost_ratio")
    hard_steps, hard_retained = _paired_series(metrics, "step", "hard_retained_cost_ratio")
    if soft_retained:
        ax.plot(soft_steps, soft_retained, label="soft retained", linewidth=1.5)
    if hard_retained:
        ax.plot(hard_steps, hard_retained, label="hard retained", linewidth=1.5)
    if target is not None:
        ax.axhline(target, color="black", linestyle="--", linewidth=1.0, label="target")
    final_ratio = training.get("mask_cost", {}).get("retained_ratio")
    if final_ratio is not None:
        ax.axhline(final_ratio, color="tab:green", linestyle=":", linewidth=1.0, label="final hard")
    ax.set_title("Budget Tracking")
    ax.set_xlabel("step")
    ax.set_ylabel("retained cost ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save(fig, output_dir / "budget_curves.png")


def _plot_mask_dynamics(plt: Any, metrics: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    temperature_steps, temperature = _paired_series(metrics, "step", "temperature")
    if temperature:
        axes[0].plot(temperature_steps, temperature, color="tab:orange")
    axes[0].set_ylabel("temperature")
    entropy_steps, entropy = _nested_paired_series(metrics, "objective", "mask_entropy")
    if entropy:
        axes[1].plot(entropy_steps, entropy, color="tab:purple")
    axes[1].set_ylabel("entropy")
    grad_steps, grad_norm = _paired_series(metrics, "step", "mask_grad_norm")
    if grad_norm:
        axes[2].plot(grad_steps, grad_norm, color="tab:red")
    axes[2].set_ylabel("grad norm")
    axes[2].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Mask Dynamics")
    return _save(fig, output_dir / "mask_dynamics.png")


def _plot_final_layer_retention(plt: Any, mask_state_path: Path | None, output_dir: Path) -> Path | None:
    if mask_state_path is None or not mask_state_path.exists():
        print("\nfinal layer plot skipped: mask_state.pt was not found.")
        return None
    try:
        from magrip.masks import load_mask_state
    except ImportError as exc:
        raise SystemExit("Could not import MaGRIP mask loader.") from exc

    masks = load_mask_state(mask_state_path)
    records = []
    for key, mask in masks.items():
        block_index = mask.target.block_index
        label = block_index if block_index is not None else key
        records.append((label, mask.retained_ratio, mask.cost_summary.retained_ratio))
    records.sort(key=lambda item: item[0] if isinstance(item[0], int) else str(item[0]))

    labels = [str(item[0]) for item in records]
    channel_ratios = [item[1] for item in records]
    cost_ratios = [item[2] for item in records]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(records)))
    ax.bar([value - 0.2 for value in x], channel_ratios, width=0.4, label="channels")
    ax.bar([value + 0.2 for value in x], cost_ratios, width=0.4, label="cost")
    ax.set_title("Final Per-Layer Retention")
    ax.set_xlabel("block index")
    ax.set_ylabel("retained ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90 if len(labels) > 24 else 0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    return _save(fig, output_dir / "final_layer_retention.png")


def _resolve_mask_state(
    explicit: str | None,
    summary_path: Path,
    summary: dict[str, Any],
    run_dir: Path,
) -> Path | None:
    if explicit:
        return Path(explicit)
    candidates = [
        run_dir / "checkpoints" / "final_mask_state.pt",
        summary_path.parent / "checkpoints" / "final_mask_state.pt",
    ]
    model_output_dir = summary.get("model_output_dir")
    if model_output_dir:
        candidates.append(Path(model_output_dir) / "mask_state.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _series(items: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _nested_series(items: list[dict[str, Any]], parent_key: str, child_key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        parent = item.get(parent_key)
        if not isinstance(parent, dict):
            continue
        value = parent.get(child_key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _paired_series(
    items: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for index, item in enumerate(items):
        y_value = item.get(y_key)
        if y_value is None:
            continue
        try:
            x = float(item.get(x_key, index))
            y = float(y_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def _nested_paired_series(
    items: list[dict[str, Any]],
    parent_key: str,
    child_key: str,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for index, item in enumerate(items):
        parent = item.get(parent_key)
        if not isinstance(parent, dict):
            continue
        y_value = parent.get(child_key)
        if y_value is None:
            continue
        try:
            x = float(item.get("step", index))
            y = float(y_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def _range_text(values: list[float]) -> str:
    if not values:
        return "not logged"
    return (
        f"min={min(values):.6g} max={max(values):.6g} "
        f"first={values[0]:.6g} last={values[-1]:.6g}"
    )


def _first_finite(values: list[float]) -> float | None:
    return values[0] if values else None


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not logged"
    if not math.isfinite(number):
        return "not finite"
    return f"{number:.6g}"


def _count_nonfinite(metrics: list[dict[str, Any]]) -> int:
    count = 0
    for metric in metrics:
        for value in metric.values():
            count += _nonfinite_in_value(value)
    return count


def _nonfinite_in_value(value: Any) -> int:
    if isinstance(value, float):
        return 0 if math.isfinite(value) else 1
    if isinstance(value, dict):
        return sum(_nonfinite_in_value(item) for item in value.values())
    if isinstance(value, list):
        return sum(_nonfinite_in_value(item) for item in value)
    return 0


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.clear()
    return path


if __name__ == "__main__":
    main()
