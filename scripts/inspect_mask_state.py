"""Inspect and validate saved MaGRIP structured mask artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pruned_dir", help="Directory containing mask_state.pt and masks.pt.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from magrip.masks import load_mask_state, total_mask_cost

    pruned_dir = Path(args.pruned_dir)
    mask_state_path = pruned_dir / "mask_state.pt"
    masks_path = pruned_dir / "masks.pt"
    manifest_path = pruned_dir / "manifest.json"

    errors: list[str] = []
    warnings: list[str] = []

    if not mask_state_path.exists():
        errors.append(f"Missing {mask_state_path}.")
    if not masks_path.exists():
        errors.append(f"Missing {masks_path}.")
    if not manifest_path.exists():
        errors.append(f"Missing {manifest_path}.")
    if errors:
        _finish(errors, warnings, strict=args.strict)

    masks = load_mask_state(mask_state_path)
    binary_tensors = torch.load(masks_path, map_location="cpu")
    manifest = json.loads(manifest_path.read_text())
    mask_summaries = manifest.get("mask_summaries", {})

    if set(masks) != set(binary_tensors):
        errors.append("mask_state.pt keys do not match masks.pt keys.")
    if set(masks) != set(mask_summaries):
        errors.append("mask_state.pt keys do not match manifest mask_summaries keys.")

    for key, mask in masks.items():
        binary = mask.binary_values.detach().cpu()
        saved_binary = binary_tensors.get(key)
        summary = mask_summaries.get(key, {})
        if saved_binary is not None and not torch.equal(binary, saved_binary.float()):
            errors.append(f"{key}: mask_state binary values do not match masks.pt.")
        if summary:
            _compare_int(errors, key, "active_channels", mask.active_channels, summary)
            _compare_int(errors, key, "total_channels", mask.total_channels, summary)
            _compare_float(errors, key, "full_cost", mask.cost_summary.full_cost, summary)
            _compare_float(errors, key, "retained_cost", mask.cost_summary.retained_cost, summary)
            _compare_float(
                errors,
                key,
                "full_flop_cost",
                mask.cost_summary.full_flop_cost,
                summary,
            )
            _compare_float(
                errors,
                key,
                "retained_flop_cost",
                mask.cost_summary.retained_flop_cost,
                summary,
            )

    aggregate = total_mask_cost(masks)
    manifest_cost = manifest.get("mask_cost", {})
    if manifest_cost:
        _compare_float(errors, "aggregate", "full_cost", aggregate.full_cost, manifest_cost)
        _compare_float(
            errors,
            "aggregate",
            "retained_cost",
            aggregate.retained_cost,
            manifest_cost,
        )
        _compare_float(
            errors,
            "aggregate",
            "full_flop_cost",
            aggregate.full_flop_cost,
            manifest_cost,
        )
        _compare_float(
            errors,
            "aggregate",
            "retained_flop_cost",
            aggregate.retained_flop_cost,
            manifest_cost,
        )
    else:
        warnings.append("manifest.json has no aggregate mask_cost block.")

    print(f"Mask targets: {len(masks)}")
    print(f"Retained parameter-cost ratio: {aggregate.retained_ratio:.6f}")
    print(f"Retained FLOP-cost ratio: {aggregate.flop_retained_ratio:.6f}")
    _finish(errors, warnings, strict=args.strict)


def _compare_int(
    errors: list[str],
    key: str,
    field: str,
    actual: int,
    expected: dict[str, object],
) -> None:
    if field in expected and int(expected[field]) != int(actual):
        errors.append(f"{key}: {field} mismatch, state={actual}, manifest={expected[field]}.")


def _compare_float(
    errors: list[str],
    key: str,
    field: str,
    actual: float,
    expected: dict[str, object],
    tolerance: float = 1e-3,
) -> None:
    if field in expected and abs(float(expected[field]) - float(actual)) > tolerance:
        errors.append(f"{key}: {field} mismatch, state={actual}, manifest={expected[field]}.")


def _finish(errors: list[str], warnings: list[str], strict: bool) -> None:
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors or (strict and warnings):
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
