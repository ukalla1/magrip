"""Inspect discovered MaGRIP FFN targets for a Hugging Face causal LM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magrip.discovery import discover_ffn_topology
from magrip.validation import validate_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--no-cache-baseline", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM

    args = parse_args()
    model_slug = args.model_name.replace("/", "__")
    baseline_dir = Path(args.models_dir) / "Baselines" / model_slug
    load_source = baseline_dir if baseline_dir.exists() and not args.no_cache_baseline else args.model_name
    token = os.environ.get(args.hf_token_env)
    auth_kwargs = {"token": token} if token else {}

    model = AutoModelForCausalLM.from_pretrained(load_source, **auth_kwargs)
    report = discover_ffn_topology(model)
    validation = validate_targets(report.targets)

    if args.json:
        print(
            json.dumps(
                {
                    "model_name": args.model_name,
                    "load_source": str(load_source),
                    "targets": [target_to_dict(target) for target in report.targets],
                    "issues": [issue.__dict__ for issue in report.issues],
                    "validation_errors": validation.errors,
                    "validation_warnings": validation.warnings,
                },
                indent=2,
            )
        )
        return

    print(f"Model: {args.model_name}")
    print(f"Load source: {load_source}")
    print(f"Discovered targets: {len(report.targets)}")
    if not report.targets:
        print("No FFN targets discovered.")

    for target in report.targets:
        expand = ", ".join(target.expand_module_paths)
        contract = ", ".join(target.contract_module_paths)
        print(
            f"[{target.block_index:03d}] {target.topology.value:<7} "
            f"{target.ffn_path} hidden={target.hidden_size} "
            f"intermediate={target.intermediate_size} "
            f"expand=[{expand}] contract=[{contract}]"
        )

    if report.issues:
        print("\nDiscovery issues:")
        for issue in report.issues:
            print(f"- {issue.severity.upper()} {issue.path}: {issue.reason}")

    if validation.errors or validation.warnings:
        print("\nValidation:")
        for warning in validation.warnings:
            print(f"- WARNING: {warning}")
        for error in validation.errors:
            print(f"- ERROR: {error}")
    else:
        print("\nValidation: OK")


def target_to_dict(target: object) -> dict[str, object]:
    return {
        "block_index": target.block_index,
        "block_path": target.block_path,
        "ffn_path": target.ffn_path,
        "topology": target.topology.value,
        "registry_name": target.registry_name,
        "expand_module_paths": list(target.expand_module_paths),
        "contract_module_paths": list(target.contract_module_paths),
        "intermediate_size": target.intermediate_size,
        "hidden_size": target.hidden_size,
    }


if __name__ == "__main__":
    main()
