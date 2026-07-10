"""Inspect discovered MaGRIP FFN targets for a Hugging Face causal LM."""

from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM

from magrip.discovery import discover_ffn_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    targets = discover_ffn_targets(model)
    if not targets:
        print("No FFN targets discovered.")
        return

    for target in targets:
        print(
            f"[{target.block_index:03d}] {target.topology.value} "
            f"{target.ffn_path} hidden={target.hidden_size} "
            f"intermediate={target.intermediate_size}"
        )


if __name__ == "__main__":
    main()
