"""Run a small GPT-2 dense-FFN MaGRIP smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--retained-ratio", type=float, default=0.7)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--calibration-source", choices=("dataset", "text"), default="dataset")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--text",
        default=(
            "Magnitude and gradient informed pruning estimates which feed-forward "
            "channels can be removed while preserving language model behavior."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from magrip.baseline import run_frozen_pruning_baseline
    from magrip.baseline import run_frozen_pruning_baseline_on_batches
    from magrip.data import (
        batches_token_count,
        load_inline_text_batches,
        load_text_calibration_batches,
    )
    from magrip.discovery import discover_ffn_targets
    from magrip.logging import RunLogger, cuda_memory_snapshot, system_info, tensor_stats

    run_name = args.run_name or time.strftime("gpt2_smoke_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    logger = RunLogger(run_dir)
    device = torch.device(args.device)
    wall_start = time.perf_counter()

    logger.log(
        "run_started",
        args=vars(args),
        run_dir=run_dir,
        system=system_info(),
        cuda_memory=cuda_memory_snapshot(),
    )

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    logger.log(
        "model_loaded",
        elapsed_seconds=time.perf_counter() - load_start,
        model_name=args.model_name,
        parameter_count=count_parameters(model),
        trainable_parameter_count=count_trainable_parameters(model),
        tokenizer_vocab_size=len(tokenizer),
        cuda_memory=cuda_memory_snapshot(),
    )

    data_start = time.perf_counter()
    if args.calibration_source == "dataset":
        calibration_batches = load_text_calibration_batches(
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            num_samples=args.num_samples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        calibration_batches = load_inline_text_batches(
            tokenizer=tokenizer,
            text=args.text,
            num_samples=args.num_samples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
    logger.log(
        "calibration_data_prepared",
        elapsed_seconds=time.perf_counter() - data_start,
        calibration_source=args.calibration_source,
        dataset_name=args.dataset_name if args.calibration_source == "dataset" else None,
        dataset_config=args.dataset_config if args.calibration_source == "dataset" else None,
        dataset_split=args.dataset_split if args.calibration_source == "dataset" else None,
        text_column=args.text_column if args.calibration_source == "dataset" else None,
        num_batches=len(calibration_batches),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        token_count=batches_token_count(calibration_batches),
    )

    targets = list(discover_ffn_targets(model))
    logger.log(
        "targets_discovered",
        target_count=len(targets),
        targets=[target_to_dict(target) for target in targets],
    )

    prune_start = time.perf_counter()
    if len(calibration_batches) == 1:
        result = run_frozen_pruning_baseline(
            model=model,
            input_ids=calibration_batches[0],
            retained_ratio=args.retained_ratio,
            targets=targets,
        )
    else:
        result = run_frozen_pruning_baseline_on_batches(
            model=model,
            batches=calibration_batches,
            retained_ratio=args.retained_ratio,
            targets=targets,
        )
    prune_elapsed = time.perf_counter() - prune_start
    logger.log(
        "frozen_pruning_completed",
        elapsed_seconds=prune_elapsed,
        baseline_loss=result.baseline_loss,
        masked_loss=result.masked_loss,
        baseline_perplexity=result.baseline_perplexity,
        masked_perplexity=result.masked_perplexity,
        num_batches=result.num_batches,
        num_tokens=result.num_tokens,
        loss_delta=result.masked_loss - result.baseline_loss,
        perplexity_delta=result.masked_perplexity - result.baseline_perplexity,
        cuda_memory=cuda_memory_snapshot(),
    )

    model_slug = args.model_name.replace("/", "__")
    models_dir = Path(args.models_dir)
    baseline_dir = models_dir / "Baselines" / model_slug
    pruned_dir = models_dir / "Pruned" / f"{model_slug}_magrip_smoke"
    pruned_dir.mkdir(parents=True, exist_ok=True)

    if args.save_baseline:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(baseline_dir)
        tokenizer.save_pretrained(baseline_dir)
        logger.log("baseline_saved", path=baseline_dir)

    torch.save(
        {
            key: mask.values.detach().cpu()
            for key, mask in result.masks.items()
        },
        pruned_dir / "masks.pt",
    )
    mask_summaries = {
        key: {
            "active_channels": mask.active_channels,
            "total_channels": mask.total_channels,
            "retained_ratio": mask.retained_ratio,
            "values": tensor_stats(mask.values),
        }
        for key, mask in result.masks.items()
    }
    saliency_summaries = {
        key: {
            "magnitude": tensor_stats(result.saliency.magnitude[key]),
            "gradient": tensor_stats(result.saliency.gradient[key]),
            "combined": tensor_stats(result.saliency.combined()[key]),
        }
        for key in result.saliency.magnitude
    }
    manifest = {
        "model_name": args.model_name,
        "retained_ratio": args.retained_ratio,
        "calibration": {
            "source": args.calibration_source,
            "dataset_name": args.dataset_name if args.calibration_source == "dataset" else None,
            "dataset_config": args.dataset_config if args.calibration_source == "dataset" else None,
            "dataset_split": args.dataset_split if args.calibration_source == "dataset" else None,
            "text_column": args.text_column if args.calibration_source == "dataset" else None,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "num_batches": result.num_batches,
            "num_tokens": result.num_tokens,
        },
        "run_dir": str(run_dir),
        "pruned_dir": str(pruned_dir),
        "baseline_dir": str(baseline_dir) if args.save_baseline else None,
        "targets": [
            target_to_dict(target)
            for target in result.targets
        ],
        "baseline_loss": result.baseline_loss,
        "masked_loss": result.masked_loss,
        "baseline_perplexity": result.baseline_perplexity,
        "masked_perplexity": result.masked_perplexity,
        "loss_delta": result.masked_loss - result.baseline_loss,
        "perplexity_delta": result.masked_perplexity - result.baseline_perplexity,
        "mask_summaries": mask_summaries,
        "saliency_summaries": saliency_summaries,
        "elapsed_seconds": time.perf_counter() - wall_start,
    }
    (pruned_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.log("artifacts_saved", pruned_dir=pruned_dir, mask_file=pruned_dir / "masks.pt")
    logger.write_summary(manifest)

    print(json.dumps(manifest, indent=2))
    print(f"Saved smoke masks to {pruned_dir}")
    print(f"Saved run logs to {run_dir}")


def count_parameters(model: object) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def count_trainable_parameters(model: object) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def target_to_dict(target: object) -> dict[str, object]:
    return {
        "block_index": target.block_index,
        "block_path": target.block_path,
        "ffn_path": target.ffn_path,
        "topology": target.topology.value,
        "expand_module_paths": list(target.expand_module_paths),
        "contract_module_paths": list(target.contract_module_paths),
        "intermediate_size": target.intermediate_size,
        "hidden_size": target.hidden_size,
    }


if __name__ == "__main__":
    main()
