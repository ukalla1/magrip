"""Run M5 MaGRIP mask/weight training."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--retained-ratio", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--calibration-source", choices=("dataset", "text"), default="dataset")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--eval-dataset-split", default="validation")
    parser.add_argument("--eval-num-samples", type=int, default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--no-cache-baseline", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--train-weights", action="store_true")
    parser.add_argument("--mask-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-learning-rate", type=float, default=1e-5)
    parser.add_argument("--budget-penalty-weight", type=float, default=1.0)
    parser.add_argument("--mask-regularization-weight", type=float, default=0.0)
    parser.add_argument("--budget-warmup-steps", type=int, default=0)
    parser.add_argument("--penalty-warmup-steps", type=int, default=0)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.05)
    parser.add_argument("--temperature-decay", type=float, default=0.99)
    parser.add_argument("--mask-update-frequency", type=int, default=1)
    parser.add_argument("--clip-mask-grad-norm", type=float, default=1.0)
    parser.add_argument("--recompute-saliency-every", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--stabilization-steps", type=int, default=0)
    parser.add_argument("--no-final-harden", action="store_true")
    parser.add_argument("--final-recovery-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from magrip.config import (
        MaGRIPConfig,
        MaskScheduleConfig,
        ObjectiveConfig,
        OptimizerConfig,
        TrainingConfig,
    )
    from magrip.data import (
        batches_token_count,
        load_inline_text_batches,
        load_text_calibration_batches,
    )
    from magrip.discovery import discover_ffn_targets
    from magrip.logging import RunLogger, cuda_memory_snapshot, system_info, tensor_stats
    from magrip.masks import save_mask_state, total_mask_cost
    from magrip.trainer import MaGRIPTrainer, config_to_dict, training_result_to_summary

    model_slug = args.model_name.replace("/", "__")
    run_name = args.run_name or time.strftime(f"{model_slug}_magrip_train_%Y%m%d_%H%M%S")
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

    models_dir = Path(args.models_dir)
    baseline_dir = models_dir / "Baselines" / model_slug
    should_cache_baseline = not args.no_cache_baseline or args.save_baseline
    baseline_cache_hit = baseline_dir.exists()
    load_source = baseline_dir if baseline_cache_hit else args.model_name
    token = os.environ.get(args.hf_token_env)
    auth_kwargs = {"token": token} if token else {}

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(load_source, **auth_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        load_source,
        torch_dtype=parse_torch_dtype(args.torch_dtype, torch),
        **auth_kwargs,
    )
    model.to(device)
    model.eval()
    baseline_saved_after_download = False
    if should_cache_baseline and not baseline_cache_hit:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(baseline_dir)
        tokenizer.save_pretrained(baseline_dir)
        baseline_saved_after_download = True

    logger.log(
        "model_loaded",
        elapsed_seconds=time.perf_counter() - load_start,
        model_name=args.model_name,
        load_source=str(load_source),
        baseline_cache_dir=baseline_dir,
        baseline_cache_hit=baseline_cache_hit,
        baseline_saved_after_download=baseline_saved_after_download,
        hf_token_env=args.hf_token_env,
        hf_token_present=bool(token),
        parameter_count=count_parameters(model),
        trainable_parameter_count=count_trainable_parameters(model),
        torch_dtype=str(next(model.parameters()).dtype),
        cuda_memory=cuda_memory_snapshot(),
    )

    data_start = time.perf_counter()
    if args.calibration_source == "dataset":
        batches = load_text_calibration_batches(
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
        eval_batches = load_text_calibration_batches(
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.eval_dataset_split,
            text_column=args.text_column,
            num_samples=args.eval_num_samples or args.num_samples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        batches = load_inline_text_batches(
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
        eval_batches = batches
    logger.log(
        "calibration_data_prepared",
        elapsed_seconds=time.perf_counter() - data_start,
        calibration_source=args.calibration_source,
        dataset_name=args.dataset_name if args.calibration_source == "dataset" else None,
        dataset_config=args.dataset_config if args.calibration_source == "dataset" else None,
        dataset_split=args.dataset_split if args.calibration_source == "dataset" else None,
        eval_dataset_split=args.eval_dataset_split if args.calibration_source == "dataset" else None,
        text_column=args.text_column if args.calibration_source == "dataset" else None,
        num_batches=len(batches),
        eval_num_batches=len(eval_batches),
        num_samples=args.num_samples,
        eval_num_samples=args.eval_num_samples or args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        token_count=batches_token_count(batches),
        eval_token_count=batches_token_count(eval_batches),
    )

    targets = list(discover_ffn_targets(model))
    logger.log(
        "targets_discovered",
        target_count=len(targets),
        targets=[target_to_dict(target) for target in targets],
    )

    config = MaGRIPConfig(
        objective=ObjectiveConfig(
            target_retained_ratio=args.retained_ratio,
            budget_penalty_weight=args.budget_penalty_weight,
            mask_regularization_weight=args.mask_regularization_weight,
            budget_warmup_steps=args.budget_warmup_steps,
            penalty_warmup_steps=args.penalty_warmup_steps,
        ),
        mask_schedule=MaskScheduleConfig(
            initial_temperature=args.initial_temperature,
            min_temperature=args.min_temperature,
            temperature_decay=args.temperature_decay,
            mask_update_frequency=args.mask_update_frequency,
        ),
        training=TrainingConfig(
            max_steps=args.max_steps,
            train_weights=args.train_weights,
            train_masks=True,
            recompute_saliency_every=args.recompute_saliency_every,
            checkpoint_every=args.checkpoint_every,
            stabilization_steps=args.stabilization_steps,
            final_harden=not args.no_final_harden,
            final_recovery_steps=args.final_recovery_steps,
            clip_mask_grad_norm=args.clip_mask_grad_norm,
        ),
        optimizer=OptimizerConfig(
            mask_learning_rate=args.mask_learning_rate,
            weight_learning_rate=args.weight_learning_rate,
        ),
    )

    trainer = MaGRIPTrainer(model=model, config=config, targets=targets)
    train_start = time.perf_counter()
    result = trainer.train(
        batches=batches,
        eval_batches=eval_batches,
        checkpoint_dir=run_dir / "checkpoints" if args.checkpoint_every else None,
    )
    logger.log(
        "training_completed",
        elapsed_seconds=time.perf_counter() - train_start,
        num_steps=result.num_steps,
        baseline_loss=result.baseline_loss,
        initial_masked_loss=result.initial_masked_loss,
        final_masked_loss=result.final_masked_loss,
        cuda_memory=cuda_memory_snapshot(),
    )

    pruned_dir = models_dir / "Pruned" / f"{model_slug}_magrip_train"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            key: mask.binary_values.detach().cpu()
            for key, mask in result.masks.as_dict().items()
        },
        pruned_dir / "masks.pt",
    )
    save_mask_state(pruned_dir / "mask_state.pt", result.masks)
    training_summary = training_result_to_summary(result)

    summary = {
        "model_name": args.model_name,
        "run_dir": str(run_dir),
        "pruned_dir": str(pruned_dir),
        "baseline_cache": {
            "enabled": should_cache_baseline,
            "path": str(baseline_dir),
            "load_source": str(load_source),
            "hit": baseline_cache_hit,
            "saved_after_download": baseline_saved_after_download,
        },
        "torch_dtype": str(next(model.parameters()).dtype),
        "calibration": {
            "source": args.calibration_source,
            "dataset_name": args.dataset_name if args.calibration_source == "dataset" else None,
            "dataset_config": args.dataset_config if args.calibration_source == "dataset" else None,
            "dataset_split": args.dataset_split if args.calibration_source == "dataset" else None,
            "eval_dataset_split": (
                args.eval_dataset_split if args.calibration_source == "dataset" else None
            ),
            "text_column": args.text_column if args.calibration_source == "dataset" else None,
            "num_samples": args.num_samples,
            "eval_num_samples": args.eval_num_samples or args.num_samples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "num_batches": len(batches),
            "eval_num_batches": len(eval_batches),
            "num_tokens": batches_token_count(batches),
            "eval_num_tokens": batches_token_count(eval_batches),
        },
        "config": config_to_dict(config),
        "targets": [target_to_dict(target) for target in result.targets],
        "mask_summaries": mask_summaries(result.masks, tensor_stats),
        "saliency_summaries": saliency_summaries(result.initial_saliency, tensor_stats),
        "training": training_summary,
        "baseline_loss": result.baseline_loss,
        "masked_loss": result.final_masked_loss,
        "baseline_perplexity": result.baseline_perplexity,
        "masked_perplexity": result.final_masked_perplexity,
        "loss_delta": result.final_masked_loss - result.baseline_loss,
        "perplexity_delta": result.final_masked_perplexity - result.baseline_perplexity,
        "mask_cost": training_summary["mask_cost"],
        "elapsed_seconds": time.perf_counter() - wall_start,
    }
    (pruned_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_metric_artifacts(run_dir, summary)
    logger.log(
        "artifacts_saved",
        pruned_dir=pruned_dir,
        mask_file=pruned_dir / "masks.pt",
        mask_state_file=pruned_dir / "mask_state.pt",
    )
    logger.write_summary(summary)

    print(json.dumps(summary, indent=2))
    print(f"Saved trained masks to {pruned_dir}")
    print(f"Saved run logs to {run_dir}")


def count_parameters(model: object) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def count_trainable_parameters(model: object) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def parse_torch_dtype(name: str, torch: object) -> object:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


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


def mask_summaries(masks: object, tensor_stats: object) -> dict[str, object]:
    return {
        key: {
            "active_channels": mask.active_channels,
            "total_channels": mask.total_channels,
            "retained_ratio": mask.retained_ratio,
            "full_cost": mask.cost_summary.full_cost,
            "retained_cost": mask.cost_summary.retained_cost,
            "cost_retained_ratio": mask.cost_summary.retained_ratio,
            "full_flop_cost": mask.cost_summary.full_flop_cost,
            "retained_flop_cost": mask.cost_summary.retained_flop_cost,
            "flop_cost_retained_ratio": mask.cost_summary.flop_retained_ratio,
            "values": tensor_stats(mask.binary_values),
            "probabilities": tensor_stats(mask.probabilities),
        }
        for key, mask in masks.as_dict().items()
    }


def saliency_summaries(saliency: object, tensor_stats: object) -> dict[str, object]:
    return {
        key: {
            "magnitude": tensor_stats(saliency.magnitude[key]),
            "gradient": tensor_stats(saliency.gradient[key]),
            "combined": tensor_stats(saliency.combined()[key]),
            "metadata": saliency.metadata.get(key, {}),
            "branch_diagnostics": {
                module_path: {
                    "magnitude": tensor_stats(branch_magnitude),
                    "gradient": tensor_stats(saliency.branch_gradient[key][module_path]),
                }
                for module_path, branch_magnitude in saliency.branch_magnitude
                .get(key, {})
                .items()
            },
        }
        for key in saliency.magnitude
    }


def write_metric_artifacts(run_dir: Path, summary: dict[str, object]) -> None:
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    training = summary.get("training", {})
    metrics = training.get("metrics", []) if isinstance(training, dict) else []
    payload = {
        "model_name": summary.get("model_name"),
        "calibration": summary.get("calibration"),
        "config": summary.get("config"),
        "baseline_loss": summary.get("baseline_loss"),
        "masked_loss": summary.get("masked_loss"),
        "baseline_perplexity": summary.get("baseline_perplexity"),
        "masked_perplexity": summary.get("masked_perplexity"),
        "loss_delta": summary.get("loss_delta"),
        "perplexity_delta": summary.get("perplexity_delta"),
        "mask_cost": summary.get("mask_cost"),
        "metrics": metrics,
    }
    with (metrics_dir / "metrics.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    (metrics_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    if metrics:
        fields = [
            "step",
            "temperature",
            "mask_update",
            "weight_update",
            "total_loss",
            "task_loss",
            "budget_penalty",
            "retained_cost_ratio",
            "target_retained_ratio",
            "hard_retained_cost_ratio",
            "mask_grad_norm",
        ]
        rows = [",".join(fields)]
        for item in metrics:
            objective = item.get("objective", {})
            row = [
                item.get("step"),
                item.get("temperature"),
                item.get("mask_update"),
                item.get("weight_update"),
                objective.get("total_loss"),
                objective.get("task_loss"),
                objective.get("budget_penalty"),
                objective.get("retained_cost_ratio"),
                objective.get("target_retained_ratio"),
                item.get("hard_retained_cost_ratio"),
                item.get("mask_grad_norm"),
            ]
            rows.append(",".join("" if value is None else str(value) for value in row))
        (metrics_dir / "metrics.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
