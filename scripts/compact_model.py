"""Compact a MaGRIP-pruned model from hardened structured masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="HF model id or local model path.")
    parser.add_argument(
        "--mask-state",
        required=True,
        help="Path to MaGRIP mask_state.pt produced after final hardening.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional MaGRIP training checkpoint containing model_state_dict.",
    )
    parser.add_argument(
        "--checkpoint-is-trusted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow loading a full MaGRIP checkpoint with weights_only=False when PyTorch "
            "safe loading rejects optimizer objects. Disable for untrusted checkpoints."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the compacted Hugging Face model will be saved.",
    )
    parser.add_argument("--device", default="cpu", help="Device used for loading and verification.")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Torch dtype for loading the model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers loaders.",
    )
    parser.add_argument(
        "--tokenizer-source",
        help=(
            "Optional tokenizer source used when preserving GGUF tokenizer assets. "
            "Defaults to --model-name."
        ),
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--verify-text",
        action="append",
        default=[],
        help="Text prompt used for masked-vs-compacted logit equivalence checks.",
    )
    parser.add_argument("--verify-max-length", type=int, default=128)
    parser.add_argument("--verify-atol", type=float, default=2e-2)
    parser.add_argument("--verify-rtol", type=float, default=2e-2)
    parser.add_argument(
        "--local-target-policy",
        choices=("dtype-aware", "strict-allclose"),
        default="dtype-aware",
        help=(
            "Local FFN-target equivalence rule. `dtype-aware` accepts tiny BF16/FP16 "
            "roundoff drift using relative error and sparse-outlier diagnostics; "
            "`strict-allclose` requires elementwise allclose."
        ),
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip masked-vs-compacted logit equivalence checks.",
    )
    parser.add_argument(
        "--incremental-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When verification is enabled, compact one FFN target at a time and check "
            "logits after each target. This localizes structural mismatches."
        ),
    )
    parser.add_argument(
        "--verification-policy",
        choices=("strict-logits", "local-targets"),
        default="strict-logits",
        help=(
            "`strict-logits` requires full masked-vs-compacted logits to match after "
            "each incremental step. `local-targets` accepts structural compaction when "
            "each compacted FFN target output matches its masked reference, while still "
            "logging full-logit drift."
        ),
    )
    parser.add_argument(
        "--debug-layer-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On verification drift, compare transformer-block outputs to localize amplification.",
    )
    parser.add_argument(
        "--enforce-hard-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force loaded masks into hard binary forward mode before verification and "
            "compaction. Structural compaction is only exact for binary masks."
        ),
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum shard size passed to save_pretrained.",
    )
    parser.add_argument(
        "--eval-num-samples",
        type=int,
        default=0,
        help="If positive, evaluate masked reference and compacted model on this many samples.",
    )
    parser.add_argument("--eval-dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--eval-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--eval-dataset-split", default="validation")
    parser.add_argument("--eval-text-column", default="text")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="After HF save, run llama.cpp conversion to GGUF.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        help="Path to a llama.cpp checkout containing convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--gguf-out",
        help="Output GGUF file. Defaults to <output-dir>.gguf when --export-gguf is set.",
    )
    parser.add_argument(
        "--gguf-outtype",
        default="bf16",
        help="Outtype passed to llama.cpp conversion, e.g. f16, bf16, q8_0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from magrip.compaction import compact_model_inplace
    from magrip.data import batches_token_count, load_text_calibration_batches
    from magrip.evaluation import causal_lm_loss, perplexity_from_loss
    from magrip.experiment import write_compaction_research_artifacts
    from magrip.masks import apply_structured_masks, load_mask_state
    from magrip.tokenizer_assets import save_tokenizer_assets

    dtype = _parse_torch_dtype(args.torch_dtype, torch)
    token = os.environ.get(args.hf_token_env)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if token:
        load_kwargs["token"] = token
    tokenizer_source = args.tokenizer_source or args.model_name
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if token:
        tokenizer_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    checkpoint_info = _load_checkpoint_if_requested(
        model,
        args.checkpoint,
        device,
        torch,
        trusted=args.checkpoint_is_trusted,
    )
    masks = load_mask_state(args.mask_state)
    mask_mode = _mask_mode_summary(masks)
    if args.enforce_hard_masks:
        _enforce_hard_masks(masks)
    hardened_mask_mode = _mask_mode_summary(masks)
    eval_batches = None
    evaluation = None
    if args.eval_num_samples > 0:
        eval_batches = load_text_calibration_batches(
            tokenizer=tokenizer,
            dataset_name=args.eval_dataset_name,
            dataset_config=args.eval_dataset_config,
            dataset_split=args.eval_dataset_split,
            text_column=args.eval_text_column,
            num_samples=args.eval_num_samples,
            max_length=args.verify_max_length,
            batch_size=args.eval_batch_size,
            device=device,
        )

    verification = None
    if not args.skip_verify:
        texts = args.verify_text or ["MaGRIP structural compaction equivalence check."]
        with torch.no_grad(), apply_structured_masks(model, masks):
            reference_logits = _collect_logits(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                max_length=args.verify_max_length,
            )

    if eval_batches is not None:
        with torch.no_grad(), apply_structured_masks(model, masks):
            masked_loss = _mean_loss(model, eval_batches, causal_lm_loss)
        evaluation = {
            "dataset_name": args.eval_dataset_name,
            "dataset_config": args.eval_dataset_config,
            "dataset_split": args.eval_dataset_split,
            "num_samples": args.eval_num_samples,
            "num_batches": len(eval_batches),
            "num_tokens": batches_token_count(eval_batches),
            "masked_reference": {
                "loss": masked_loss,
                "perplexity": perplexity_from_loss(masked_loss),
                "num_batches": len(eval_batches),
                "num_tokens": batches_token_count(eval_batches),
            },
        }

    if not args.skip_verify and args.incremental_verify:
        report, verification = _compact_incrementally_with_verification(
            model=model,
            masks=masks,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=args.verify_max_length,
            reference_logits=reference_logits,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
            verification_policy=args.verification_policy,
            debug_layer_outputs=args.debug_layer_outputs,
            local_target_policy=args.local_target_policy,
            torch_dtype=args.torch_dtype,
        )
    else:
        report = compact_model_inplace(model, masks, update_config=True)

        if not args.skip_verify:
            with torch.no_grad():
                compacted_logits = _collect_logits(
                    model=model,
                    tokenizer=tokenizer,
                    texts=args.verify_text
                    or ["MaGRIP structural compaction equivalence check."],
                    device=device,
                    max_length=args.verify_max_length,
                )
            verification = _compare_logits(
                reference_logits,
                compacted_logits,
                atol=args.verify_atol,
                rtol=args.verify_rtol,
            )
            if not verification["ok"]:
                raise SystemExit(
                    "Compacted logits did not match masked logits within tolerance: "
                    + json.dumps(verification)
                )

    if eval_batches is not None and evaluation is not None:
        compacted_loss = _mean_loss(model, eval_batches, causal_lm_loss)
        evaluation["compacted"] = {
            "loss": compacted_loss,
            "perplexity": perplexity_from_loss(compacted_loss),
            "num_batches": len(eval_batches),
            "num_tokens": batches_token_count(eval_batches),
        }
        evaluation["loss_delta_compacted_vs_masked_reference"] = (
            evaluation["compacted"]["loss"] - evaluation["masked_reference"]["loss"]
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer_info = save_tokenizer_assets(
        tokenizer=tokenizer,
        output_dir=output_dir,
        tokenizer_source=tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        require_sentencepiece_assets=args.export_gguf,
        token=token,
    )

    gguf = None
    if args.export_gguf:
        gguf = _export_gguf(
            hf_dir=output_dir,
            llama_cpp_dir=args.llama_cpp_dir,
            gguf_out=args.gguf_out,
            outtype=args.gguf_outtype,
        )

    manifest = {
        "model_name": args.model_name,
        "tokenizer": tokenizer_info,
        "hf_token_env": args.hf_token_env,
        "hf_token_present": bool(token),
        "mask_state": args.mask_state,
        "verification_policy": args.verification_policy,
        "verification_tolerances": {
            "atol": args.verify_atol,
            "rtol": args.verify_rtol,
            "local_target_policy": args.local_target_policy,
        },
        "mask_mode": {
            "loaded": mask_mode,
            "used_for_compaction": hardened_mask_mode,
            "enforce_hard_masks": args.enforce_hard_masks,
        },
        "checkpoint": checkpoint_info,
        "output_dir": str(output_dir),
        "compaction": report.to_dict(),
        "verification": verification,
        "evaluation": evaluation,
        "gguf": gguf,
    }
    research_artifacts = write_compaction_research_artifacts(
        output_dir=output_dir,
        manifest=manifest,
    )
    manifest["research_artifacts"] = research_artifacts
    (output_dir / "magrip_compaction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def _load_checkpoint_if_requested(
    model: object,
    checkpoint_path: str | None,
    device: object,
    torch: object,
    *,
    trusted: bool,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    checkpoint, load_mode = _load_checkpoint(path, "cpu", torch, trusted=trusted)
    state_dict = _extract_model_state_dict(checkpoint, path)
    if state_dict is None:
        raise ValueError(f"Checkpoint {path} does not contain model_state_dict.")
    incompatible = model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(path),
        "load_mode": load_mode,
        "step": checkpoint.get("step") if isinstance(checkpoint, dict) else None,
        "final": checkpoint.get("final") if isinstance(checkpoint, dict) else None,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _load_checkpoint(
    path: Path,
    device: object,
    torch: object,
    *,
    trusted: bool,
) -> tuple[object, str]:
    try:
        return torch.load(path, map_location=device, weights_only=True), "weights_only"
    except Exception as exc:
        if not trusted:
            raise RuntimeError(
                f"Safe checkpoint loading failed for {path}. The checkpoint may contain "
                "non-tensor optimizer objects. Re-run with --checkpoint-is-trusted only "
                "for checkpoints you created and trust."
            ) from exc
        return torch.load(path, map_location=device, weights_only=False), "trusted_full_pickle"


def _save_tokenizer_assets(
    *,
    tokenizer: object,
    output_dir: Path,
    tokenizer_source: str,
    trust_remote_code: bool,
    require_gguf_assets: bool,
) -> dict[str, Any]:
    saved_files = [str(path) for path in tokenizer.save_pretrained(output_dir)]
    info: dict[str, Any] = {
        "source": tokenizer_source,
        "saved_files": saved_files,
        "gguf_assets_required": require_gguf_assets,
        "sentencepiece_model": None,
        "fallbacks": [],
    }
    if not require_gguf_assets:
        return info

    sentencepiece_path = output_dir / "tokenizer.model"
    if sentencepiece_path.exists():
        info["sentencepiece_model"] = str(sentencepiece_path)
        return info
    if not _expects_sentencepiece_model(tokenizer, tokenizer_source):
        info["sentencepiece_model"] = None
        info["sentencepiece_required"] = False
        return info
    info["sentencepiece_required"] = True

    slow_info = _try_save_slow_tokenizer(
        output_dir=output_dir,
        tokenizer_source=tokenizer_source,
        trust_remote_code=trust_remote_code,
    )
    info["fallbacks"].append(slow_info)
    if sentencepiece_path.exists():
        info["sentencepiece_model"] = str(sentencepiece_path)
        return info

    copy_info = _try_copy_sentencepiece_model(tokenizer_source, output_dir)
    info["fallbacks"].append(copy_info)
    if sentencepiece_path.exists():
        info["sentencepiece_model"] = str(sentencepiece_path)
        return info

    raise FileNotFoundError(
        "GGUF export needs a SentencePiece tokenizer.model, but it was not found in "
        f"{output_dir} and could not be recovered from tokenizer source "
        f"{tokenizer_source!r}. Install sentencepiece or provide --tokenizer-source "
        "pointing to a Hugging Face repo/local directory that contains tokenizer.model."
    )


def _expects_sentencepiece_model(tokenizer: object, tokenizer_source: str) -> bool:
    tokenizer_class = type(tokenizer).__name__.lower()
    source = str(tokenizer_source).lower()
    markers = (
        "gemma",
        "llama",
        "mistral",
        "mixtral",
        "sentencepiece",
        "spm",
        "t5",
    )
    return any(marker in tokenizer_class or marker in source for marker in markers)


def _try_save_slow_tokenizer(
    *,
    output_dir: Path,
    tokenizer_source: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer

        slow_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        saved_files = [str(path) for path in slow_tokenizer.save_pretrained(output_dir)]
        return {
            "method": "slow_tokenizer_save",
            "ok": True,
            "saved_files": saved_files,
        }
    except Exception as exc:
        return {
            "method": "slow_tokenizer_save",
            "ok": False,
            "error": repr(exc),
        }


def _try_copy_sentencepiece_model(tokenizer_source: str, output_dir: Path) -> dict[str, Any]:
    destination = output_dir / "tokenizer.model"
    local_source = Path(tokenizer_source)
    if local_source.exists():
        candidate = local_source / "tokenizer.model"
        if candidate.exists():
            return _copy_tokenizer_model(candidate, destination, method="local_tokenizer_source")

    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=tokenizer_source,
                filename="tokenizer.model",
            )
        )
        return _copy_tokenizer_model(downloaded, destination, method="hf_hub_download")
    except Exception as exc:
        return {
            "method": "hf_hub_download",
            "ok": False,
            "error": repr(exc),
        }


def _copy_tokenizer_model(source: Path, destination: Path, *, method: str) -> dict[str, Any]:
    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "method": method,
        "ok": True,
        "source": str(source),
        "destination": str(destination),
    }


def _extract_model_state_dict(checkpoint: object, path: Path) -> dict[str, Any] | None:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} did not load as a dictionary.")
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
        return checkpoint
    return None


def _enforce_hard_masks(masks: dict[str, object]) -> None:
    for mask in masks.values():
        mask.hard = True
        mask.ste = False
        mask.trainable = False
        mask.logits.requires_grad_(False)


def _mask_mode_summary(masks: dict[str, object]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "target_count": len(masks),
        "hard": 0,
        "soft": 0,
        "ste": 0,
        "trainable": 0,
        "active_channels": 0,
        "total_channels": 0,
    }
    for mask in masks.values():
        if mask.hard:
            summary["hard"] += 1
        else:
            summary["soft"] += 1
        if mask.ste:
            summary["ste"] += 1
        if mask.logits.requires_grad:
            summary["trainable"] += 1
        summary["active_channels"] += int(mask.active_channels)
        summary["total_channels"] += int(mask.total_channels)
    total = max(1, int(summary["total_channels"]))
    summary["active_ratio"] = float(summary["active_channels"] / total)
    return summary


def _collect_logits(
    *,
    model: object,
    tokenizer: object,
    texts: list[str],
    device: object,
    max_length: int,
) -> list[Any]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)
    return [outputs.logits.detach().cpu().float()]


def _collect_module_outputs(
    *,
    model: object,
    tokenizer: object,
    texts: list[str],
    device: object,
    max_length: int,
    module_path: str,
) -> list[Any]:
    from magrip.module_utils import get_module_by_path

    captured = []
    module = get_module_by_path(model, module_path)

    def hook(module: object, inputs: tuple[object, ...], output: object) -> None:
        if hasattr(output, "detach"):
            captured.append(output.detach().cpu().float())
        elif isinstance(output, (tuple, list)) and output and hasattr(output[0], "detach"):
            captured.append(output[0].detach().cpu().float())

    handle = module.register_forward_hook(hook)
    try:
        _collect_logits(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=max_length,
        )
    finally:
        handle.remove()
    return captured


def _collect_many_module_outputs(
    *,
    model: object,
    tokenizer: object,
    texts: list[str],
    device: object,
    max_length: int,
    module_paths: list[str],
) -> dict[str, list[Any]]:
    from magrip.module_utils import get_module_by_path

    captured: dict[str, list[Any]] = {path: [] for path in module_paths}
    handles = []

    def make_hook(path: str):
        def hook(module: object, inputs: tuple[object, ...], output: object) -> None:
            if hasattr(output, "detach"):
                captured[path].append(output.detach().cpu().float())
            elif isinstance(output, (tuple, list)) and output and hasattr(output[0], "detach"):
                captured[path].append(output[0].detach().cpu().float())

        return hook

    try:
        for path in module_paths:
            module = get_module_by_path(model, path)
            handles.append(module.register_forward_hook(make_hook(path)))
        _collect_logits(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=max_length,
        )
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _compare_logits(
    reference: list[Any],
    compacted: list[Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import torch

    max_abs = 0.0
    mean_abs = 0.0
    squared_abs = 0.0
    mean_reference_abs = 0.0
    squared_reference = 0.0
    max_tolerance_violation = 0.0
    outlier_count = 0
    element_count = 0
    count = 0
    ok = len(reference) == len(compacted)
    for expected, actual in zip(reference, compacted):
        diff = (expected - actual).abs()
        tolerance = atol + rtol * expected.abs()
        violation = diff - tolerance
        over_tolerance = violation > 0
        max_abs = max(max_abs, float(diff.max().item()))
        mean_abs += float(diff.mean().item())
        squared_abs += float((diff * diff).mean().item())
        mean_reference_abs += float(expected.abs().mean().item())
        squared_reference += float((expected * expected).mean().item())
        max_tolerance_violation = max(
            max_tolerance_violation,
            float(violation.clamp_min(0).max().item()),
        )
        outlier_count += int(over_tolerance.sum().item())
        element_count += int(diff.numel())
        count += 1
        ok = ok and bool(torch.allclose(expected, actual, atol=atol, rtol=rtol))
    mean_abs_error = mean_abs / max(1, count)
    rmse_abs_error = (squared_abs / max(1, count)) ** 0.5
    mean_reference_abs_error = mean_reference_abs / max(1, count)
    reference_rmse = (squared_reference / max(1, count)) ** 0.5
    return {
        "ok": ok,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs_error,
        "rmse_abs_error": rmse_abs_error,
        "mean_reference_abs": mean_reference_abs_error,
        "reference_rmse": reference_rmse,
        "relative_mean_abs_error": mean_abs_error / max(1e-12, mean_reference_abs_error),
        "relative_rmse_error": rmse_abs_error / max(1e-12, reference_rmse),
        "max_tolerance_violation": max_tolerance_violation,
        "outlier_count": outlier_count,
        "element_count": element_count,
        "outlier_fraction": outlier_count / max(1, element_count),
        "batches": count,
        "reference_batches": len(reference),
        "actual_batches": len(compacted),
    }


def _local_target_check_ok(
    check: dict[str, Any],
    *,
    policy: str,
    torch_dtype: str,
) -> bool:
    if bool(check["ok"]):
        return True
    if policy == "strict-allclose":
        return False
    dtype_eps = _verification_dtype_epsilon(torch_dtype)
    relative_limit = 8.0 * dtype_eps
    outlier_limit = 0.01 if dtype_eps >= 1e-3 else 0.001
    return (
        float(check["relative_rmse_error"]) <= relative_limit
        and float(check["outlier_fraction"]) <= outlier_limit
    )


def _local_target_acceptance_summary(
    check: dict[str, Any],
    *,
    policy: str,
    torch_dtype: str,
) -> dict[str, Any]:
    dtype_eps = _verification_dtype_epsilon(torch_dtype)
    relative_limit = 8.0 * dtype_eps
    outlier_limit = 0.01 if dtype_eps >= 1e-3 else 0.001
    return {
        "policy": policy,
        "torch_dtype": torch_dtype,
        "strict_allclose": bool(check["ok"]),
        "accepted": _local_target_check_ok(
            check,
            policy=policy,
            torch_dtype=torch_dtype,
        ),
        "relative_rmse_error": check["relative_rmse_error"],
        "relative_rmse_limit": relative_limit,
        "outlier_fraction": check["outlier_fraction"],
        "outlier_fraction_limit": outlier_limit,
        "mean_abs_error": check["mean_abs_error"],
        "mean_reference_abs": check["mean_reference_abs"],
    }


def _verification_dtype_epsilon(torch_dtype: str) -> float:
    if torch_dtype == "bfloat16":
        return 2.0**-7
    if torch_dtype == "float16":
        return 2.0**-10
    if torch_dtype == "float32":
        return 2.0**-23
    return 2.0**-7


def _compare_module_output_maps(
    reference: dict[str, list[Any]],
    actual: dict[str, list[Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    path_checks = []
    first_failed_path = None
    for path in reference:
        check = _compare_logits(
            reference[path],
            actual.get(path, []),
            atol=atol,
            rtol=rtol,
        )
        path_check = {"path": path, **check}
        path_checks.append(path_check)
        if first_failed_path is None and not check["ok"]:
            first_failed_path = path
    return {
        "ok": first_failed_path is None,
        "first_failed_path": first_failed_path,
        "paths": path_checks,
    }


def _diagnostic_block_paths(masks: dict[str, object]) -> list[str]:
    paths = []
    seen = set()
    for mask in masks.values():
        path = mask.target.block_path
        if path and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _compact_incrementally_with_verification(
    *,
    model: object,
    masks: dict[str, object],
    tokenizer: object,
    texts: list[str],
    device: object,
    max_length: int,
    reference_logits: list[Any],
    atol: float,
    rtol: float,
    verification_policy: str,
    debug_layer_outputs: bool,
    local_target_policy: str,
    torch_dtype: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    from magrip.compaction import (
        CompactionReport,
        compact_model_inplace,
        update_model_config_for_compaction,
    )
    from magrip.masks import apply_structured_masks

    mask_map = dict(masks)
    remaining_masks = dict(mask_map)
    reports = []
    steps = []
    retained_counts: dict[str, int] = {}
    diagnostic_paths = _diagnostic_block_paths(mask_map) if debug_layer_outputs else []

    for index, (key, mask) in enumerate(mask_map.items(), start=1):
        with torch.no_grad(), apply_structured_masks(model, remaining_masks):
            expected_target_outputs = _collect_module_outputs(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                max_length=max_length,
                module_path=key,
            )
            expected_layer_outputs = (
                _collect_many_module_outputs(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    device=device,
                    max_length=max_length,
                    module_paths=diagnostic_paths,
                )
                if diagnostic_paths
                else None
            )

        report = compact_model_inplace(model, {key: mask}, update_config=False)
        reports.append(report)
        remaining_masks.pop(key, None)
        for target in report.targets:
            retained_counts[target.ffn_path] = target.retained_channels

        with torch.no_grad(), apply_structured_masks(model, remaining_masks):
            logits = _collect_logits(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                max_length=max_length,
            )
            actual_target_outputs = _collect_module_outputs(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                max_length=max_length,
                module_path=key,
            )
            actual_layer_outputs = (
                _collect_many_module_outputs(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    device=device,
                    max_length=max_length,
                    module_paths=diagnostic_paths,
                )
                if diagnostic_paths
                else None
            )
        check = _compare_logits(reference_logits, logits, atol=atol, rtol=rtol)
        target_output_check = _compare_logits(
            expected_target_outputs,
            actual_target_outputs,
            atol=atol,
            rtol=rtol,
        )
        local_target_ok = _local_target_check_ok(
            target_output_check,
            policy=local_target_policy,
            torch_dtype=torch_dtype,
        )
        strict_logits_ok = bool(check["ok"])
        policy_ok = strict_logits_ok or (
            verification_policy == "local-targets" and local_target_ok
        )
        step = {
            **check,
            "strict_logits_ok": strict_logits_ok,
            "local_target_ok": local_target_ok,
            "accepted_by_policy": policy_ok,
            "verification_policy": verification_policy,
            "step": index,
            "target": key,
            "remaining_masks": len(remaining_masks),
            "target_output": target_output_check,
            "local_target_acceptance": _local_target_acceptance_summary(
                target_output_check,
                policy=local_target_policy,
                torch_dtype=torch_dtype,
            ),
        }
        if not policy_ok and expected_layer_outputs is not None and actual_layer_outputs is not None:
            step["layer_outputs"] = _compare_module_output_maps(
                expected_layer_outputs,
                actual_layer_outputs,
                atol=atol,
                rtol=rtol,
            )
        steps.append(step)
        if not policy_ok:
            raise SystemExit(
                "Incremental compaction mismatch after target "
                f"{key}: "
                + json.dumps(
                    {
                        "failing_step": step,
                        "completed_steps": steps,
                    }
                )
            )

    config_updates = update_model_config_for_compaction(model, retained_counts)
    with torch.no_grad():
        final_logits = _collect_logits(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=max_length,
        )
    final_check = _compare_logits(reference_logits, final_logits, atol=atol, rtol=rtol)
    local_targets_ok = all(bool(step["local_target_ok"]) for step in steps)
    policy_ok = bool(final_check["ok"]) or (
        verification_policy == "local-targets" and local_targets_ok
    )
    verification = {
        **final_check,
        "ok": policy_ok,
        "strict_logits_ok": bool(final_check["ok"]),
        "local_targets_ok": local_targets_ok,
        "verification_policy": verification_policy,
        "accepted_by_policy": policy_ok,
        "mode": "incremental",
        "incremental_steps": steps,
    }
    if not policy_ok:
        raise SystemExit(
            "Compacted model did not satisfy the selected verification policy: "
            + json.dumps(verification)
        )

    targets = [target for report in reports for target in report.targets]
    return CompactionReport(targets=targets, config_updates=config_updates), verification


def _mean_loss(model: object, batches: list[Any], loss_fn: object) -> float:
    losses = [loss_fn(model, input_ids=batch, labels=batch) for batch in batches]
    if not losses:
        raise ValueError("Evaluation batches must not be empty.")
    return float(sum(losses) / len(losses))


def _export_gguf(
    *,
    hf_dir: Path,
    llama_cpp_dir: str | None,
    gguf_out: str | None,
    outtype: str,
) -> dict[str, Any]:
    if llama_cpp_dir is None:
        raise ValueError("--llama-cpp-dir is required with --export-gguf.")
    converter = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Could not find llama.cpp converter at {converter}.")
    output = Path(gguf_out) if gguf_out else hf_dir.with_suffix(".gguf")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(converter),
        str(hf_dir),
        "--outfile",
        str(output),
        "--outtype",
        outtype,
    ]
    subprocess.run(command, check=True)
    return {
        "path": str(output),
        "outtype": outtype,
        "converter": str(converter),
        "command": command,
    }


def _parse_torch_dtype(name: str, torch: object) -> object:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


if __name__ == "__main__":
    main()
