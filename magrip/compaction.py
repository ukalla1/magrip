"""Structural compaction utilities for hardened FFN masks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from magrip.masks import MaskCollection, StructuredMask
from magrip.module_utils import get_module_by_path, set_module_by_path
from magrip.topology import FFNTopologyKind


@dataclass(frozen=True)
class TargetCompactionSummary:
    """Compaction summary for one FFN target."""

    ffn_path: str
    topology: str
    original_channels: int
    retained_channels: int
    expand_module_paths: tuple[str, ...]
    contract_module_paths: tuple[str, ...]

    @property
    def retained_ratio(self) -> float:
        if self.original_channels <= 0:
            return 0.0
        return self.retained_channels / self.original_channels


@dataclass(frozen=True)
class CompactionReport:
    """Summary returned by structural compaction."""

    targets: list[TargetCompactionSummary] = field(default_factory=list)
    config_updates: dict[str, Any] = field(default_factory=dict)

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def full_channels(self) -> int:
        return sum(target.original_channels for target in self.targets)

    @property
    def retained_channels(self) -> int:
        return sum(target.retained_channels for target in self.targets)

    @property
    def retained_ratio(self) -> float:
        if self.full_channels <= 0:
            return 0.0
        return self.retained_channels / self.full_channels

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_count": self.target_count,
            "full_channels": self.full_channels,
            "retained_channels": self.retained_channels,
            "retained_ratio": self.retained_ratio,
            "config_updates": self.config_updates,
            "targets": [
                {
                    "ffn_path": target.ffn_path,
                    "topology": target.topology,
                    "original_channels": target.original_channels,
                    "retained_channels": target.retained_channels,
                    "retained_ratio": target.retained_ratio,
                    "expand_module_paths": list(target.expand_module_paths),
                    "contract_module_paths": list(target.contract_module_paths),
                }
                for target in self.targets
            ],
        }


def compaction_available() -> bool:
    """Return whether structural compaction has been implemented."""

    return True


def compact_model_inplace(
    model: nn.Module,
    masks: Mapping[str, StructuredMask] | MaskCollection,
    *,
    update_config: bool = True,
) -> CompactionReport:
    """Physically remove masked FFN intermediate channels from ``model``.

    Dense FFNs compact expansion outputs and contraction inputs. Gated FFNs apply the
    same retained channel indices to every expansion branch and to the contraction input.
    The operation is exact for hardened binary masks because the removed channels are
    precisely the channels that the masked model multiplies by zero.
    """

    mask_map = masks.as_dict() if isinstance(masks, MaskCollection) else dict(masks)
    summaries: list[TargetCompactionSummary] = []
    retained_counts: dict[str, int] = {}

    for key, mask in mask_map.items():
        target = mask.target
        if target.topology not in {FFNTopologyKind.DENSE, FFNTopologyKind.GATED}:
            raise ValueError(
                f"Compaction for {target.topology.value!r} target {target.ffn_path} "
                "is not supported in M7."
            )
        keep = _kept_channel_indices(mask)
        original_channels = int(mask.total_channels)
        retained_channels = int(keep.numel())
        if retained_channels <= 0:
            raise ValueError(f"Mask for {key} retained zero channels.")

        for module_path in target.expand_module_paths:
            module = get_module_by_path(model, module_path)
            replacement = compact_module_output_channels(module, keep)
            set_module_by_path(model, module_path, replacement)

        for module_path in target.contract_module_paths:
            module = get_module_by_path(model, module_path)
            replacement = compact_module_input_channels(module, keep)
            set_module_by_path(model, module_path, replacement)

        _update_ffn_module_attributes(model, target.ffn_path, retained_channels)
        retained_counts[target.ffn_path] = retained_channels
        summaries.append(
            TargetCompactionSummary(
                ffn_path=target.ffn_path,
                topology=target.topology.value,
                original_channels=original_channels,
                retained_channels=retained_channels,
                expand_module_paths=target.expand_module_paths,
                contract_module_paths=target.contract_module_paths,
            )
        )

    config_updates = (
        update_model_config_for_compaction(model, retained_counts) if update_config else {}
    )
    return CompactionReport(targets=summaries, config_updates=config_updates)


def compact_module_output_channels(module: object, keep: Tensor) -> nn.Module:
    """Return a copy of ``module`` with output channels restricted to ``keep``."""

    keep = keep.detach().cpu().long()
    if isinstance(module, nn.Linear):
        return _compact_linear_output(module, keep)
    if _is_hf_conv1d(module):
        return _compact_conv1d_output(module, keep)
    raise TypeError(f"Unsupported expansion module type for compaction: {type(module).__name__}.")


def compact_module_input_channels(module: object, keep: Tensor) -> nn.Module:
    """Return a copy of ``module`` with input channels restricted to ``keep``."""

    keep = keep.detach().cpu().long()
    if isinstance(module, nn.Linear):
        return _compact_linear_input(module, keep)
    if _is_hf_conv1d(module):
        return _compact_conv1d_input(module, keep)
    raise TypeError(f"Unsupported contraction module type for compaction: {type(module).__name__}.")


def update_model_config_for_compaction(
    model: nn.Module,
    retained_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Update common Hugging Face config fields for uniform FFN compaction.

    Most Transformers model classes assume one global FFN intermediate size in their
    config. Saving in vanilla Hugging Face format is therefore only reloadable when all
    compacted FFN targets retain the same number of channels.
    """

    if not retained_counts or not hasattr(model, "config"):
        return {}
    unique_counts = sorted(set(int(value) for value in retained_counts.values()))
    if len(unique_counts) != 1:
        return {
            "hf_config_updated": False,
            "reason": "nonuniform per-layer retained channel counts",
            "retained_channel_counts": unique_counts,
        }

    retained = unique_counts[0]
    updates: dict[str, Any] = {
        "hf_config_updated": False,
        "retained_intermediate_size": retained,
    }
    config = model.config
    changed: dict[str, Any] = {}
    for field_name in ("intermediate_size", "n_inner", "ffn_dim"):
        if hasattr(config, field_name):
            old_value = getattr(config, field_name)
            setattr(config, field_name, retained)
            changed[field_name] = {"old": old_value, "new": retained}
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for field_name in ("intermediate_size", "n_inner", "ffn_dim"):
            if hasattr(text_config, field_name):
                old_value = getattr(text_config, field_name)
                setattr(text_config, field_name, retained)
                changed[f"text_config.{field_name}"] = {"old": old_value, "new": retained}
    updates["hf_config_updated"] = bool(changed)
    updates["fields"] = changed
    return updates


def _kept_channel_indices(mask: StructuredMask) -> Tensor:
    binary = mask.binary_values.detach().cpu().bool()
    if binary.ndim != 1:
        raise ValueError(f"Expected a 1D mask for {mask.target.ffn_path}.")
    return torch.nonzero(binary, as_tuple=False).flatten().long()


def _compact_linear_output(module: nn.Linear, keep: Tensor) -> nn.Linear:
    replacement = nn.Linear(
        in_features=module.in_features,
        out_features=int(keep.numel()),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(module.weight.index_select(0, keep.to(module.weight.device)))
        if module.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(module.bias.index_select(0, keep.to(module.bias.device)))
    replacement.training = module.training
    return replacement


def _compact_linear_input(module: nn.Linear, keep: Tensor) -> nn.Linear:
    replacement = nn.Linear(
        in_features=int(keep.numel()),
        out_features=module.out_features,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(module.weight.index_select(1, keep.to(module.weight.device)))
        if module.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(module.bias)
    replacement.training = module.training
    return replacement


def _compact_conv1d_output(module: object, keep: Tensor) -> nn.Module:
    weight = module.weight
    bias = module.bias
    replacement = type(module)(int(keep.numel()), int(weight.shape[0]))
    replacement = replacement.to(device=weight.device, dtype=weight.dtype)
    keep_device = keep.to(weight.device)
    with torch.no_grad():
        replacement.weight.copy_(weight.index_select(1, keep_device))
        replacement.bias.copy_(bias.index_select(0, keep.to(bias.device)))
    replacement.training = module.training
    return replacement


def _compact_conv1d_input(module: object, keep: Tensor) -> nn.Module:
    weight = module.weight
    bias = module.bias
    replacement = type(module)(int(weight.shape[1]), int(keep.numel()))
    replacement = replacement.to(device=weight.device, dtype=weight.dtype)
    keep_device = keep.to(weight.device)
    with torch.no_grad():
        replacement.weight.copy_(weight.index_select(0, keep_device))
        replacement.bias.copy_(bias)
    replacement.training = module.training
    return replacement


def _is_hf_conv1d(module: object) -> bool:
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    return (
        hasattr(module, "nf")
        and weight is not None
        and bias is not None
        and getattr(weight, "ndim", None) == 2
        and getattr(bias, "ndim", None) == 1
    )


def _update_ffn_module_attributes(model: nn.Module, ffn_path: str, retained_channels: int) -> None:
    try:
        ffn = get_module_by_path(model, ffn_path)
    except (AttributeError, IndexError, TypeError):
        return
    for field_name in (
        "intermediate_size",
        "intermediate_features",
        "hidden_features",
        "ffn_dim",
    ):
        if hasattr(ffn, field_name):
            try:
                setattr(ffn, field_name, retained_channels)
            except (AttributeError, TypeError):
                pass
