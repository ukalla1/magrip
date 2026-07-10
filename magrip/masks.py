"""Structured FFN mask parameterization and application utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.autograd import Function

from magrip.module_utils import get_module_by_path
from magrip.topology import FFNTarget, FFNTopologyKind


MASK_STATE_VERSION = 1


class BinaryMaskSTE(Function):
    """Straight-through estimator for hard binary masks."""

    @staticmethod
    def forward(ctx: object, probabilities: Tensor, threshold: Tensor) -> Tensor:
        return torch.where(
            probabilities >= threshold,
            torch.ones_like(probabilities),
            torch.zeros_like(probabilities),
        )

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        return torch.clamp(grad_output, min=-1.0, max=1.0), None


@dataclass(frozen=True)
class MaskCostSummary:
    """Cost accounting for one structured mask."""

    full_cost: float
    retained_cost: float
    full_flop_cost: float = 0.0
    retained_flop_cost: float = 0.0

    @property
    def retained_ratio(self) -> float:
        if self.full_cost <= 0.0:
            return 0.0
        return self.retained_cost / self.full_cost

    @property
    def flop_retained_ratio(self) -> float:
        if self.full_flop_cost <= 0.0:
            return 0.0
        return self.retained_flop_cost / self.full_flop_cost


class StructuredMask(nn.Module):
    """A structured channel mask associated with one FFN target.

    The mask has one scalar per discovered FFN intermediate channel. Dense FFNs use that
    scalar on the expansion output. Gated FFNs share the same scalar across all expansion
    branches, matching the structured unit definition in ``docs/THEORY.tex``.
    """

    def __init__(
        self,
        target: FFNTarget,
        logits: Tensor | None = None,
        values: Tensor | None = None,
        temperature: float = 1.0,
        threshold: float = 0.5,
        hard: bool = True,
        ste: bool = False,
        trainable: bool = False,
        cost_per_channel: Tensor | None = None,
        flop_cost_per_channel: Tensor | None = None,
    ) -> None:
        super().__init__()
        if logits is None and values is None:
            raise ValueError("Either logits or values must be provided.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        initial = logits if logits is not None else _values_to_logits(values)
        if initial is None:
            raise ValueError("Unable to initialize mask logits.")

        expected_channels = target.intermediate_size
        if expected_channels is not None and int(initial.numel()) != int(expected_channels):
            raise ValueError(
                f"Mask for {target.ffn_path} has {initial.numel()} channels, "
                f"expected {expected_channels}."
            )

        self.target = target
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.hard = bool(hard)
        self.ste = bool(ste)
        self.trainable = bool(trainable)

        parameter = nn.Parameter(initial.detach().clone().float(), requires_grad=trainable)
        self.logits = parameter
        if cost_per_channel is None:
            cost_per_channel = torch.ones_like(parameter.detach())
        self.register_buffer("cost_per_channel", cost_per_channel.detach().clone().float())
        if flop_cost_per_channel is None:
            flop_cost_per_channel = torch.ones_like(parameter.detach())
        self.register_buffer(
            "flop_cost_per_channel",
            flop_cost_per_channel.detach().clone().float(),
        )

    @classmethod
    def from_binary_values(
        cls,
        target: FFNTarget,
        values: Tensor,
        cost_per_channel: Tensor | None = None,
        flop_cost_per_channel: Tensor | None = None,
    ) -> "StructuredMask":
        """Create a fixed hard mask from binary values."""

        return cls(
            target=target,
            values=values.float(),
            hard=True,
            ste=False,
            trainable=False,
            cost_per_channel=cost_per_channel,
            flop_cost_per_channel=flop_cost_per_channel,
        )

    @classmethod
    def from_saliency_logits(
        cls,
        target: FFNTarget,
        saliency: Tensor,
        retained_ratio: float,
        temperature: float = 1.0,
        init_scale: float = 2.0,
        ste: bool = True,
        cost_per_channel: Tensor | None = None,
        flop_cost_per_channel: Tensor | None = None,
    ) -> "StructuredMask":
        """Create a trainable mask whose logits are initialized from saliency."""

        logits = saliency_to_logits(saliency, scale=init_scale)
        threshold = threshold_for_retained_ratio(
            torch.sigmoid(logits / temperature),
            retained_ratio,
        )
        return cls(
            target=target,
            logits=logits,
            temperature=temperature,
            threshold=threshold,
            hard=True,
            ste=ste,
            trainable=True,
            cost_per_channel=cost_per_channel,
            flop_cost_per_channel=flop_cost_per_channel,
        )

    @property
    def probabilities(self) -> Tensor:
        """Relaxed keep probabilities ``q = sigmoid(phi / tau)``."""

        return torch.sigmoid(self.logits / self.temperature)

    @property
    def values(self) -> Tensor:
        """Mask values used by the current forward mode."""

        if not self.hard:
            return self.probabilities

        threshold = self.logits.new_tensor(self.threshold)
        if self.ste and self.logits.requires_grad:
            return BinaryMaskSTE.apply(self.probabilities, threshold)
        return (self.probabilities >= threshold).to(dtype=self.logits.dtype)

    @property
    def binary_values(self) -> Tensor:
        """Detached hard binary mask values."""

        return (self.probabilities.detach() >= self.threshold).to(dtype=self.logits.dtype)

    @property
    def active_channels(self) -> int:
        return int(self.binary_values.count_nonzero().item())

    @property
    def total_channels(self) -> int:
        return int(self.logits.numel())

    @property
    def retained_ratio(self) -> float:
        if self.total_channels == 0:
            return 0.0
        return self.active_channels / self.total_channels

    @property
    def cost_summary(self) -> MaskCostSummary:
        binary = self.binary_values.to(device=self.cost_per_channel.device)
        retained = torch.sum(binary * self.cost_per_channel).detach().cpu().item()
        full = torch.sum(self.cost_per_channel).detach().cpu().item()
        flop_binary = self.binary_values.to(device=self.flop_cost_per_channel.device)
        retained_flops = torch.sum(flop_binary * self.flop_cost_per_channel).detach().cpu().item()
        full_flops = torch.sum(self.flop_cost_per_channel).detach().cpu().item()
        return MaskCostSummary(
            full_cost=float(full),
            retained_cost=float(retained),
            full_flop_cost=float(full_flops),
            retained_flop_cost=float(retained_flops),
        )

    def set_temperature(self, temperature: float) -> None:
        """Update the relaxation temperature."""

        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def harden_topk_(self, retained_ratio: float) -> None:
        """Convert the mask to an exact top-k hard mask in-place."""

        binary = topk_binary_mask(self.probabilities.detach(), retained_ratio)
        with torch.no_grad():
            self.logits.copy_(_values_to_logits(binary).to(device=self.logits.device))
        self.threshold = 0.5
        self.hard = True
        self.ste = False
        self.trainable = False
        self.logits.requires_grad_(False)


class MaskCollection(nn.Module):
    """A module container for structured masks keyed by FFN path."""

    def __init__(self, masks: Mapping[str, StructuredMask]) -> None:
        super().__init__()
        self._keys = tuple(masks.keys())
        for index, key in enumerate(self._keys):
            self.add_module(f"mask_{index}", masks[key])

    def as_dict(self) -> dict[str, StructuredMask]:
        """Return masks keyed by their target FFN path."""

        return {
            key: getattr(self, f"mask_{index}")
            for index, key in enumerate(self._keys)
        }

    @property
    def full_cost(self) -> float:
        return sum(mask.cost_summary.full_cost for mask in self.as_dict().values())

    @property
    def retained_cost(self) -> float:
        return sum(mask.cost_summary.retained_cost for mask in self.as_dict().values())

    @property
    def retained_cost_ratio(self) -> float:
        if self.full_cost <= 0.0:
            return 0.0
        return self.retained_cost / self.full_cost

    @property
    def full_flop_cost(self) -> float:
        return sum(mask.cost_summary.full_flop_cost for mask in self.as_dict().values())

    @property
    def retained_flop_cost(self) -> float:
        return sum(mask.cost_summary.retained_flop_cost for mask in self.as_dict().values())

    @property
    def retained_flop_cost_ratio(self) -> float:
        if self.full_flop_cost <= 0.0:
            return 0.0
        return self.retained_flop_cost / self.full_flop_cost


def masks_from_saliency(
    saliency_by_target: Mapping[str, Tensor],
    targets: Sequence[FFNTarget],
    retained_ratio: float,
    model: object | None = None,
) -> dict[str, StructuredMask]:
    """Create exact top-k binary masks from per-channel saliency scores."""

    _validate_retained_ratio(retained_ratio)

    masks: dict[str, StructuredMask] = {}
    for target in targets:
        scores = _scores_for_target(saliency_by_target, target)
        values = topk_binary_mask(scores, retained_ratio)
        parameter_cost = (
            infer_channel_costs(model, target, device=values.device)
            if model is not None
            else None
        )
        flop_cost = (
            infer_channel_flop_costs(model, target, device=values.device)
            if model is not None
            else None
        )
        masks[target.ffn_path] = StructuredMask.from_binary_values(
            target=target,
            values=values,
            cost_per_channel=parameter_cost,
            flop_cost_per_channel=flop_cost,
        )
    return masks


def trainable_masks_from_saliency(
    saliency_by_target: Mapping[str, Tensor],
    targets: Sequence[FFNTarget],
    retained_ratio: float,
    model: object | None = None,
    temperature: float = 1.0,
    init_scale: float = 2.0,
    ste: bool = True,
) -> MaskCollection:
    """Create trainable STE masks initialized from layer-local saliency."""

    _validate_retained_ratio(retained_ratio)

    masks: dict[str, StructuredMask] = {}
    for target in targets:
        scores = _scores_for_target(saliency_by_target, target)
        parameter_cost = (
            infer_channel_costs(model, target, device=scores.device)
            if model is not None
            else None
        )
        flop_cost = (
            infer_channel_flop_costs(model, target, device=scores.device)
            if model is not None
            else None
        )
        masks[target.ffn_path] = StructuredMask.from_saliency_logits(
            target=target,
            saliency=scores,
            retained_ratio=retained_ratio,
            temperature=temperature,
            init_scale=init_scale,
            ste=ste,
            cost_per_channel=parameter_cost,
            flop_cost_per_channel=flop_cost,
        )
    return MaskCollection(masks)


def topk_binary_mask(scores: Tensor, retained_ratio: float) -> Tensor:
    """Return an exact top-k binary mask with shape derived from ``scores``."""

    _validate_retained_ratio(retained_ratio)
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D per-channel scores, got shape {tuple(scores.shape)}.")
    channels = int(scores.numel())
    if channels <= 0:
        raise ValueError("scores must contain at least one channel.")
    keep = max(1, int(round(channels * retained_ratio)))
    topk = torch.topk(scores.detach().float(), k=keep, largest=True).indices
    values = torch.zeros_like(scores, dtype=torch.float32)
    values[topk] = 1.0
    return values


def saliency_to_logits(saliency: Tensor, scale: float = 2.0, eps: float = 1e-6) -> Tensor:
    """Layer-local saliency warm start from ``docs/THEORY.tex`` Stage 0."""

    if saliency.ndim != 1:
        raise ValueError(f"Expected 1D saliency, got shape {tuple(saliency.shape)}.")
    values = saliency.detach().float()
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(eps)
    return scale * (values - mean) / std


def threshold_for_retained_ratio(probabilities: Tensor, retained_ratio: float) -> float:
    """Choose a threshold that approximately keeps the requested probability top-k."""

    _validate_retained_ratio(retained_ratio)
    if probabilities.ndim != 1:
        raise ValueError(
            f"Expected 1D per-channel probabilities, got shape {tuple(probabilities.shape)}."
        )
    channels = int(probabilities.numel())
    keep = max(1, int(round(channels * retained_ratio)))
    cutoff = torch.topk(probabilities.detach().float(), k=keep, largest=True).values[-1]
    return float(cutoff.cpu().item())


@contextmanager
def apply_structured_masks(
    model: object,
    masks: Mapping[str, StructuredMask] | MaskCollection,
) -> Iterator[None]:
    """Temporarily apply FFN channel masks to expansion outputs."""

    mask_map = masks.as_dict() if isinstance(masks, MaskCollection) else masks
    handles = []

    def make_hook(structured_mask: StructuredMask, module_path: str):
        def hook(module: object, inputs: tuple[object, ...], output: Tensor) -> Tensor:
            mask = structured_mask.values
            broadcast = _broadcast_mask(mask.to(device=output.device, dtype=output.dtype), output)
            return output * broadcast

        return hook

    try:
        for structured_mask in mask_map.values():
            for module_path in structured_mask.target.expand_module_paths:
                module = get_module_by_path(model, module_path)
                handles.append(
                    module.register_forward_hook(
                        make_hook(structured_mask, module_path),
                    )
                )
        yield
    finally:
        for handle in handles:
            handle.remove()


def infer_channel_costs(
    model: object | None,
    target: FFNTarget,
    device: torch.device | str | None = None,
) -> Tensor:
    """Infer per-channel FFN parameter cost from module weight shapes."""

    channels = _target_channels(target)
    if model is None:
        return torch.ones(channels, device=device)

    per_channel_cost = 0.0
    for module_path in target.expand_module_paths:
        module = get_module_by_path(model, module_path)
        per_channel_cost += _output_channel_weight_cost(module, channels)
        per_channel_cost += _output_channel_bias_cost(module, channels)

    for module_path in target.contract_module_paths:
        module = get_module_by_path(model, module_path)
        per_channel_cost += _input_channel_weight_cost(module, channels)

    return torch.full((channels,), float(per_channel_cost), dtype=torch.float32, device=device)


def infer_channel_flop_costs(
    model: object | None,
    target: FFNTarget,
    device: torch.device | str | None = None,
) -> Tensor:
    """Infer per-token FFN FLOP proxy from module weight shapes.

    The returned value counts one multiply-add path per weight touched by a channel. Bias
    terms are excluded because they do not affect pruning ratios in the same way.
    """

    channels = _target_channels(target)
    if model is None:
        return torch.ones(channels, device=device)

    per_channel_cost = 0.0
    for module_path in target.expand_module_paths:
        module = get_module_by_path(model, module_path)
        per_channel_cost += _output_channel_weight_cost(module, channels)

    for module_path in target.contract_module_paths:
        module = get_module_by_path(model, module_path)
        per_channel_cost += _input_channel_weight_cost(module, channels)

    return torch.full((channels,), float(per_channel_cost), dtype=torch.float32, device=device)


def total_mask_cost(masks: Mapping[str, StructuredMask] | MaskCollection) -> MaskCostSummary:
    """Return aggregate full and retained cost across masks."""

    mask_map = masks.as_dict() if isinstance(masks, MaskCollection) else masks
    full = sum(mask.cost_summary.full_cost for mask in mask_map.values())
    retained = sum(mask.cost_summary.retained_cost for mask in mask_map.values())
    full_flops = sum(mask.cost_summary.full_flop_cost for mask in mask_map.values())
    retained_flops = sum(mask.cost_summary.retained_flop_cost for mask in mask_map.values())
    return MaskCostSummary(
        full_cost=float(full),
        retained_cost=float(retained),
        full_flop_cost=float(full_flops),
        retained_flop_cost=float(retained_flops),
    )


def annealed_temperature(
    step: int,
    initial_temperature: float,
    min_temperature: float,
    decay: float,
) -> float:
    """Exponential temperature schedule from ``docs/THEORY.tex`` Stage 2."""

    if step < 0:
        raise ValueError("step must be non-negative.")
    if initial_temperature <= 0.0 or min_temperature <= 0.0:
        raise ValueError("temperatures must be positive.")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1].")
    return max(float(min_temperature), float(initial_temperature) * float(decay) ** int(step))


def set_mask_temperature(
    masks: Mapping[str, StructuredMask] | MaskCollection,
    temperature: float,
) -> None:
    """Set a shared temperature on every structured mask."""

    mask_map = masks.as_dict() if isinstance(masks, MaskCollection) else masks
    for mask in mask_map.values():
        mask.set_temperature(temperature)


def save_mask_state(
    path: str | Path,
    masks: Mapping[str, StructuredMask] | MaskCollection,
) -> None:
    """Serialize structured masks and their target metadata."""

    mask_map = masks.as_dict() if isinstance(masks, MaskCollection) else masks
    state = {
        "version": MASK_STATE_VERSION,
        "masks": {
            key: {
                "target": target_to_dict(mask.target),
                "logits": mask.logits.detach().cpu(),
                "temperature": mask.temperature,
                "threshold": mask.threshold,
                "hard": mask.hard,
                "ste": mask.ste,
                "trainable": mask.trainable,
                "cost_per_channel": mask.cost_per_channel.detach().cpu(),
                "flop_cost_per_channel": mask.flop_cost_per_channel.detach().cpu(),
            }
            for key, mask in mask_map.items()
        },
    }
    torch.save(state, Path(path))


def load_mask_state(path: str | Path) -> dict[str, StructuredMask]:
    """Load masks saved by :func:`save_mask_state`."""

    state = torch.load(Path(path), map_location="cpu")
    if state.get("version") != MASK_STATE_VERSION:
        raise ValueError(f"Unsupported mask state version: {state.get('version')!r}")

    masks: dict[str, StructuredMask] = {}
    for key, item in state["masks"].items():
        target = target_from_dict(item["target"])
        masks[key] = StructuredMask(
            target=target,
            logits=item["logits"],
            temperature=float(item["temperature"]),
            threshold=float(item["threshold"]),
            hard=bool(item["hard"]),
            ste=bool(item["ste"]),
            trainable=bool(item["trainable"]),
            cost_per_channel=item["cost_per_channel"],
            flop_cost_per_channel=item.get("flop_cost_per_channel"),
        )
    return masks


def target_to_dict(target: FFNTarget) -> dict[str, Any]:
    """Serialize an FFN target."""

    return {
        "block_index": target.block_index,
        "block_path": target.block_path,
        "ffn_path": target.ffn_path,
        "topology": target.topology.value,
        "expand_module_paths": list(target.expand_module_paths),
        "contract_module_paths": list(target.contract_module_paths),
        "intermediate_size": target.intermediate_size,
        "hidden_size": target.hidden_size,
        "registry_name": target.registry_name,
    }


def target_from_dict(data: Mapping[str, Any]) -> FFNTarget:
    """Deserialize an FFN target."""

    intermediate_size = data.get("intermediate_size")
    hidden_size = data.get("hidden_size")
    return FFNTarget(
        block_index=int(data["block_index"]),
        block_path=str(data["block_path"]),
        ffn_path=str(data["ffn_path"]),
        topology=FFNTopologyKind(str(data["topology"])),
        expand_module_paths=tuple(data.get("expand_module_paths", ())),
        contract_module_paths=tuple(data.get("contract_module_paths", ())),
        intermediate_size=int(intermediate_size) if intermediate_size is not None else None,
        hidden_size=int(hidden_size) if hidden_size is not None else None,
        registry_name=data.get("registry_name"),
    )


def _scores_for_target(
    saliency_by_target: Mapping[str, Tensor],
    target: FFNTarget,
) -> Tensor:
    try:
        scores = saliency_by_target[target.ffn_path].detach()
    except KeyError as exc:
        raise KeyError(f"No saliency scores found for {target.ffn_path}.") from exc

    expected_channels = _target_channels(target)
    if int(scores.numel()) != expected_channels:
        raise ValueError(
            f"Saliency for {target.ffn_path} has {scores.numel()} channels, "
            f"expected {expected_channels}."
        )
    return scores.float()


def _broadcast_mask(mask: Tensor, output: Tensor) -> Tensor:
    if output.shape[-1] != mask.numel():
        raise ValueError(
            f"Mask length {mask.numel()} does not match module output width {output.shape[-1]}."
        )
    view_shape = [1] * output.ndim
    view_shape[-1] = mask.numel()
    return mask.view(*view_shape)


def _values_to_logits(values: Tensor | None, eps: float = 1e-4) -> Tensor | None:
    if values is None:
        return None
    clipped = values.detach().float().clamp(min=eps, max=1.0 - eps)
    return torch.logit(clipped)


def _target_channels(target: FFNTarget) -> int:
    if target.intermediate_size is None or target.intermediate_size <= 0:
        raise ValueError(f"Target {target.ffn_path} has invalid intermediate size.")
    return int(target.intermediate_size)


def _validate_retained_ratio(retained_ratio: float) -> None:
    if not 0.0 < retained_ratio <= 1.0:
        raise ValueError("retained_ratio must be in (0, 1].")


def _weight_shape(module: object) -> tuple[int, ...]:
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None:
        raise ValueError(f"Module {type(module).__name__} has no weight shape.")
    return tuple(int(dim) for dim in shape)


def _output_axis(module: object, channel_count: int) -> int:
    shape = _weight_shape(module)
    if hasattr(module, "out_features"):
        return 0
    if hasattr(module, "nf"):
        return len(shape) - 1
    matches = [index for index, dim in enumerate(shape) if dim == channel_count]
    if len(matches) == 1:
        return matches[0]
    return len(shape) - 1


def _input_axis(module: object, channel_count: int) -> int:
    shape = _weight_shape(module)
    output_axis = _output_axis(module, channel_count)
    matches = [index for index, dim in enumerate(shape) if dim == channel_count]
    for index in matches:
        if index != output_axis:
            return index
    if matches:
        return matches[0]
    if len(shape) == 2:
        return 1 if output_axis == 0 else 0
    raise ValueError(
        f"Cannot infer input channel axis for {type(module).__name__} with shape {shape}."
    )


def _axis_slice_cost(module: object, axis: int) -> int:
    shape = _weight_shape(module)
    cost = 1
    for index, dim in enumerate(shape):
        if index != axis:
            cost *= dim
    return int(cost)


def _output_channel_weight_cost(module: object, channel_count: int) -> int:
    return _axis_slice_cost(module, _output_axis(module, channel_count))


def _input_channel_weight_cost(module: object, channel_count: int) -> int:
    return _axis_slice_cost(module, _input_axis(module, channel_count))


def _output_channel_bias_cost(module: object, channel_count: int) -> int:
    bias = getattr(module, "bias", None)
    shape = getattr(bias, "shape", None)
    if shape is None or len(shape) != 1:
        return 0
    return 1 if int(shape[0]) == int(channel_count) else 0
