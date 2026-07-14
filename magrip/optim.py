"""Optimizer backend selection for MaGRIP."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import inspect
from typing import Any

from torch import nn
from torch.optim import AdamW, Optimizer, SGD

from magrip.config import OptimizerConfig


def apollo_available() -> bool:
    """Return whether the external APOLLO optimizer package is importable."""

    try:
        _load_apollo_adamw()
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class ApolloParameterStats:
    """Summary of APOLLO parameter grouping."""

    total_parameters: int
    lowrank_parameters: int
    regular_parameters: int
    lowrank_tensors: int
    regular_tensors: int
    lowrank_auxiliary_elements: int
    lowrank_projection_elements: int
    regular_optimizer_state_elements: int
    estimated_optimizer_state_elements: int
    estimated_optimizer_state_bytes_fp32: int
    adamw_optimizer_state_elements: int

    @property
    def adapted_parameters(self) -> int:
        return self.lowrank_parameters + self.regular_parameters

    @property
    def estimated_state_ratio_vs_adamw(self) -> float:
        if self.adamw_optimizer_state_elements <= 0:
            return 0.0
        return self.estimated_optimizer_state_elements / self.adamw_optimizer_state_elements


def build_mask_optimizer(parameters: Iterable[nn.Parameter], config: OptimizerConfig) -> Optimizer:
    """Build the mask optimizer."""

    return _build_optimizer(
        name=config.mask_optimizer,
        parameters=parameters,
        learning_rate=config.mask_learning_rate,
        weight_decay=config.mask_weight_decay,
        purpose="mask",
    )


def build_weight_optimizer(
    parameters: nn.Module | Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> Optimizer:
    """Build the model-weight optimizer."""

    if _uses_apollo(config):
        named_parameters = _named_parameters(parameters)
        groups, _stats = build_apollo_parameter_groups(named_parameters, config)
        apollo_cls = _load_apollo_adamw()
        return apollo_cls(
            groups,
            **_filtered_kwargs(
                apollo_cls,
                {
                    "lr": config.weight_learning_rate,
                    "weight_decay": config.weight_decay,
                },
            ),
        )
    return _build_optimizer(
        name=config.weight_optimizer,
        parameters=_parameters(parameters),
        learning_rate=config.weight_learning_rate,
        weight_decay=config.weight_decay,
        purpose="weight",
    )


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return trainable parameters from a module."""

    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def build_apollo_parameter_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    config: OptimizerConfig,
) -> tuple[list[dict[str, Any]], ApolloParameterStats]:
    """Create APOLLO low-rank and regular parameter groups."""

    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("No trainable weight parameters were provided.")

    lowrank_params = [
        parameter
        for _name, parameter in trainable
        if parameter.ndim >= 2
    ]
    regular_params = [
        parameter
        for _name, parameter in trainable
        if parameter.ndim < 2
    ]

    groups: list[dict[str, Any]] = []
    if regular_params:
        groups.append({"params": regular_params})
    if lowrank_params:
        groups.append(
            {
                "params": lowrank_params,
                "rank": _apollo_rank(config),
                "proj": config.apollo_proj,
                "scale_type": _apollo_scale_type(config),
                "scale": _apollo_scale(config),
                "update_proj_gap": config.apollo_update_proj_gap,
                "proj_type": config.apollo_proj_type,
            }
        )

    stats = ApolloParameterStats(
        total_parameters=sum(parameter.numel() for _name, parameter in trainable),
        lowrank_parameters=sum(parameter.numel() for parameter in lowrank_params),
        regular_parameters=sum(parameter.numel() for parameter in regular_params),
        lowrank_tensors=len(lowrank_params),
        regular_tensors=len(regular_params),
        lowrank_auxiliary_elements=_lowrank_auxiliary_elements(lowrank_params, config),
        lowrank_projection_elements=_lowrank_projection_elements(lowrank_params, config),
        regular_optimizer_state_elements=2 * sum(parameter.numel() for parameter in regular_params),
        estimated_optimizer_state_elements=(
            _lowrank_auxiliary_elements(lowrank_params, config)
            + _lowrank_projection_elements(lowrank_params, config)
            + 2 * sum(parameter.numel() for parameter in regular_params)
        ),
        estimated_optimizer_state_bytes_fp32=4
        * (
            _lowrank_auxiliary_elements(lowrank_params, config)
            + _lowrank_projection_elements(lowrank_params, config)
            + 2 * sum(parameter.numel() for parameter in regular_params)
        ),
        adamw_optimizer_state_elements=2 * sum(parameter.numel() for _name, parameter in trainable),
    )
    return groups, stats


def apollo_parameter_stats(
    parameters: nn.Module | Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> ApolloParameterStats | None:
    """Return APOLLO grouping stats if APOLLO is enabled."""

    if not _uses_apollo(config):
        return None
    _groups, stats = build_apollo_parameter_groups(_named_parameters(parameters), config)
    return stats


def optimizer_state_diagnostics(optimizer: Optimizer | None) -> dict[str, float | int]:
    """Best-effort tensor norm diagnostics from an optimizer state dict."""

    if optimizer is None:
        return {}

    total_norm_sq = 0.0
    projected_norm_sq = 0.0
    update_norm_sq = 0.0
    tensor_count = 0
    tensor_elements = 0
    projected_tensor_count = 0
    update_tensor_count = 0
    key_norms: dict[str, float] = {}

    for state in optimizer.state.values():
        for key, value in _iter_state_tensors(state):
            norm = float(value.detach().float().norm().cpu().item())
            elements = int(value.numel())
            tensor_count += 1
            tensor_elements += elements
            total_norm_sq += norm * norm
            key_norms[key] = key_norms.get(key, 0.0) + norm * norm
            normalized_key = key.lower()
            if any(
                token in normalized_key
                for token in ("proj", "rank", "lowrank", "exp_avg", "exp_avg_sq")
            ):
                projected_tensor_count += 1
                projected_norm_sq += norm * norm
            if any(token in normalized_key for token in ("update", "scaled_grad")):
                update_tensor_count += 1
                update_norm_sq += norm * norm

    diagnostics: dict[str, float | int] = {
        "optimizer_state_tensor_count": tensor_count,
        "optimizer_state_tensor_elements": tensor_elements,
        "optimizer_state_tensor_norm": total_norm_sq**0.5,
        "projected_state_tensor_count": projected_tensor_count,
        "projected_state_tensor_norm": projected_norm_sq**0.5,
        "update_state_tensor_count": update_tensor_count,
        "update_state_tensor_norm": update_norm_sq**0.5,
    }
    for key, norm_sq in sorted(key_norms.items()):
        diagnostics[f"state_norm_{_sanitize_key(key)}"] = norm_sq**0.5
    return diagnostics


def _build_optimizer(
    name: str,
    parameters: Iterable[nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    purpose: str,
) -> Optimizer:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError(f"No trainable {purpose} parameters were provided.")
    normalized = name.lower()
    if normalized == "adamw":
        return AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "sgd":
        return SGD(params, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported {purpose} optimizer: {name!r}.")


def _uses_apollo(config: OptimizerConfig) -> bool:
    return config.use_apollo or config.weight_optimizer.lower() in {
        "apollo",
        "apollo-mini",
        "apollo_mini",
    }


def _load_apollo_adamw() -> type[Optimizer]:
    try:
        from apollo_torch import APOLLOAdamW
    except ImportError as exc:
        raise ImportError(
            "APOLLO is required for M6 weight adaptation. Install it with "
            "`pip install apollo-torch` or install the APOLLO repository with `pip install -e .`."
        ) from exc
    return APOLLOAdamW


def _named_parameters(
    parameters: nn.Module | Iterable[nn.Parameter],
) -> list[tuple[str, nn.Parameter]]:
    if isinstance(parameters, nn.Module):
        return list(parameters.named_parameters())
    return [(f"parameter_{index}", parameter) for index, parameter in enumerate(parameters)]


def _parameters(parameters: nn.Module | Iterable[nn.Parameter]) -> Iterable[nn.Parameter]:
    if isinstance(parameters, nn.Module):
        return parameters.parameters()
    return parameters


def _filtered_kwargs(callable_obj: object, kwargs: dict[str, object]) -> dict[str, object]:
    signature = inspect.signature(callable_obj)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def _apollo_variant(config: OptimizerConfig) -> str:
    if config.weight_optimizer.lower() in {"apollo-mini", "apollo_mini"}:
        return "apollo-mini"
    return config.apollo_variant.lower().replace("_", "-")


def _apollo_rank(config: OptimizerConfig) -> int:
    if _apollo_variant(config) == "apollo-mini":
        return 1
    return config.apollo_rank


def _apollo_scale(config: OptimizerConfig) -> float:
    if _apollo_variant(config) == "apollo-mini":
        return 128.0
    return config.apollo_scale


def _apollo_scale_type(config: OptimizerConfig) -> str:
    if _apollo_variant(config) == "apollo-mini":
        return "tensor"
    return config.apollo_scale_type


def _lowrank_auxiliary_elements(
    parameters: Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> int:
    rank = _apollo_rank(config)
    total = 0
    for parameter in parameters:
        rows, columns = _matrix_view_shape(parameter)
        total += 2 * rank * columns
    return int(total)


def _lowrank_projection_elements(
    parameters: Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> int:
    rank = _apollo_rank(config)
    return int(sum(rank * _matrix_view_shape(parameter)[0] for parameter in parameters))


def _matrix_view_shape(parameter: nn.Parameter) -> tuple[int, int]:
    rows = int(parameter.shape[0])
    columns = int(parameter.numel() // max(1, rows))
    return rows, columns


def _iter_state_tensors(value: object, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if hasattr(value, "detach") and hasattr(value, "numel"):
        yield prefix or "tensor", value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_state_tensors(item, next_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_state_tensors(item, next_prefix)


def _sanitize_key(key: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in key.lower())
