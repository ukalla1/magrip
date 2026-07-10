"""Optimizer backend selection for MaGRIP."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn
from torch.optim import AdamW, Optimizer, SGD

from magrip.config import OptimizerConfig


def apollo_available() -> bool:
    """Return whether APOLLO integration is available.

    APOLLO support is planned for M6 and is not wired yet.
    """

    return False


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
    parameters: Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> Optimizer:
    """Build the model-weight optimizer."""

    if config.use_apollo:
        raise RuntimeError("APOLLO is reserved for M6 and is not wired in M5.")
    return _build_optimizer(
        name=config.weight_optimizer,
        parameters=parameters,
        learning_rate=config.weight_learning_rate,
        weight_decay=config.weight_decay,
        purpose="weight",
    )


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return trainable parameters from a module."""

    return [parameter for parameter in module.parameters() if parameter.requires_grad]


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
