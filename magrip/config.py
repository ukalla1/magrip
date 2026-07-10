"""Configuration objects for MaGRIP."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObjectiveConfig:
    """Weights for the MaGRIP budget-aware objective."""

    target_retained_ratio: float = 0.7
    budget_penalty_weight: float = 1.0
    mask_regularization_weight: float = 0.0
    distillation_weight: float = 0.0


@dataclass
class MaskScheduleConfig:
    """Schedules controlling soft-mask behavior."""

    initial_temperature: float = 1.0
    min_temperature: float = 0.05
    temperature_decay: float = 0.99
    mask_update_frequency: int = 1
    max_mask_update: float | None = None


@dataclass
class OptimizerConfig:
    """Optimizer choices for weights and masks."""

    weight_optimizer: str = "adamw"
    mask_optimizer: str = "adamw"
    weight_learning_rate: float = 1e-5
    mask_learning_rate: float = 1e-3
    use_apollo: bool = False


@dataclass
class MaGRIPConfig:
    """Top-level MaGRIP configuration.

    Distillation is intentionally disabled by default. The theory document keeps it as an
    optional stabilizer, but the implementation should assume beta = 0 unless explicitly
    configured otherwise.
    """

    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    mask_schedule: MaskScheduleConfig = field(default_factory=MaskScheduleConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seed: int = 42
