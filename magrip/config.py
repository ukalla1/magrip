"""Configuration objects for MaGRIP."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObjectiveConfig:
    """Weights for the MaGRIP budget-aware objective."""

    target_retained_ratio: float = 0.7
    initial_retained_ratio: float = 1.0
    budget_penalty_weight: float = 1.0
    initial_budget_penalty_weight: float = 0.0
    budget_warmup_steps: int = 0
    penalty_warmup_steps: int = 0
    mask_regularization_weight: float = 0.0
    distillation_weight: float = 0.0
    distillation_temperature: float = 1.0
    distillation_mode: str = "disabled"


@dataclass
class MaskScheduleConfig:
    """Schedules controlling soft-mask behavior."""

    initial_temperature: float = 1.0
    min_temperature: float = 0.05
    temperature_decay: float = 0.99
    mask_update_frequency: int = 1
    max_mask_update: float | None = None
    init_scale: float = 2.0
    use_ste: bool = True


@dataclass
class TrainingConfig:
    """Controls for the M5 joint optimization loop."""

    max_steps: int = 20
    log_every: int = 1
    show_progress: bool = False
    eval_every: int = 0
    checkpoint_every: int = 0
    train_weights: bool = False
    train_masks: bool = True
    recompute_saliency_every: int = 0
    stabilization_steps: int = 0
    final_harden: bool = True
    final_recovery_steps: int = 0
    clip_mask_grad_norm: float | None = 1.0
    clip_weight_grad_norm: float | None = None


@dataclass
class OptimizerConfig:
    """Optimizer choices for weights and masks."""

    weight_optimizer: str = "adamw"
    mask_optimizer: str = "adamw"
    weight_learning_rate: float = 1e-5
    mask_learning_rate: float = 1e-3
    weight_decay: float = 0.0
    mask_weight_decay: float = 0.0
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
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seed: int = 42
