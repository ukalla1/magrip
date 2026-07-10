"""Budget-aware MaGRIP objective terms."""

from __future__ import annotations

from magrip.config import ObjectiveConfig


def distillation_is_enabled(config: ObjectiveConfig) -> bool:
    """Return whether optional distillation should be used."""

    return config.distillation_weight > 0.0

