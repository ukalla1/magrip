"""Training loop orchestration for MaGRIP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from magrip.config import MaGRIPConfig
from magrip.discovery import discover_ffn_targets
from magrip.evaluation import causal_lm_loss, perplexity_from_loss
from magrip.masks import (
    MaskCollection,
    apply_structured_masks,
    annealed_temperature,
    save_mask_state,
    set_mask_temperature,
    total_mask_cost,
    trainable_masks_from_saliency,
)
from magrip.objectives import ObjectiveBreakdown, compute_magrip_objective
from magrip.optim import (
    apollo_parameter_stats,
    build_mask_optimizer,
    build_weight_optimizer,
    optimizer_state_diagnostics,
)
from magrip.saliency import (
    SaliencyConfig,
    SaliencyRefreshSchedule,
    SaliencyResult,
    SaliencyTracker,
    collect_saliency,
)
from magrip.topology import FFNTarget


@dataclass
class TrainingStepMetrics:
    """Scalar metrics from one M5 optimization step."""

    step: int
    epoch_index: int
    batch_index: int
    temperature: float
    mask_update: bool
    weight_update: bool
    objective: dict[str, float]
    hard_retained_cost_ratio: float
    hard_retained_flop_ratio: float
    mask_grad_norm: float | None = None
    mask_grad_mean: float | None = None
    mask_grad_min: float | None = None
    mask_grad_max: float | None = None
    mask_grad_nonzero_count: int | None = None
    mask_grad_target_count: int | None = None
    mask_update_mean_abs: float | None = None
    mask_update_max_abs: float | None = None
    weight_grad_norm: float | None = None
    apollo_diagnostics: dict[str, float | int] | None = None
    active_to_inactive_count: int | None = None
    inactive_to_active_count: int | None = None
    mask_flip_count: int | None = None
    validation_loss: float | None = None
    validation_perplexity: float | None = None
    saliency_drift: dict[str, float] | None = None


@dataclass
class TrainingResult:
    """Output from MaGRIP M5 training."""

    targets: list[FFNTarget]
    masks: MaskCollection
    initial_saliency: SaliencyResult
    metrics: list[TrainingStepMetrics]
    baseline_loss: float
    initial_masked_loss: float
    final_masked_loss: float
    num_steps: int
    num_tokens: int
    checkpoints: list[str] = field(default_factory=list)
    weight_trainable_parameter_count: int = 0
    apollo_parameter_stats: dict[str, float | int] | None = None
    initial_to_final_mask_flips: dict[str, int] | None = None

    @property
    def baseline_perplexity(self) -> float:
        return perplexity_from_loss(self.baseline_loss)

    @property
    def initial_masked_perplexity(self) -> float:
        return perplexity_from_loss(self.initial_masked_loss)

    @property
    def final_masked_perplexity(self) -> float:
        return perplexity_from_loss(self.final_masked_loss)


class MaGRIPTrainer:
    """Joint soft-mask adaptation trainer.

    M5 defaults to mask-only adaptation. M6 enables joint weight and mask adaptation by
    setting ``config.training.train_weights`` and selecting APOLLO in the optimizer config.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MaGRIPConfig | None = None,
        targets: list[FFNTarget] | None = None,
        saliency_config: SaliencyConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or MaGRIPConfig()
        self.targets = targets if targets is not None else list(discover_ffn_targets(model))
        if not self.targets:
            raise RuntimeError("No prunable FFN targets were discovered.")
        self.saliency_config = saliency_config or SaliencyConfig(
            collect_branch_diagnostics=False,
            require_parameter_gradients=False,
        )
        self.masks: MaskCollection | None = None
        self.initial_saliency: SaliencyResult | None = None
        self.mask_optimizer: Optimizer | None = None
        self.weight_optimizer: Optimizer | None = None
        self._original_requires_grad: list[bool] | None = None
        self._initial_binary_masks: dict[str, Tensor] | None = None

    def initialize_masks(self, batches: Sequence[Tensor]) -> MaskCollection:
        """Run Stage 0 saliency warm start and create trainable masks."""

        saliency = collect_saliency_over_batches(
            model=self.model,
            targets=self.targets,
            batches=batches,
            config=self.saliency_config,
        )
        self.initial_saliency = saliency
        self.masks = trainable_masks_from_saliency(
            saliency.combined_from_config(self.saliency_config),
            targets=self.targets,
            retained_ratio=self.config.objective.target_retained_ratio,
            model=self.model,
            temperature=self.config.mask_schedule.initial_temperature,
            init_scale=self.config.mask_schedule.init_scale,
            ste=self.config.mask_schedule.use_ste,
        )
        self._initial_binary_masks = self._capture_binary_masks()
        return self.masks

    def train(
        self,
        batches: Sequence[Tensor],
        eval_batches: Sequence[Tensor] | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> TrainingResult:
        """Run MaGRIP mask/weight adaptation."""

        if not batches:
            raise ValueError("batches must not be empty.")
        if eval_batches is None:
            eval_batches = batches
        if self.masks is None:
            self.initialize_masks(batches)
        if self.initial_saliency is None or self.masks is None:
            raise RuntimeError("Masks were not initialized.")

        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_path is not None:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        baseline_loss = mean_causal_lm_loss(self.model, eval_batches)
        initial_masked_loss = mean_causal_lm_loss(
            self.model,
            eval_batches,
            masks=self.masks,
        )

        self._prepare_trainable_parameters()
        self.mask_optimizer = (
            build_mask_optimizer(self.masks.parameters(), self.config.optimizer)
            if self.config.training.train_masks
            else None
        )
        weight_stats = self._weight_trainable_stats()
        self.weight_optimizer = (
            build_weight_optimizer(self.model, self.config.optimizer)
            if self.config.training.train_weights
            else None
        )

        saliency_tracker = SaliencyTracker(
            SaliencyRefreshSchedule(
                every_steps=self.config.training.recompute_saliency_every,
                start_step=1,
            )
        )
        saliency_tracker.update(self.initial_saliency, step=0)

        metrics: list[TrainingStepMetrics] = []
        checkpoints: list[str] = []
        step = 0
        progress = self._make_progress_bar()
        try:
            while step < self.config.training.max_steps:
                for batch_index, batch in enumerate(batches):
                    if step >= self.config.training.max_steps:
                        break
                    metric = self._train_step(
                        batch=batch,
                        step=step,
                        epoch_index=step // len(batches),
                        batch_index=batch_index,
                        saliency_tracker=saliency_tracker,
                    )
                    next_step = step + 1
                    if self._should_evaluate(next_step):
                        metric.validation_loss = mean_causal_lm_loss(
                            self.model,
                            eval_batches,
                            masks=self.masks,
                        )
                        metric.validation_perplexity = perplexity_from_loss(
                            metric.validation_loss,
                        )
                    metrics.append(metric)
                    self._update_progress_bar(progress, metric)
                    step = next_step
                    if checkpoint_path is not None and self._should_checkpoint(step):
                        checkpoints.append(str(self._save_checkpoint(checkpoint_path, step)))

            if self.config.training.stabilization_steps > 0:
                self._run_stabilization(batches)

            if self.config.training.final_harden:
                self.harden_masks()

            if self.config.training.final_recovery_steps > 0:
                self._run_final_recovery(batches)

            final_masked_loss = mean_causal_lm_loss(
                self.model,
                eval_batches,
                masks=self.masks,
            )

            if checkpoint_path is not None:
                checkpoints.append(str(self._save_checkpoint(checkpoint_path, step, final=True)))

            return TrainingResult(
                targets=self.targets,
                masks=self.masks,
                initial_saliency=self.initial_saliency,
                metrics=metrics,
                baseline_loss=baseline_loss,
                initial_masked_loss=initial_masked_loss,
                final_masked_loss=final_masked_loss,
                num_steps=step,
                num_tokens=int(sum(batch.numel() for batch in batches)),
                checkpoints=checkpoints,
                weight_trainable_parameter_count=weight_stats["trainable_parameter_count"],
                apollo_parameter_stats=weight_stats["apollo_parameter_stats"],
                initial_to_final_mask_flips=self._initial_to_current_mask_flips(),
            )
        finally:
            self._restore_trainable_parameters()
            if progress is not None:
                progress.close()

    def harden_masks(self) -> None:
        """Harden masks to the configured final retained budget."""

        if self.masks is None:
            raise RuntimeError("Masks were not initialized.")
        for mask in self.masks.as_dict().values():
            mask.harden_topk_(self.config.objective.target_retained_ratio)

    def _train_step(
        self,
        batch: Tensor,
        step: int,
        epoch_index: int,
        batch_index: int,
        saliency_tracker: SaliencyTracker,
    ) -> TrainingStepMetrics:
        if self.masks is None:
            raise RuntimeError("Masks were not initialized.")

        temperature = annealed_temperature(
            step=step,
            initial_temperature=self.config.mask_schedule.initial_temperature,
            min_temperature=self.config.mask_schedule.min_temperature,
            decay=self.config.mask_schedule.temperature_decay,
        )
        set_mask_temperature(self.masks, temperature)
        self._set_mask_forward_mode(step)

        mask_update = self._should_update_mask(step)
        weight_update = self.config.training.train_weights
        previous_logits = self._capture_mask_logits() if mask_update else None
        previous_binary = self._capture_binary_masks() if mask_update else None

        self.model.train(weight_update)
        if self.mask_optimizer is not None:
            self.mask_optimizer.zero_grad(set_to_none=True)
        if self.weight_optimizer is not None:
            self.weight_optimizer.zero_grad(set_to_none=True)

        objective = self._forward_objective(batch=batch, step=step)
        objective.total_loss.backward()

        mask_grad_stats = self._mask_gradient_stats() if mask_update else {}
        mask_grad_norm = self._clip_mask_gradients() if mask_update else None
        weight_grad_norm = self._clip_weight_gradients() if weight_update else None

        if self.weight_optimizer is not None and weight_update:
            self.weight_optimizer.step()
        apollo_diagnostics = (
            optimizer_state_diagnostics(self.weight_optimizer)
            if weight_update and self.config.optimizer.use_apollo
            else None
        )
        if self.mask_optimizer is not None and mask_update:
            self.mask_optimizer.step()
            if previous_logits is not None:
                self._clip_mask_update_(previous_logits)
        mask_update_stats = (
            self._mask_update_stats(previous_logits)
            if mask_update and previous_logits is not None
            else {}
        )
        mask_flip_stats = (
            self._mask_flip_stats(previous_binary)
            if mask_update and previous_binary is not None
            else {}
        )

        if self.mask_optimizer is not None:
            self.mask_optimizer.zero_grad(set_to_none=True)
        if self.weight_optimizer is not None:
            self.weight_optimizer.zero_grad(set_to_none=True)

        drift_payload = None
        if saliency_tracker.should_recompute(step + 1):
            saliency = collect_saliency_over_batches(
                model=self.model,
                targets=self.targets,
                batches=[batch],
                config=self.saliency_config,
            )
            drift = saliency_tracker.update(saliency, step=step + 1)
            if drift is not None:
                drift_payload = {
                    "mean_relative_l2": drift.mean_relative_l2,
                    "max_relative_l2": drift.max_relative_l2,
                    "mean_cosine_distance": drift.mean_cosine_distance,
                }

        hard_cost = total_mask_cost(self.masks)
        return TrainingStepMetrics(
            step=step,
            epoch_index=epoch_index,
            batch_index=batch_index,
            temperature=temperature,
            mask_update=mask_update,
            weight_update=weight_update,
            objective=objective.detached(),
            hard_retained_cost_ratio=hard_cost.retained_ratio,
            hard_retained_flop_ratio=hard_cost.flop_retained_ratio,
            mask_grad_norm=mask_grad_norm,
            mask_grad_mean=mask_grad_stats.get("mean"),
            mask_grad_min=mask_grad_stats.get("min"),
            mask_grad_max=mask_grad_stats.get("max"),
            mask_grad_nonzero_count=mask_grad_stats.get("nonzero_count"),
            mask_grad_target_count=mask_grad_stats.get("target_count"),
            mask_update_mean_abs=mask_update_stats.get("mean_abs"),
            mask_update_max_abs=mask_update_stats.get("max_abs"),
            weight_grad_norm=weight_grad_norm,
            apollo_diagnostics=apollo_diagnostics,
            active_to_inactive_count=mask_flip_stats.get("active_to_inactive"),
            inactive_to_active_count=mask_flip_stats.get("inactive_to_active"),
            mask_flip_count=mask_flip_stats.get("total"),
            saliency_drift=drift_payload,
        )

    def _forward_objective(self, batch: Tensor, step: int) -> ObjectiveBreakdown:
        if self.masks is None:
            raise RuntimeError("Masks were not initialized.")
        with apply_structured_masks(self.model, self.masks):
            outputs = self.model(input_ids=batch, labels=batch)
        return compute_magrip_objective(
            task_loss=outputs.loss,
            masks=self.masks,
            config=self.config.objective,
            step=step,
            student_logits=getattr(outputs, "logits", None),
            teacher_logits=None,
        )

    def _prepare_trainable_parameters(self) -> None:
        self._original_requires_grad = [
            parameter.requires_grad
            for parameter in self.model.parameters()
        ]
        if not self.config.training.train_weights:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            return

        scope = self.config.optimizer.apollo_parameter_scope.lower()
        if scope == "all":
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            return
        if scope != "ffn":
            raise ValueError(f"Unsupported adaptation parameter scope: {scope!r}.")

        ffn_prefixes = self._ffn_parameter_prefixes()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(_matches_any_prefix(name, ffn_prefixes))

    def _restore_trainable_parameters(self) -> None:
        if self._original_requires_grad is None:
            return
        for parameter, requires_grad in zip(self.model.parameters(), self._original_requires_grad):
            parameter.requires_grad_(requires_grad)
        self._original_requires_grad = None

    def _should_update_mask(self, step: int) -> bool:
        if not self.config.training.train_masks:
            return False
        frequency = max(1, self.config.mask_schedule.mask_update_frequency)
        return step % frequency == 0

    def _should_evaluate(self, step: int) -> bool:
        every = self.config.training.eval_every
        return every > 0 and step > 0 and step % every == 0

    def _set_mask_forward_mode(self, step: int) -> None:
        if self.masks is None:
            return
        soft_warmup_steps = max(0, self.config.mask_schedule.soft_warmup_steps)
        use_soft_masks = step < soft_warmup_steps
        for mask in self.masks.as_dict().values():
            mask.hard = not use_soft_masks
            mask.ste = False if use_soft_masks else self.config.mask_schedule.use_ste

    def _should_checkpoint(self, step: int) -> bool:
        every = self.config.training.checkpoint_every
        return every > 0 and step > 0 and step % every == 0

    def _capture_mask_logits(self) -> dict[str, Tensor]:
        if self.masks is None:
            raise RuntimeError("Masks were not initialized.")
        return {
            key: mask.logits.detach().clone()
            for key, mask in self.masks.as_dict().items()
        }

    def _capture_binary_masks(self) -> dict[str, Tensor]:
        if self.masks is None:
            raise RuntimeError("Masks were not initialized.")
        return {
            key: mask.binary_values.detach().clone().bool()
            for key, mask in self.masks.as_dict().items()
        }

    def _clip_mask_update_(self, previous_logits: dict[str, Tensor]) -> None:
        max_update = self.config.mask_schedule.max_mask_update
        if max_update is None or self.masks is None:
            return
        with torch.no_grad():
            for key, mask in self.masks.as_dict().items():
                previous = previous_logits[key].to(device=mask.logits.device)
                delta = torch.clamp(mask.logits - previous, min=-max_update, max=max_update)
                mask.logits.copy_(previous + delta)

    def _clip_mask_gradients(self) -> float | None:
        if self.masks is None or self.config.training.clip_mask_grad_norm is None:
            return None
        norm = clip_grad_norm_(self.masks.parameters(), self.config.training.clip_mask_grad_norm)
        return float(norm.detach().cpu().item())

    def _mask_gradient_stats(self) -> dict[str, float | int]:
        if self.masks is None:
            return {}
        norms: list[float] = []
        for mask in self.masks.as_dict().values():
            if mask.logits.grad is None:
                norms.append(0.0)
                continue
            grad_norm = torch.linalg.vector_norm(mask.logits.grad.detach().float())
            norms.append(float(grad_norm.cpu().item()))
        if not norms:
            return {}
        nonzero = sum(value > 0.0 for value in norms)
        return {
            "mean": float(sum(norms) / len(norms)),
            "min": float(min(norms)),
            "max": float(max(norms)),
            "nonzero_count": int(nonzero),
            "target_count": int(len(norms)),
        }

    def _mask_update_stats(self, previous_logits: dict[str, Tensor]) -> dict[str, float]:
        if self.masks is None:
            return {}
        mean_abs_values: list[float] = []
        max_abs_values: list[float] = []
        with torch.no_grad():
            for key, mask in self.masks.as_dict().items():
                previous = previous_logits[key].to(device=mask.logits.device)
                delta = (mask.logits.detach() - previous).float().abs()
                mean_abs_values.append(float(delta.mean().cpu().item()))
                max_abs_values.append(float(delta.max().cpu().item()))
        if not mean_abs_values:
            return {}
        return {
            "mean_abs": float(sum(mean_abs_values) / len(mean_abs_values)),
            "max_abs": float(max(max_abs_values)),
        }

    def _mask_flip_stats(self, previous_binary: dict[str, Tensor]) -> dict[str, int]:
        if self.masks is None:
            return {}
        active_to_inactive = 0
        inactive_to_active = 0
        for key, mask in self.masks.as_dict().items():
            previous = previous_binary[key].to(device=mask.logits.device)
            current = mask.binary_values.detach().bool()
            active_to_inactive += int((previous & ~current).sum().cpu().item())
            inactive_to_active += int((~previous & current).sum().cpu().item())
        return {
            "active_to_inactive": active_to_inactive,
            "inactive_to_active": inactive_to_active,
            "total": active_to_inactive + inactive_to_active,
        }

    def _clip_weight_gradients(self) -> float | None:
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            return None
        if self.config.training.clip_weight_grad_norm is None:
            return _parameter_gradient_norm(parameters)
        norm = clip_grad_norm_(parameters, self.config.training.clip_weight_grad_norm)
        return float(norm.detach().cpu().item())

    def _weight_trainable_stats(self) -> dict[str, int | dict[str, float | int] | None]:
        trainable_count = int(
            sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        )
        apollo_stats = apollo_parameter_stats(self.model, self.config.optimizer)
        return {
            "trainable_parameter_count": trainable_count,
            "apollo_parameter_stats": (
                None
                if apollo_stats is None
                else {
                    "total_parameters": apollo_stats.total_parameters,
                    "adapted_parameters": apollo_stats.adapted_parameters,
                    "lowrank_parameters": apollo_stats.lowrank_parameters,
                    "regular_parameters": apollo_stats.regular_parameters,
                    "lowrank_tensors": apollo_stats.lowrank_tensors,
                    "regular_tensors": apollo_stats.regular_tensors,
                    "lowrank_auxiliary_elements": apollo_stats.lowrank_auxiliary_elements,
                    "lowrank_projection_elements": apollo_stats.lowrank_projection_elements,
                    "regular_optimizer_state_elements": (
                        apollo_stats.regular_optimizer_state_elements
                    ),
                    "estimated_optimizer_state_elements": (
                        apollo_stats.estimated_optimizer_state_elements
                    ),
                    "estimated_optimizer_state_bytes_fp32": (
                        apollo_stats.estimated_optimizer_state_bytes_fp32
                    ),
                    "estimated_optimizer_state_mib_fp32": (
                        apollo_stats.estimated_optimizer_state_bytes_fp32 / (1024**2)
                    ),
                    "adamw_optimizer_state_elements": apollo_stats.adamw_optimizer_state_elements,
                    "estimated_state_ratio_vs_adamw": (
                        apollo_stats.estimated_state_ratio_vs_adamw
                    ),
                }
            ),
        }

    def _ffn_parameter_prefixes(self) -> tuple[str, ...]:
        prefixes: set[str] = set()
        for target in self.targets:
            prefixes.add(target.ffn_path)
            prefixes.update(target.expand_module_paths)
            prefixes.update(target.contract_module_paths)
        return tuple(sorted(prefixes))

    def _initial_to_current_mask_flips(self) -> dict[str, int] | None:
        if self._initial_binary_masks is None or self.masks is None:
            return None
        return self._mask_flip_stats(self._initial_binary_masks)

    def _run_final_recovery(self, batches: Sequence[Tensor]) -> None:
        if not self.config.training.train_weights:
            return
        if self.weight_optimizer is None or self.masks is None:
            return
        for index in range(self.config.training.final_recovery_steps):
            batch = batches[index % len(batches)]
            if self.mask_optimizer is not None:
                self.mask_optimizer.zero_grad(set_to_none=True)
            self.weight_optimizer.zero_grad(set_to_none=True)
            with apply_structured_masks(self.model, self.masks):
                outputs = self.model(input_ids=batch, labels=batch)
            outputs.loss.backward()
            if self.mask_optimizer is not None:
                self.mask_optimizer.zero_grad(set_to_none=True)
            self._clip_weight_gradients()
            self.weight_optimizer.step()

    def _run_stabilization(self, batches: Sequence[Tensor]) -> None:
        """Pause mask updates and let enabled weights adapt under the current masks."""

        if not self.config.training.train_weights:
            return
        if self.weight_optimizer is None or self.masks is None:
            return
        for index in range(self.config.training.stabilization_steps):
            batch = batches[index % len(batches)]
            if self.mask_optimizer is not None:
                self.mask_optimizer.zero_grad(set_to_none=True)
            self.weight_optimizer.zero_grad(set_to_none=True)
            with apply_structured_masks(self.model, self.masks):
                outputs = self.model(input_ids=batch, labels=batch)
            outputs.loss.backward()
            if self.mask_optimizer is not None:
                self.mask_optimizer.zero_grad(set_to_none=True)
            self._clip_weight_gradients()
            self.weight_optimizer.step()

    def _save_checkpoint(self, checkpoint_dir: Path, step: int, final: bool = False) -> Path:
        if self.masks is None:
            raise RuntimeError("Masks were not initialized.")
        name = "final_training_checkpoint.pt" if final else f"training_checkpoint_step_{step}.pt"
        path = checkpoint_dir / name
        checkpoint: dict[str, Any] = {
            "step": step,
            "final": final,
            "config": config_to_dict(self.config),
            "mask_state_dict": self.masks.state_dict(),
            "mask_optimizer_state_dict": (
                self.mask_optimizer.state_dict() if self.mask_optimizer is not None else None
            ),
            "weight_optimizer_state_dict": (
                self.weight_optimizer.state_dict() if self.weight_optimizer is not None else None
            ),
        }
        if self.config.training.train_weights:
            checkpoint["model_state_dict"] = self.model.state_dict()
        torch.save(checkpoint, path)
        mask_state_name = "final_mask_state.pt" if final else f"mask_state_step_{step}.pt"
        save_mask_state(checkpoint_dir / mask_state_name, self.masks)
        return path

    def _make_progress_bar(self) -> object | None:
        if not self.config.training.show_progress:
            return None
        from tqdm.auto import tqdm

        return tqdm(
            total=self.config.training.max_steps,
            desc="MaGRIP training",
            dynamic_ncols=True,
            leave=True,
        )

    def _update_progress_bar(
        self,
        progress: object | None,
        metric: TrainingStepMetrics,
    ) -> None:
        if progress is None:
            return
        objective = metric.objective
        postfix = {
            "task": f"{objective['task_loss']:.4f}",
            "total": f"{objective['total_loss']:.4f}",
            "soft": f"{objective['retained_cost_ratio']:.3f}",
            "hard": f"{metric.hard_retained_cost_ratio:.3f}",
            "grad": (
                "nan"
                if metric.mask_grad_norm is None
                else f"{metric.mask_grad_norm:.3f}"
            ),
            "tau": f"{metric.temperature:.3f}",
        }
        if metric.validation_loss is not None:
            postfix["val"] = f"{metric.validation_loss:.4f}"
        progress.set_postfix(postfix)
        progress.update(1)


def collect_saliency_over_batches(
    model: nn.Module,
    targets: list[FFNTarget],
    batches: Sequence[Tensor],
    config: SaliencyConfig | None = None,
) -> SaliencyResult:
    """Collect and average saliency over calibration batches."""

    if not batches:
        raise ValueError("batches must not be empty.")
    accumulated: SaliencyResult | None = None
    for batch in batches:
        result = collect_saliency(
            model=model,
            targets=targets,
            input_ids=batch,
            labels=batch,
            config=config,
        )
        if accumulated is None:
            accumulated = result
        else:
            accumulated.add_(result)
    if accumulated is None:
        raise RuntimeError("No saliency was collected.")
    accumulated.divide_(len(batches))
    return accumulated


def mean_causal_lm_loss(
    model: nn.Module,
    batches: Sequence[Tensor],
    masks: MaskCollection | None = None,
) -> float:
    """Average causal-LM loss over batches."""

    if not batches:
        raise ValueError("batches must not be empty.")
    losses = [
        causal_lm_loss(model, input_ids=batch, labels=batch, masks=masks)
        for batch in batches
    ]
    return float(sum(losses) / len(losses))


def training_result_to_summary(result: TrainingResult) -> dict[str, Any]:
    """Convert a training result into JSON-friendly summary fields."""

    mask_cost = total_mask_cost(result.masks)
    return {
        "num_steps": result.num_steps,
        "num_tokens": result.num_tokens,
        "baseline_loss": result.baseline_loss,
        "initial_masked_loss": result.initial_masked_loss,
        "final_masked_loss": result.final_masked_loss,
        "baseline_perplexity": result.baseline_perplexity,
        "initial_masked_perplexity": result.initial_masked_perplexity,
        "final_masked_perplexity": result.final_masked_perplexity,
        "mask_cost": {
            "full_cost": mask_cost.full_cost,
            "retained_cost": mask_cost.retained_cost,
            "retained_ratio": mask_cost.retained_ratio,
            "full_flop_cost": mask_cost.full_flop_cost,
            "retained_flop_cost": mask_cost.retained_flop_cost,
            "flop_retained_ratio": mask_cost.flop_retained_ratio,
        },
        "metrics": [
            {
                "step": item.step,
                "epoch_index": item.epoch_index,
                "batch_index": item.batch_index,
                "temperature": item.temperature,
                "mask_update": item.mask_update,
                "weight_update": item.weight_update,
                "objective": item.objective,
                "hard_retained_cost_ratio": item.hard_retained_cost_ratio,
                "hard_retained_flop_ratio": item.hard_retained_flop_ratio,
                "mask_grad_norm": item.mask_grad_norm,
                "mask_grad_mean": item.mask_grad_mean,
                "mask_grad_min": item.mask_grad_min,
                "mask_grad_max": item.mask_grad_max,
                "mask_grad_nonzero_count": item.mask_grad_nonzero_count,
                "mask_grad_target_count": item.mask_grad_target_count,
                "mask_update_mean_abs": item.mask_update_mean_abs,
                "mask_update_max_abs": item.mask_update_max_abs,
                "weight_grad_norm": item.weight_grad_norm,
                "apollo_diagnostics": item.apollo_diagnostics,
                "active_to_inactive_count": item.active_to_inactive_count,
                "inactive_to_active_count": item.inactive_to_active_count,
                "mask_flip_count": item.mask_flip_count,
                "validation_loss": item.validation_loss,
                "validation_perplexity": item.validation_perplexity,
                "saliency_drift": item.saliency_drift,
            }
            for item in result.metrics
        ],
        "checkpoints": result.checkpoints,
        "weight_trainable_parameter_count": result.weight_trainable_parameter_count,
        "apollo_parameter_stats": result.apollo_parameter_stats,
        "initial_to_final_mask_flips": result.initial_to_final_mask_flips,
    }


def _matches_any_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _parameter_gradient_norm(parameters: Sequence[nn.Parameter]) -> float | None:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return None
    total = torch.linalg.vector_norm(torch.stack(norms))
    return float(total.cpu().item())


def config_to_dict(config: MaGRIPConfig) -> dict[str, Any]:
    """Return a JSON/checkpoint friendly config dictionary."""

    return {
        "objective": dict(config.objective.__dict__),
        "mask_schedule": dict(config.mask_schedule.__dict__),
        "training": dict(config.training.__dict__),
        "optimizer": dict(config.optimizer.__dict__),
        "seed": config.seed,
    }
