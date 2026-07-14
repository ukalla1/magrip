from types import SimpleNamespace

import torch
from torch import nn

from magrip.config import (
    MaGRIPConfig,
    MaskScheduleConfig,
    ObjectiveConfig,
    OptimizerConfig,
    TrainingConfig,
)
from magrip.trainer import MaGRIPTrainer
from magrip.topology import FFNTarget, FFNTopologyKind


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_in = nn.Linear(4, 6)
        self.fc_out = nn.Linear(6, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(torch.relu(self.fc_in(x)))


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = ToyMLP()


class ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = ToyBlock()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        logits = self.block.mlp(input_ids.float())
        loss = (logits - input_ids.float()).pow(2).mean()
        return SimpleNamespace(loss=loss, logits=logits)


def dense_target() -> FFNTarget:
    return FFNTarget(
        block_index=0,
        block_path="block",
        ffn_path="block.mlp",
        topology=FFNTopologyKind.DENSE,
        expand_module_paths=("block.mlp.fc_in",),
        contract_module_paths=("block.mlp.fc_out",),
        intermediate_size=6,
        hidden_size=4,
        registry_name="toy_dense",
    )


def test_trainer_runs_mask_only_step() -> None:
    torch.manual_seed(0)
    model = ToyCausalLM()
    config = MaGRIPConfig(
        objective=ObjectiveConfig(
            target_retained_ratio=0.5,
            budget_penalty_weight=0.1,
        ),
        mask_schedule=MaskScheduleConfig(
            initial_temperature=1.0,
            min_temperature=0.5,
            temperature_decay=0.95,
        ),
        training=TrainingConfig(
            max_steps=2,
            train_weights=False,
            train_masks=True,
            final_harden=True,
        ),
        optimizer=OptimizerConfig(mask_learning_rate=1e-2),
    )
    trainer = MaGRIPTrainer(model=model, config=config, targets=[dense_target()])
    batches = [torch.randn(2, 3, 4)]

    result = trainer.train(batches)

    assert result.num_steps == 2
    assert len(result.metrics) == 2
    assert result.masks.as_dict()["block.mlp"].active_channels == 3
    assert result.metrics[0].objective["task_loss"] > 0.0
    assert result.metrics[0].mask_grad_target_count == 1
    assert result.metrics[0].mask_grad_nonzero_count == 1
    assert result.metrics[0].mask_update_mean_abs is not None


def test_soft_mask_warmup_uses_relaxed_forward_mode() -> None:
    torch.manual_seed(0)
    model = ToyCausalLM()
    config = MaGRIPConfig(
        objective=ObjectiveConfig(target_retained_ratio=0.5),
        mask_schedule=MaskScheduleConfig(
            soft_warmup_steps=2,
            initial_temperature=1.0,
        ),
        training=TrainingConfig(
            max_steps=1,
            train_weights=False,
            train_masks=True,
            final_harden=False,
        ),
        optimizer=OptimizerConfig(mask_learning_rate=1e-2),
    )
    trainer = MaGRIPTrainer(model=model, config=config, targets=[dense_target()])
    batches = [torch.randn(2, 3, 4)]

    result = trainer.train(batches)
    mask = result.masks.as_dict()["block.mlp"]

    assert mask.hard is False
    assert mask.ste is False


def test_soft_mask_warmup_switches_to_hard_ste_after_warmup() -> None:
    torch.manual_seed(0)
    model = ToyCausalLM()
    config = MaGRIPConfig(
        objective=ObjectiveConfig(target_retained_ratio=0.5),
        mask_schedule=MaskScheduleConfig(
            soft_warmup_steps=1,
            initial_temperature=1.0,
            use_ste=True,
        ),
        training=TrainingConfig(
            max_steps=2,
            train_weights=False,
            train_masks=True,
            final_harden=False,
        ),
        optimizer=OptimizerConfig(mask_learning_rate=1e-2),
    )
    trainer = MaGRIPTrainer(model=model, config=config, targets=[dense_target()])
    batches = [torch.randn(2, 3, 4)]

    result = trainer.train(batches)
    mask = result.masks.as_dict()["block.mlp"]

    assert mask.hard is True
    assert mask.ste is True
