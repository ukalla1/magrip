# Objectives and Training Loop

M5 implements the budget-aware objective and training loop from `docs/THEORY.tex`.

## Objective

The default objective is:

```text
L_total = L_task
        + lambda_t (Cost(q) / Cost(1) - rho_t)^2
        + gamma * Omega(phi)
```

where:

- `q = sigmoid(phi / tau)` is the relaxed mask probability;
- the forward path may use hard STE masks;
- `Cost(q)` is differentiable and uses M3 channel costs;
- `rho_t` can anneal from an initial budget to the final retained budget;
- `lambda_t` can anneal from weak pressure to the final penalty weight;
- `Omega(phi)` defaults to entropy-style `q(1-q)` regularization.

Distillation remains disabled by default. M5 supports cached/teacher logits as an optional
objective mode, but no training script enables it unless explicitly configured.

## Training Flow

The M5 trainer performs:

1. Stage 0 saliency warm start using M4 saliency.
2. Trainable mask initialization from saliency logits.
3. Joint optimization loop:
   - mask optimizer updates `phi`;
   - optional weight optimizer updates model parameters;
   - APOLLO is intentionally reserved for M6.
4. Optional saliency recomputation for drift diagnostics.
5. Optional stabilization with mask updates paused.
6. Final top-k hardening.
7. Optional final weight-only recovery.

The default is mask-only training, which is the safest way to validate M5 before APOLLO.

## Server Smoke Command

```bash
python scripts/run_magrip_train.py \
  --model-name gpt2 \
  --device cuda \
  --torch-dtype bfloat16 \
  --max-steps 20 \
  --retained-ratio 0.7 \
  --dataset-split train \
  --eval-dataset-split validation \
  --num-samples 8 \
  --max-length 128 \
  --batch-size 1
```

For gated models:

```bash
python scripts/run_magrip_train.py \
  --model-name google/gemma-2b \
  --device cuda \
  --torch-dtype bfloat16 \
  --max-steps 20 \
  --retained-ratio 0.7 \
  --dataset-split train \
  --eval-dataset-split validation \
  --num-samples 8 \
  --max-length 128 \
  --batch-size 1
```

The output directory contains:

- `summary.json` with objective traces under `training.metrics`;
- `metrics/metrics.pkl`, `metrics/metrics.json`, and `metrics/metrics.csv` for analysis;
- `events.jsonl` with structured run events;
- `models/Pruned/<model>_magrip_train/mask_state.pt`;
- `models/Pruned/<model>_magrip_train/masks.pt`;
- optional full training checkpoints when `--checkpoint-every` is set.

With Qwen3-8B:

```bash
python scripts/run_magrip_train.py \
  --model-name Qwen/Qwen3-8B \
  --device cuda \
  --torch-dtype bfloat16 \
  --max-steps 100 \
  --retained-ratio 0.7 \
  --dataset-split train \
  --eval-dataset-split validation \
  --num-samples 64 \
  --eval-num-samples 16 \
  --max-length 256 \
  --batch-size 1 \
  --budget-penalty-weight 10.0 \
  --mask-learning-rate 5e-3 \
  --temperature-decay 0.995 \
  --recompute-saliency-every 0 \
  --checkpoint-every 50
```

## Tests

M5 adds:

- `tests/test_objectives.py` for budget schedules and differentiable objective terms;
- `tests/test_trainer.py` for a tiny mask-only training loop.
