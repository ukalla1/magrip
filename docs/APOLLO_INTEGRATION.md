# APOLLO Integration

M6 moves MaGRIP from mask-only refinement to joint differentiable adaptation. The model
weights `theta` are updated with APOLLO/APOLLO-Mini, while mask logits `phi` stay on their
own lightweight optimizer. Distillation remains disabled with `beta = 0`.

## Dependency

Install the APOLLO package in the server environment:

```bash
pip install apollo-torch
```

or install the upstream repository in editable mode:

```bash
git clone https://github.com/zhuhanqing/APOLLO.git
cd APOLLO
pip install -e .
```

The MaGRIP optimizer builder imports `apollo_torch.APOLLOAdamW`.

## Training Flow

M6 keeps the M5 saliency warm start:

1. Run the unmasked model on calibration data.
2. Compute MaGRIP saliency for FFN intermediate channels.
3. Initialize mask logits from saliency and shift them to the target retained budget.
4. Jointly update:
   - `theta` with APOLLO/APOLLO-Mini;
   - `phi` with the mask optimizer;
   - the task, budget, and mask regularization terms with `beta = 0`.
5. Use optional soft-mask warmup before hard STE.
6. Harden final top-k masks.

## APOLLO Settings

The runner exposes:

- `--use-apollo`
- `--apollo-variant {apollo,apollo-mini}`
- `--apollo-rank`
- `--apollo-scale`
- `--apollo-update-proj-gap`
- `--apollo-proj`
- `--apollo-proj-type`
- `--apollo-scale-type {channel,tensor}`
- `--apollo-parameter-scope {all,ffn}`

`--use-apollo` implies model-weight training. APOLLO-Mini uses rank `1`, tensor scaling,
and scale `128` in the optimizer adapter.

Use `--apollo-parameter-scope all` for the main M6 path. Use `ffn` only when debugging
memory pressure or isolating FFN-local behavior.

## Soft-Mask Warmup

Use `--soft-warmup-steps N` to keep masks relaxed for the first `N` optimization steps.
During warmup, forward hooks apply the continuous probabilities `q = sigmoid(phi / tau)`.
After warmup, the trainer switches back to hard STE masks.

This gives initially inactive channels a smoother route to re-enter the retained set
before final top-k hardening.

## Qwen3-8B Starter Command

```bash
python scripts/run_magrip_train.py \
  --model-name Qwen/Qwen3-8B \
  --device cuda \
  --torch-dtype bfloat16 \
  --use-apollo \
  --apollo-variant apollo-mini \
  --apollo-parameter-scope all \
  --max-steps 600 \
  --retained-ratio 0.6 \
  --dataset-split train \
  --eval-dataset-split validation \
  --num-samples 512 \
  --eval-num-samples 64 \
  --max-length 256 \
  --batch-size 1 \
  --budget-penalty-weight 25.0 \
  --mask-learning-rate 2e-3 \
  --weight-learning-rate 1e-5 \
  --temperature-decay 0.999 \
  --soft-warmup-steps 100 \
  --eval-every 50 \
  --checkpoint-every 100
```

Inspect:

- `training.weight_trainable_parameter_count`
- `training.apollo_parameter_stats`
- `training.apollo_parameter_stats.estimated_optimizer_state_mib_fp32`
- `training.apollo_parameter_stats.estimated_state_ratio_vs_adamw`
- `training.metrics[*].weight_grad_norm`
- `training.metrics[*].apollo_diagnostics.optimizer_state_tensor_norm`
- `training.metrics[*].apollo_diagnostics.projected_state_tensor_norm`
- `training.metrics[*].apollo_diagnostics.update_state_tensor_norm`
- `training.metrics[*].mask_grad_nonzero_count`
- `training.metrics[*].mask_flip_count`
- `training.metrics[*].validation_loss`
- `training.initial_to_final_mask_flips`
- `training.initial_masked_loss` versus `training.final_masked_loss`
