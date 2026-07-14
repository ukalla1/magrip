# MaGRIP v2

MaGRIP v2 is a work-in-progress framework for Magnitude and Gradient Informed Pruning of large language models.

The current focus is topology-aware FFN discovery and smoke-test validation. The mathematical reference is [`docs/THEORY.tex`](docs/THEORY.tex), and the implementation roadmap is [`PLAN.md`](PLAN.md).

## Current Status

- Theory document: drafted.
- Framework plan: drafted.
- Python package skeleton: initialized.
- M1 dense GPT-2 frozen-pruning baseline: validated.
- M1 Gemma/gated frozen-pruning baseline: validated.
- M2 FFN discovery registry: implemented for common dense and gated transformer FFNs.
- M2 artifact validation: available for smoke-run summaries.
- M3 mask system: implemented with structured masks, STE-ready logits, cost accounting,
  and serialization.
- M4 saliency system: implemented with contraction-input saliency, branch diagnostics,
  normalization modes, and drift tracking.
- M5 objective/training loop: implemented for mask-only training by default, with optional
  AdamW weight updates before APOLLO integration.

## Default Assumptions

- FFN pruning targets are restricted to transformer blocks.
- Distillation is disabled by default with `beta = 0.0`.
- APOLLO will be integrated later as an optional optimizer backend for model-weight adaptation.

## GPT-2 Smoke Test

Install dependencies in your environment, then run:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/run_gpt2_smoke.py --model-name gpt2 --retained-ratio 0.7
```

By default, the smoke test uses `Salesforce/wikitext` with `wikitext-2-raw-v1` validation as a small calibration dataset:

```bash
python scripts/run_gpt2_smoke.py \
  --model-name gpt2 \
  --device cuda \
  --retained-ratio 0.7 \
  --num-samples 8 \
  --max-length 128 \
  --batch-size 1
```

The smoke test saves mask artifacts under `models/Pruned/`. Baseline models are cached under `models/Baselines/` by default so repeat runs load locally instead of downloading from Hugging Face. Use `--no-cache-baseline` to disable this.

Each run also writes structured logs under `outputs/runs/<run-name>/`:

- `events.jsonl`: timestamped events for loading, target discovery, pruning, metrics, and artifacts.
- `summary.json`: final manifest with loss/perplexity deltas, mask stats, and saliency stats.
  M4 saliency summaries include the primary saliency source and optional branch diagnostics.

The pruned artifact directory also includes:

- `masks.pt`: binary mask tensors for quick inspection.
- `mask_state.pt`: reloadable structured masks with logits, thresholds, temperatures, and
  channel-cost metadata.
- `summary.json`: aggregate retained FFN parameter and FLOP-cost ratios under `mask_cost`.

Both `models/` outputs and `outputs/` logs are ignored by git by default.

Validated M1 reference runs:

- Dense GPT-2: `outputs/runs/gpt2_smoke_20260710_001150`
- Gated Gemma: `outputs/runs/gpt2_smoke_20260710_121506`

For a pure wiring check without downloading a dataset, use:

```bash
python scripts/run_gpt2_smoke.py --calibration-source text --num-samples 1
```

## FFN Discovery Inspection

Before a pruning run, inspect what MaGRIP will target:

```bash
python scripts/inspect_model.py --model-name gpt2
python scripts/inspect_model.py --model-name google/gemma-2b
```

Expected M2 signals:

- GPT-2 reports 12 dense FFN targets.
- Gemma-2B reports 18 gated FFN targets.
- The command ends with `Validation: OK`.

Validate existing smoke artifacts with:

```bash
python scripts/validate_smoke_artifact.py outputs/runs/gpt2_smoke_20260710_001150/summary.json --expected-topology dense
python scripts/validate_smoke_artifact.py outputs/runs/gpt2_smoke_20260710_121506/summary.json --expected-topology gated
```

Inspect saved structured mask artifacts with:

```bash
python scripts/inspect_mask_state.py models/Pruned/gpt2_magrip_smoke --strict
python scripts/inspect_mask_state.py models/Pruned/google__gemma-2b_magrip_smoke --strict
```

## Gated FFN Smoke Test

For gated models such as Gemma, first authenticate with Hugging Face on the server:

```bash
huggingface-cli login
```

or export a token:

```bash
export HF_TOKEN=...
```

Then run the generic smoke entrypoint:

```bash
python scripts/run_magrip_smoke.py \
  --model-name google/gemma-2b \
  --device cuda \
  --retained-ratio 0.7 \
  --num-samples 8 \
  --max-length 128 \
  --batch-size 1
```

The same entrypoint also works for dense models:

```bash
python scripts/run_magrip_smoke.py --model-name gpt2 --device cuda
```

## M5 Training Loop

Run mask-only M5 training with:

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

For Gemma:

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

The training summary writes objective traces under `training.metrics`.
Additional analysis artifacts are written under `outputs/runs/<run>/metrics/`.

Audit and plot a completed M5 run with:

```bash
python scripts/plot_magrip_run.py outputs/runs/<run>/summary.json \
  --mask-state models/Pruned/<model>_magrip_train/mask_state.pt
```

## M6 APOLLO Training

M6 enables joint APOLLO weight adaptation and mask-logit training:

```bash
python scripts/run_magrip_train.py \
  --model-name Qwen/Qwen3-8B \
  --device cuda \
  --torch-dtype bfloat16 \
  --use-apollo \
  --apollo-variant apollo-mini \
  --soft-warmup-steps 100 \
  --retained-ratio 0.6
```

See `docs/APOLLO_INTEGRATION.md` for the full server command and expected diagnostics.
