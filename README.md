# MaGRIP v2

MaGRIP v2 is a work-in-progress framework for Magnitude and Gradient Informed Pruning of large language models.

The current focus is project scaffolding and design. The mathematical reference is [`docs/THEORY.tex`](docs/THEORY.tex), and the implementation roadmap is [`PLAN.md`](PLAN.md).

## Current Status

- Theory document: drafted.
- Framework plan: drafted.
- Python package skeleton: initialized.
- M1 dense GPT-2 frozen-pruning baseline: validated.
- M1 Gemma/gated frozen-pruning baseline: validated.

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

Both `models/` outputs and `outputs/` logs are ignored by git by default.

Validated M1 reference runs:

- Dense GPT-2: `outputs/runs/gpt2_smoke_20260710_001150`
- Gated Gemma: `outputs/runs/gpt2_smoke_20260710_121506`

For a pure wiring check without downloading a dataset, use:

```bash
python scripts/run_gpt2_smoke.py --calibration-source text --num-samples 1
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
