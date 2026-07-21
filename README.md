# MaGRIP v2

MaGRIP v2 is a working alpha framework for Magnitude and Gradient Informed Pruning of
large language models. It discovers transformer-block FFN targets, learns structured
intermediate-channel masks, jointly adapts weights with APOLLO, physically compacts the
selected FFN channels, and exports deployable Hugging Face and GGUF artifacts.

The mathematical reference is [`docs/THEORY.tex`](docs/THEORY.tex). The completed v2
alpha roadmap is archived in [`docs/COMPLETED_ROADMAP.md`](docs/COMPLETED_ROADMAP.md).

## Current Status

v2 alpha is complete for dense and gated FFN architectures.

- M1-M4: topology-aware discovery, masks, saliency, and v1 behavior preservation are implemented.
- M5: budget-aware objective and mask training loop are implemented.
- M6: APOLLO/APOLLO-Mini joint mask/weight adaptation is implemented and inspected on GPT-2,
  Gemma-2B, GPT2-XL, and Qwen3-8B runs.
- M7: structural compaction is implemented for dense and gated FFNs, with Hugging Face save
  and optional llama.cpp GGUF export.
- M8: experiment tracking writes research artifacts for training, validation, compaction,
  and deployment inspection.
- End-to-end deployment proof point: Gemma-2B-IT at 60 percent FFN retention was trained,
  compacted, converted to GGUF, loaded in llama.cpp, and queried successfully.

Observed server-side artifact size for the Gemma-2B-IT/Gemma-family GGUF path:

- baseline GGUF: about 5 GB;
- compacted MaGRIP GGUF: about 3.2 GB;
- reduction: about 34 percent on disk.

## Default Assumptions

- FFN pruning targets are restricted to transformer blocks.
- Distillation is disabled by default with `beta = 0.0`.
- M6 uses APOLLO/APOLLO-Mini as the model-weight optimizer path for joint mask/weight
  adaptation.
- M7 writes a Transformers/Hugging Face compacted model first. GGUF is a separate
  llama.cpp deployment artifact exported from that compacted HF directory.

## End-To-End Gemma-2B-IT Run

Use the instruction-tuned Gemma model for interactive llama.cpp behavior:

```bash
pip install -r requirements.txt
pip install -e .
huggingface-cli login
```

Train/prune with APOLLO:

```bash
python scripts/run_experiment_config.py \
  configs/experiments/m6_train_gemma-2b-it_wikitext2_r060_full_v1.json
```

Compact and export GGUF:

```bash
python scripts/run_experiment_config.py \
  configs/experiments/m7_compact_gemma-2b-it_wikitext2_r060_full_v1.json
```

Run the compacted model with llama.cpp:

```bash
./llama.cpp/build/bin/llama-cli \
  -m models/Compacted/google__gemma-2b-it_magrip_r060_v1/gguf/gemma-2b-it_magrip_r060_v1.gguf \
  --conversation \
  --chat-template gemma \
  -p "Artificial intelligence is" \
  -n 100 \
  --temp 0.2 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  -c 2048 \
  --n-gpu-layers all \
  -t 8
```

The compacted Gemma-2B-IT model produced coherent completions in llama.cpp after GGUF
export. Base `google/gemma-2b` is not recommended for chat-style interactive testing;
use `google/gemma-2b-it` or another instruction-tuned model for deployment checks.

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
M8 research artifacts include:

- `stage_metrics.csv`: baseline, initial masked, and final masked loss/perplexity.
- `validation_curve.csv`: held-out evaluation trajectory.
- `training_windows.csv`: coarse dynamics summaries for long runs.
- `layer_diagnostics.csv`: per-FFN retained ratios plus mask/saliency statistics.
- `channel_diagnostics.pkl`: per-channel masks, logits, probabilities, saliency, and costs.
- `research_summary.json`: paper-oriented summary of loss, budget, APOLLO, and mask dynamics.
- `RUN_CARD.md`: short human-readable run card.

Audit and plot a completed M5 run with:

```bash
python scripts/plot_magrip_run.py outputs/runs/<run>/summary.json \
  --mask-state models/Pruned/<model>_magrip_train/mask_state.pt
```

Experiment recipes live under `configs/experiments/` and can be launched with:

```bash
python scripts/run_experiment_config.py configs/experiments/m6_train_gemma-2b_wikitext2_r060_full_v3.json
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

## M7 Structural Compaction

M7 physically removes the FFN channels selected out by final hardened masks. Use the
final training checkpoint when available so compaction starts from APOLLO-adapted
weights rather than the original baseline weights:

```bash
python scripts/compact_model.py \
  --model-name Qwen/Qwen3-8B \
  --checkpoint outputs/runs/<run>/checkpoints/final_model_state_dict.pt \
  --mask-state outputs/runs/<run>/checkpoints/final_mask_state.pt \
  --output-dir models/Compacted/Qwen__Qwen3-8B_magrip_r060 \
  --device cuda \
  --torch-dtype bfloat16 \
  --verify-text "MaGRIP compaction check." \
  --verification-policy local-targets \
  --local-target-policy dtype-aware \
  --eval-num-samples 512
```

New training runs also save `checkpoints/final_model_state_dict.pt` and
`checkpoints/final_mask_state.pt`. Use those two files from the same run directory for
compaction so the adapted weights and hardened masks cannot drift apart. Older APOLLO
runs may only have `final_training_checkpoint.pt`; `compact_model.py` can load those
trusted local MaGRIP checkpoints by falling back to PyTorch full-pickle loading.

The script verifies masked-model logits against compacted-model logits before saving,
then writes:

- `config.json`, model shards, and tokenizer files in Hugging Face format.
- `magrip_compaction_manifest.json` with retained-channel counts and verification stats.
- `metrics/compaction_stage_metrics.csv` with masked-reference vs compacted evaluation
  when `--eval-num-samples` is positive.
- `metrics/logit_equivalence.csv` with masked-vs-compacted logit error statistics.
  Verification is incremental by default, so structural mismatches report the first FFN
  target whose compaction breaks equivalence. For BF16 compaction, use
  `--verification-policy local-targets --local-target-policy dtype-aware` to accept
  compaction when each compacted FFN target matches its masked reference up to
  dtype-scaled local numerical drift, while still logging full-logit drift.

To export GGUF for llama.cpp after HF compaction:

```bash
python scripts/compact_model.py \
  --model-name Qwen/Qwen3-8B \
  --checkpoint outputs/runs/<run>/checkpoints/final_model_state_dict.pt \
  --mask-state outputs/runs/<run>/checkpoints/final_mask_state.pt \
  --output-dir models/Compacted/Qwen__Qwen3-8B_magrip_r060 \
  --device cuda \
  --torch-dtype bfloat16 \
  --export-gguf \
  --llama-cpp-dir /path/to/llama.cpp \
  --gguf-out models/Compacted/Qwen__Qwen3-8B_magrip_r060.gguf \
  --gguf-outtype bf16
```

GGUF export depends on the installed llama.cpp converter supporting the compacted model
architecture. For SentencePiece models such as Gemma or LLaMA, `compact_model.py`
preserves `tokenizer.model` from the original tokenizer source before running the
converter. If the tokenizer should come from a different local directory or Hub repo,
pass `--tokenizer-source <path-or-repo-id>`.
