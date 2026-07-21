# MaGRIP Experiments

This document is the running experiment ledger for MaGRIP v2. It records what each run
is meant to prove, where its artifacts live, and what evidence is needed before using a
result in a paper figure or table.

## Artifact Contract

Every M6+ training run should produce:

- `summary.json`: full run manifest and final scalar metrics.
- `events.jsonl`: timestamped load, discovery, data, training, and artifact events.
- `metrics/metrics.json`, `metrics/metrics.csv`, `metrics/metrics.pkl`: raw per-step traces.
- `metrics/stage_metrics.csv`: baseline, initial masked, and final masked loss/perplexity.
- `metrics/validation_curve.csv`: held-out loss/perplexity over training.
- `metrics/training_windows.csv`: coarse dynamics summaries for long runs.
- `metrics/layer_diagnostics.csv`: per-FFN retained ratio, mask statistics, and saliency statistics.
- `metrics/channel_diagnostics.pkl`: per-channel masks, logits, probabilities, saliency, and costs.
- `metrics/channel_diagnostics_manifest.json`: lightweight channel artifact index.
- `metrics/research_summary.json`: paper-oriented run summary.
- `RUN_CARD.md`: human-readable run card.

Every M7 compaction run should produce:

- Hugging Face model files under `models/Compacted/<model>`.
- `magrip_compaction_manifest.json`: compaction, verification, optional GGUF, and optional eval metadata.
- `metrics/compaction_summary.json`: copy of the compaction manifest for analysis.
- `metrics/compaction_stage_metrics.csv`: masked-reference vs compacted loss/perplexity when eval is enabled.
- `metrics/compaction_targets.csv`: per-target retained channels after physical compaction.
- `metrics/logit_equivalence.csv`: masked-vs-compacted logit equivalence statistics.

## Current Validated Milestones

M6 has been inspected on dense and gated architectures:

- GPT-2 smoke: validates plumbing.
- Gemma-2B: gated target-scale behavior at 60 percent FFN retention.
- GPT2-XL: dense large-model behavior; mechanically healthy but needs longer/tighter budget tuning.
- Qwen3-8B: gated target-scale behavior at 60 percent FFN retention.

M7/M8 are validated end-to-end on Gemma-2B-IT:

- trained/pruned at 60 percent FFN retention;
- structurally compacted from final hard masks;
- saved in Hugging Face format;
- converted to GGUF through llama.cpp;
- loaded and queried successfully with `llama-cli`;
- reduced server-side GGUF disk footprint from roughly 5 GB baseline to 3.2 GB compacted.

## Experiment Recipes

Reference configs live in `configs/experiments/`:

- `m6_train_gpt2_wikitext2_r070_smoke_v1.json`
- `m6_train_gemma-2b_wikitext2_r060_full_v3.json`
- `m6_train_gemma-2b-it_wikitext2_r060_full_v1.json`
- `m6_train_gpt2-xl_wikitext2_r060_tuned_v1.json`
- `m6_train_qwen3-8b_wikitext2_r060_full_v1.json`
- `m7_compact_gemma-2b_wikitext2_r060_full_v3.json`
- `m7_compact_gemma-2b-it_wikitext2_r060_full_v1.json`

These files record reproducible command arguments. They are intentionally simple JSON
recipes rather than a new config framework.

## Minimum Paper-Quality Checks

Before a run is used in a figure or table, verify:

- Discovery found the expected number and topology of FFN targets.
- `per_target_grad_coverage` is complete throughout mask updates.
- APOLLO diagnostics are nonzero for M6 runs.
- Validation loss/perplexity is logged and does not show an obvious overfit pattern.
- Final saved mask cost matches the target retained ratio.
- Compaction passes dtype-aware local FFN-target equivalence and logs strict-logit drift.
- Compacted model reloads from Hugging Face format.
- GGUF export succeeds for llama.cpp-supported architectures when deployment is claimed.
- Deployment smoke prompt produces coherent text for instruction-tuned models.

## Planned Ablations

- MaGRIP masks only, no APOLLO.
- APOLLO only, no pruning.
- Random 60 percent structured FFN mask plus APOLLO.
- MaGRIP 60 percent structured FFN mask plus APOLLO.
- Retention sweeps: 70, 60, 50 percent FFN retention.
- Dense vs gated architecture comparison under matched token budget.
