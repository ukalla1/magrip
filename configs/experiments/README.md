# MaGRIP Experiment Recipes

These JSON files capture reproducible command recipes for common MaGRIP runs. They are
not a separate config system yet; each file records the intended script and CLI
arguments so server runs can be repeated exactly.

Recipe filenames follow:

```text
m<stage>_<action>_<model>_<dataset>_r<retain>_<variant>_v<version>.json
```

Training recipes set an explicit `run-name` so downstream compaction recipes can refer
to deterministic checkpoint paths. Compaction recipes do not have a `run-name` argument;
version them through `output-dir`, `checkpoint`, and `gguf-out`.

- `m6_train_gpt2_wikitext2_r070_smoke_v1.json`: quick dense APOLLO smoke run.
- `m6_train_gemma-2b_wikitext2_r060_full_v3.json`: gated Gemma-2B base run.
- `m6_train_gemma-2b-it_wikitext2_r060_full_v1.json`: gated Gemma-2B-IT instruction run.
- `m6_train_gpt2-xl_wikitext2_r060_tuned_v1.json`: dense large-model tuned run.
- `m6_train_qwen3-8b_wikitext2_r060_full_v1.json`: target-scale gated run.
- `m7_compact_gemma-2b_wikitext2_r060_full_v3.json`: HF/GGUF compaction for Gemma-2B.
- `m7_compact_gemma-2b-it_wikitext2_r060_full_v1.json`: HF/GGUF compaction for Gemma-2B-IT.
