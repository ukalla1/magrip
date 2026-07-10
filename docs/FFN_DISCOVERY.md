# FFN Discovery

MaGRIP only discovers prunable FFNs inside repeated transformer blocks. This keeps M2 away from embeddings, LM heads, classifiers, and other non-transformer modules.

## Block Stacks

M2 checks these common block-stack paths:

- `transformer.h`
- `model.layers`
- `language_model.model.layers`
- `transformer.layers`
- `decoder.layers`
- `gpt_neox.layers`

## Supported Topologies

Dense FFNs:

- `c_fc` -> `c_proj`
- `fc_in` -> `fc_out`
- `dense_h_to_4h` -> `dense_4h_to_h`

Gated FFNs:

- `gate_proj`, `up_proj` -> `down_proj`
- `w1`, `w3` -> `w2`

MoE-like FFNs are detected and skipped in M2. They should become a separate design path later because expert routing and per-expert pruning need different accounting.

## Validation

Use:

```bash
python scripts/inspect_model.py --model-name gpt2
python scripts/inspect_model.py --model-name google/gemma-2b
```

Expected signs:

- GPT-2 reports 12 `dense` targets.
- Gemma-2B reports 18 `gated` targets.
- `Validation: OK`.

Smoke artifact validation:

```bash
python scripts/validate_smoke_artifact.py outputs/runs/gpt2_smoke_20260710_001150/summary.json --expected-topology dense
python scripts/validate_smoke_artifact.py outputs/runs/gpt2_smoke_20260710_121506/summary.json --expected-topology gated
```

Old low-precision artifacts may warn that `active_channels` differs from the mask tensor sum. Future artifacts use robust nonzero counting.
