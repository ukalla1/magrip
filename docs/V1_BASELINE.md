# MaGRIP v1 Baseline Assumptions

This document records the behavior we want to preserve while refactoring `magrip_v1.py`.

## Preserved Core Behavior

- Start from a pretrained causal language model.
- Keep model weights frozen in the sense that no optimizer step is applied to weights.
- Run calibration text through the model.
- Measure FFN channel saliency from activation magnitude and gradient sensitivity.
- Build structured channel masks from saliency scores.
- Apply binary masks to FFN intermediate activations.
- Evaluate the masked model without permanently modifying model weights.

## M1 Baseline Scope

M1 implements the simplest dense-FFN version of this flow on GPT-2:

- GPT-2 block path: `transformer.h.{i}`.
- Dense FFN path: `transformer.h.{i}.mlp`.
- Expansion module: `c_fc`.
- Contraction module: `c_proj`.
- Mask location: output channels of `c_fc`.
- Default calibration data: `wikitext/wikitext-2-raw-v1`, split `validation`.
- Default smoke calibration size: 8 fixed-length token windows.

This is not yet the full Gemma/gated path from v1. The same structured-mask principle will be extended to gated FFNs by sharing one intermediate mask across branches such as `gate_proj` and `up_proj`, then applying the aligned mask before `down_proj`.

## Explicitly Removed From v1

- Notebook shell commands.
- Hardcoded Hugging Face tokens.
- Hardcoded model names.
- Global device constants.
- Plotting side effects during pruning.
- Gemma-only traversal assumptions.
