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
- Mask location: post-activation intermediate channels entering `c_proj`.
- Default calibration data: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `validation`.
- Default smoke calibration size: 8 fixed-length token windows.

The same structured-mask principle is used for gated FFNs by sharing one intermediate mask across branches such as `gate_proj` and `up_proj`, then applying the aligned mask before `down_proj`.

## Gated FFN Compatibility Scope

The M1 gated path targets Gemma/LLaMA/Qwen-style FFNs:

- Decoder block path: usually `model.layers.{i}`.
- Gated FFN path: `model.layers.{i}.mlp`.
- Expansion branches: `gate_proj` and `up_proj`.
- Contraction module: `down_proj`.
- Mask location: shared post-gating intermediate channels entering `down_proj`.
- Saliency aggregation: branch-averaged activation magnitude and gradient sensitivity.

This preserves the v1 Gemma idea of one shared intermediate mask across the gated FFN branches.

## M1 Validation Artifacts

- Dense GPT-2 smoke run: `outputs/runs/gpt2_smoke_20260710_001150`.
- Gated Gemma smoke run: `outputs/runs/gpt2_smoke_20260710_121506`.

Both runs use WikiText-2 validation, 8 calibration windows, frozen weights, saliency-derived masks, and temporary masked evaluation.

## Explicitly Removed From v1

- Notebook shell commands.
- Hardcoded Hugging Face tokens.
- Hardcoded model names.
- Global device constants.
- Plotting side effects during pruning.
- Gemma-only traversal assumptions.
