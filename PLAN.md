# MaGRIP v2 Framework Plan

This plan converts `magrip_v1.py` from a Gemma-specific pruning script into a reusable framework for Magnitude and Gradient Informed Pruning. The mathematical anchor for the project is `docs/THEORY.tex`.

The design goal is to keep MaGRIP modular: model discovery, FFN topology handling, mask parameterization, saliency collection, optimization, APOLLO integration, compaction, and evaluation should be separable pieces.

## Guiding Principles

- Prune structured FFN intermediate units, not arbitrary scalar weights.
- Only target FFNs inside transformer blocks by default.
- Treat saliency as a state-dependent signal that can be recomputed as weights and masks co-evolve.
- Use joint soft-mask adaptation as the primary training flow, with careful schedules for budget, temperature, and mask update frequency.
- Keep APOLLO as a weight-optimizer backend, not as part of FFN discovery or mask logic.
- Make distillation optional and hardware-aware.
- Prefer small, verifiable milestones over a single large rewrite.

## Target Architecture

```text
magrip/
  __init__.py
  baseline.py
  config.py
  data.py
  discovery.py
  module_utils.py
  topology.py
  masks.py
  saliency.py
  objectives.py
  optim.py
  trainer.py
  evaluation.py
  compaction.py
  logging.py
scripts/
  inspect_model.py
  run_magrip.py
  run_magrip_smoke.py
  run_gpt2_smoke.py
  compact_model.py
tests/
  test_discovery.py
  test_masks.py
  test_saliency.py
  test_objectives.py
  fixtures/
docs/
  THEORY.tex
  V1_BASELINE.md
  FFN_DISCOVERY.md
  APOLLO_INTEGRATION.md
  EXPERIMENTS.md
models/
  Baselines/
  Pruned/
```

## Milestone Checklist

### M0: Project Scaffold

- [x] Create `docs/THEORY.tex` as the mathematical reference.
- [x] Create this project plan and checklist.
- [x] Add Python package skeleton under `magrip/`.
- [x] Add `pyproject.toml` with formatting, linting, and test configuration.
- [x] Add a minimal README describing the project goal and current status.

### M1: Extract and Preserve v1 Behavior

- [x] Copy the useful algorithmic pieces from `magrip_v1.py` into isolated modules.
- [x] Remove notebook-only commands, hardcoded tokens, global constants, and plotting side effects.
- [x] Add gated FFN discovery, shared masks, and branch-averaged saliency.
- [x] Validate Gemma/gated smoke run artifacts.
- [x] Add a GPT-2 smoke test for the dense-FFN baseline.
- [x] Use WikiText-2 validation as the default small calibration dataset for smoke tests.
- [x] Validate GPT-2 dense smoke run artifacts: discovery, saliency, masks, metrics, and logs.
- [x] Record expected v1 assumptions: gated FFN, shared intermediate mask, frozen weights.

M1 is complete. The GPT-2 dense baseline is validated by `outputs/runs/gpt2_smoke_20260710_001150`, and the Gemma gated baseline is validated by `outputs/runs/gpt2_smoke_20260710_121506`. Smoke tests use WikiText-2 validation by default so saliency is estimated from a small dataset rather than a single sentence.

### M2: FFN Discovery and Topology Registry

- [ ] Implement transformer block discovery for common Hugging Face layouts.
- [ ] Restrict prunable search to repeated transformer blocks.
- [ ] Detect dense FFNs, gated FFNs, and branched FFNs from module shapes and names.
- [ ] Detect MoE blocks and skip them with a clear warning in v2.
- [ ] Add `FFNTarget` and `FFNTopology` data structures.
- [ ] Move M1 dense/gated discovery heuristics into an extensible topology registry.
- [ ] Add topology sanity checks for saliency length, mask length, and expected channel count.
- [ ] Add artifact validation checks for dense and gated smoke runs.
- [ ] Write discovery tests using small synthetic transformer blocks.
- [ ] Write model inspection output that explains what MaGRIP will prune before training starts.

### M3: Mask System

- [ ] Implement structured FFN channel masks.
- [ ] Support shared intermediate masks for gated FFNs.
- [ ] Implement soft mask logits, binary hard masks, STE behavior, and temperature schedules.
- [ ] Add parameter/FLOP cost accounting from discovered topology, not hardcoded hidden sizes.
- [ ] Add mask serialization and reload support.
- [ ] Add tests for mask shapes, broadcast behavior, and cost calculation.

### M4: Saliency System

- [ ] Implement activation magnitude saliency.
- [ ] Implement gradient-informed saliency using the first-order proxy in `docs/THEORY.tex`.
- [ ] Add layer-local normalization and optional global ranking.
- [ ] Add saliency recomputation hooks during joint training.
- [ ] Add diagnostics for saliency drift as weights adapt.
- [ ] Add tests that compare mask-gradient saliency with explicit mask gradients on toy modules.

### M5: Objectives and Training Loop

- [ ] Implement task loss wrapper for causal language modeling.
- [ ] Implement budget-aware objective with default `beta = 0`.
- [ ] Add optional distillation modes: disabled, cached logits, top-k logits, hidden states, EMA teacher.
- [ ] Implement joint two-time-scale optimization for weights and mask logits.
- [ ] Add schedules for retained budget `rho_t`, budget pressure `lambda_t`, temperature `tau_t`, and mask update frequency.
- [ ] Add gradient clipping for mask parameters.
- [ ] Add stabilization stage and final weight-only recovery stage.
- [ ] Add checkpointing for model weights, masks, optimizer states, and run config.

### M6: APOLLO Integration

- [ ] Add APOLLO as an optional optimizer backend for model weights.
- [ ] Keep mask optimizer separate from APOLLO.
- [ ] Build APOLLO parameter groups for full-model adaptation.
- [ ] Add configuration for APOLLO rank, scale, projection update gap, and mini mode.
- [ ] Add fallback to AdamW for small CPU/GPU tests.
- [ ] Document memory tradeoffs in `docs/APOLLO_INTEGRATION.md`.

### M7: Structural Compaction

- [ ] Convert final binary masks into physically smaller FFN modules.
- [ ] Compact dense FFNs.
- [ ] Compact gated FFNs by removing aligned gate/up rows and down columns.
- [ ] Verify compacted model logits match masked model logits within tolerance.
- [ ] Save compacted model and tokenizer in Hugging Face format.

### M8: Evaluation and Experiment Tracking

- [ ] Evaluate perplexity before pruning, during soft-mask training, after hardening, and after compaction.
- [ ] Report retained parameters, retained FFN parameters, approximate FLOPs, latency, and memory.
- [ ] Add experiment configs for tiny, small, and target-scale models.
- [ ] Track run artifacts in a predictable output directory.
- [ ] Maintain `docs/EXPERIMENTS.md` with results and lessons learned.

## Implementation Sequence

1. Build the scaffold and configuration layer.
2. Implement FFN discovery with a dry-run model inspection command.
3. Implement masks and cost accounting.
4. Port v1 saliency into topology-aware collectors.
5. Build the default objective and joint training loop without APOLLO.
6. Validate on tiny models and synthetic FFNs.
7. Add APOLLO as the model-weight optimizer backend.
8. Implement compaction and equivalence checks.
9. Run progressively larger experiments.

## Design Decisions To Keep Revisited

- Global budget vs. per-layer budget vs. hybrid budget.
- Whether mask saliency should be based on intermediate activations, mask gradients, or both.
- How often to update masks relative to weights.
- Whether APOLLO should adapt all weights or only FFN-heavy parameter groups.
- Whether distillation is worth the hardware cost for each experiment.
- How to handle MoE architectures after the dense/gated path is stable.

## Definition of Done for v2 Alpha

- A user can run MaGRIP on at least one dense-FFN model and one gated-FFN model without architecture-specific code edits.
- The framework prints discovered FFN targets before pruning.
- The training loop supports joint soft-mask adaptation with configurable schedules.
- The final masks can be hardened and saved.
- A compacted model can be produced for at least gated FFNs.
- A small automated test suite covers discovery, masks, saliency, and objective behavior.
