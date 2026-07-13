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
  MASK_SYSTEM.md
  SALIENCY_SYSTEM.md
  TRAINING_LOOP.md
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

- [x] Implement transformer block discovery for common Hugging Face layouts.
- [x] Restrict prunable search to repeated transformer blocks.
- [x] Detect dense FFNs, gated FFNs, and branched FFNs from module shapes and names.
- [x] Detect MoE blocks and skip them with a clear warning in v2.
- [x] Add `FFNTarget` and `FFNTopology` data structures.
- [x] Move M1 dense/gated discovery heuristics into an extensible topology registry.
- [x] Add topology sanity checks for saliency length, mask length, and expected channel count.
- [x] Add artifact validation checks for dense and gated smoke runs.
- [x] Write discovery tests using small synthetic transformer blocks.
- [x] Write model inspection output that explains what MaGRIP will prune before training starts.

M2 is complete. Discovery is now registry-backed, restricted to known repeated transformer block stacks, and reports skipped MoE-like FFNs. The inspection and artifact-validation scripts provide the main server-side checks before pruning.

### M3: Mask System

- [x] Implement structured FFN channel masks.
- [x] Support shared intermediate masks for gated FFNs.
- [x] Implement soft mask logits, binary hard masks, STE behavior, and temperature schedules.
- [x] Add parameter/FLOP cost accounting from discovered topology, not hardcoded hidden sizes.
- [x] Add mask serialization and reload support.
- [x] Add tests for mask shapes, broadcast behavior, and cost calculation.
- [x] Inspect dense and gated M3 smoke results for technical correctness.

M3 is complete. The smoke path still preserves M1 frozen-mask behavior, but masks are now represented by topology-aware `StructuredMask` objects with logits, temperatures, STE-compatible hard values, serialization, and model-derived FFN channel costs.

### M4: Saliency System

- [x] Implement activation magnitude saliency.
- [x] Implement gradient-informed saliency using the first-order proxy in `docs/THEORY.tex`.
- [x] Add layer-local normalization and optional global ranking.
- [x] Add saliency recomputation hooks during joint training.
- [x] Add diagnostics for saliency drift as weights adapt.
- [x] Add tests that compare mask-gradient saliency with explicit mask gradients on toy modules.
- [x] Inspect saliency-system results for technical correctness on dense and gated smoke runs.

M4 implementation is complete. The primary saliency signal is now collected at the FFN
contraction input, matching the theory-level intermediate `u`; branch-level expansion
signals remain available as diagnostics. Dense and gated smoke artifacts have been
inspected for source metadata, channel consistency, branch diagnostics, retained budget
accounting, and loss/perplexity behavior.

### M5: Objectives and Training Loop

- [x] Implement task loss wrapper for causal language modeling.
- [x] Implement budget-aware objective with default `beta = 0`.
- [x] Add optional distillation scaffold: disabled by default plus cached-logit support.
- [x] Implement joint two-time-scale optimization for weights and mask logits.
- [x] Add schedules for retained budget `rho_t`, budget pressure `lambda_t`, temperature `tau_t`, and mask update frequency.
- [x] Add gradient clipping for mask parameters.
- [x] Add stabilization stage and final weight-only recovery stage.
- [x] Add checkpointing for model weights, masks, optimizer states, and run config.
- [x] Inspect objective/training-loop results for technical correctness on dense and gated smoke runs.

M5 implementation is complete. The trainer defaults to mask-only adaptation, supports
optional AdamW weight updates, keeps APOLLO reserved for M6, and records objective,
budget, mask-gradient, retained-cost, checkpoint, and saliency-drift signals. M5 result
inspection passed on GPT-2 dense and Gemma gated smoke runs. The quick runs are
technically coherent, but longer runs should use stronger budget pressure because relaxed
mask probabilities can drift below the final hard-mask target before hardening. Follow-up
M5 tuning added budget-calibrated saliency-logit initialization so the relaxed objective
starts near the target retained-cost ratio.

### M6: APOLLO Integration

- [ ] Add APOLLO as an optional optimizer backend for model weights.
- [ ] Keep mask optimizer separate from APOLLO.
- [ ] Build APOLLO parameter groups for full-model adaptation.
- [ ] Add configuration for APOLLO rank, scale, projection update gap, and mini mode.
- [ ] Add fallback to AdamW for small CPU/GPU tests.
- [ ] Document memory tradeoffs in `docs/APOLLO_INTEGRATION.md`.
- [ ] Inspect APOLLO integration results for technical correctness against the AdamW fallback.

### M7: Structural Compaction

- [ ] Convert final binary masks into physically smaller FFN modules.
- [ ] Compact dense FFNs.
- [ ] Compact gated FFNs by removing aligned gate/up rows and down columns.
- [ ] Verify compacted model logits match masked model logits within tolerance.
- [ ] Save compacted model and tokenizer in Hugging Face format.
- [ ] Inspect compacted dense and gated model artifacts for technical correctness.

### M8: Evaluation and Experiment Tracking

- [ ] Evaluate perplexity before pruning, during soft-mask training, after hardening, and after compaction.
- [ ] Report retained parameters, retained FFN parameters, approximate FLOPs, latency, and memory.
- [ ] Add experiment configs for tiny, small, and target-scale models.
- [ ] Track run artifacts in a predictable output directory.
- [ ] Maintain `docs/EXPERIMENTS.md` with results and lessons learned.
- [ ] Inspect evaluation and experiment-tracking outputs for technical correctness.

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
