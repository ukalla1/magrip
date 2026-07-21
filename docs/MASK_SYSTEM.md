# Mask System

M3 implements the structured FFN mask contract from `docs/THEORY.tex`.

## Mask Unit

Each `StructuredMask` has one scalar per discovered FFN intermediate channel:

- dense FFN: one mask value for each post-activation channel entering the contraction
  projection;
- gated FFN: one shared mask value for each post-gating product channel entering
  `down_proj`.

The mask length is always taken from `FFNTarget.intermediate_size`; it is not hardcoded by
architecture.

## Relaxation

Trainable masks use logits `phi` and temperature `tau`:

```text
q = sigmoid(phi / tau)
z = 1{q >= kappa}
```

When `ste=True`, the forward pass uses hard binary values and the backward pass routes
gradients through the probability path. Frozen M1-style masks are still represented as
`StructuredMask` objects, but with non-trainable logits initialized from exact binary values.

## Cost Accounting

Per-channel parameter and per-token FFN FLOP proxy costs are inferred from the discovered
projection modules:

- expansion modules contribute one output-channel weight slice;
- expansion biases count toward parameter cost when present;
- contraction modules contribute one input-channel weight slice;
- gated FFNs sum the gate, up, and down contributions for the shared unit.

This gives MaGRIP model-derived estimates of retained FFN parameter and compute cost:

```text
Cost(z) = sum_l sum_i c_{l,i} z_{l,i}
```

## Smoke-Test Artifacts

Smoke runs now save:

- `masks.pt`: simple binary tensors for quick inspection;
- `mask_state.pt`: reloadable structured masks with target metadata, logits, thresholds,
  temperatures, and channel costs;
- `summary.json`: per-mask and aggregate retained parameter/FLOP cost metrics.

## Tests

`tests/test_masks.py` covers:

- exact top-k mask creation from saliency;
- dense and gated shape-derived parameter/FLOP cost accounting;
- contraction-input mask application for dense and gated FFNs;
- STE gradient flow to mask logits;
- temperature schedule updates;
- mask serialization round trip.

Saved smoke artifacts can be inspected with:

```bash
python scripts/inspect_mask_state.py models/Pruned/gpt2_magrip_smoke --strict
python scripts/inspect_mask_state.py models/Pruned/google__gemma-2b_magrip_smoke --strict
```

The inspector verifies that `mask_state.pt`, `masks.pt`, and `manifest.json` agree on
mask keys, binary values, active channel counts, and aggregate parameter/FLOP costs.
