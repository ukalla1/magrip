# Saliency System

M4 implements the saliency score derived in `docs/THEORY.tex`.

## Measurement Point

The primary MaGRIP saliency signal is measured at the tensor entering the FFN contraction
projection:

- dense FFN: the input to `c_proj`, `fc_out`, or equivalent;
- gated FFN: the input to `down_proj`, i.e. the gated intermediate product.

This tensor is the structured intermediate `u_{\ell,i}` used in the first-order pruning
derivation:

```text
s_grad[l, i] = | < dL / du[l, i], u[l, i] > |
```

Expansion branch activations are still collected as diagnostics when enabled, but they are
not the primary pruning score.

## Score Components

For each target and channel, M4 computes:

- magnitude saliency: `||u_i||_2` over batch/sequence dimensions;
- gradient saliency: `|sum(u_i * dL/du_i)|` over batch/sequence dimensions;
- combined saliency: weighted magnitude plus weighted gradient.

Layer-median normalization is the default, matching the theory document. Global median and
no-normalization modes are available for later experiments.

## Recomputing Saliency

`SaliencyTracker` and `SaliencyRefreshSchedule` provide the small amount of state needed by
the future joint training loop:

- decide whether to recompute saliency at a step;
- keep the latest saliency result;
- report relative-L2 and cosine-distance drift from the previous saliency state.

This is intentionally independent from APOLLO and the optimizer stack.

## Tests

`tests/test_saliency.py` covers:

- the first-order saliency and explicit mask-gradient identity;
- contract-input saliency shape on a toy FFN;
- branch diagnostics;
- saliency recomputation scheduling and drift reporting.
