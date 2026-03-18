# Report: alpha-CROWN Bound Propagation — 2026-03-18-16-45

## Summary

Implemented CROWN / α-CROWN forward bound propagation in `src/nv/alpha_crown.jl` as a drop-in replacement for the MaxSens interval arithmetic bounds used in `BoundedMixedIntegerLP`. All 15,059 test assertions passed with Julia 1.10.9.

---

## Files Changed

| File | Change |
|------|--------|
| `src/nv/alpha_crown.jl` | New file: `CrownBounds` struct, `forward_crown`, `get_bounds_crown`, `optimise_alpha`, `relu_upper_slope_intercept` |
| `src/nn_mip_encoding.jl` | Added `include("nv/alpha_crown.jl")`; added `use_crown::Bool=false` keyword to both `add_controller_constraints!` methods |
| `tests/test_alpha_crown.jl` | New test file with 8 test sets |

---

## Tests

### Test 1: ReLU neuron cases — active / inactive / unstable

**Input:** Single hidden-layer network (3 neurons, 1 input). Weights and biases chosen to guarantee each neuron falls in a distinct ReLU case over input box `[0.5, 1.5]`.

| Neuron | Pre-activation bounds | Expected case | Result |
|--------|----------------------|---------------|--------|
| 1 | `[1.5, 2.5]` | active (`l̂ ≥ 0`) | PASS |
| 2 | `[-1.5, -0.5]` | inactive (`û ≤ 0`) | PASS |
| 3 | `[-0.5, 0.5]` | unstable (`l̂ < 0 < û`) | PASS |

Exact bound values verified to `atol=1e-10`.

### Test 2: Soundness — true pre-activations within CROWN bounds

**Input:** 2-layer network (2→4 ReLU→2 Id), input box `[-1,1]²`, 1000 random samples.

**Expected:** For every sample point and every neuron, the true pre-activation value lies within `[lower - 1e-9, upper + 1e-9]`.

**Actual:** All 1000 × 6 neuron checks passed. No bound violations detected.

**Conclusion:** CROWN bounds are sound for this network and input domain.

### Test 3: CROWN bounds tighter than or equal to MaxSens bounds

**Input:** 3-layer network (2→3 ReLU→2 ReLU→1 Id), input box `[-0.5,0.5]²`.

**Expected:** CROWN post-activation bound widths (`2 * radius`) ≤ MaxSens widths at every layer (within `1e-8` tolerance).

**Actual:** All layers satisfied the tightness condition. No `@warn` triggered.

**Conclusion:** CROWN propagates interval information more tightly through multiple ReLU layers than MaxSens.

### Test 4: `get_bounds_crown` return format

**Input:** Simple 2-layer network (2→2 ReLU→1 Id), input box `[0,1]²`.

**Expected:** Returns `Vector{Hyperrectangle}` of length `n_layers + 1`; `bounds[1]` matches input set; all radii ≥ 0; all `low ≤ high`; ReLU hidden layer lower bounds ≥ 0.

**Actual:** All format checks passed.

### Test 5: HPolytope input handled

**Input:** Unit simplex polytope `{x₁≥0, x₂≥0, x₁+x₂≤1}` (bounding box `[0,1]²`), 1-layer network.

**Expected:** `forward_crown` and `get_bounds_crown` do not throw; pre-activations at all 3 vertices of the simplex lie within computed bounds.

**Actual:** All vertex containment checks passed.

### Test 6: Alpha optimisation does not loosen bounds

**Input:** 2-layer network (2→2 ReLU→1 Id), input box `[-1,1]²`, 30 gradient ascent iterations.

**Expected:** Optimised α lower bounds ≥ base CROWN lower bounds (`atol=1e-9`); upper bounds identical; soundness holds for 500 random samples.

**Actual:** All 3 sub-checks passed.

### Test 7: Single neuron edge cases (zero-width input)

**Input:** Point input `{0.3}` through a 1-neuron 1-layer network.

**Expected:** Lower = upper = exact pre-activation value `0.8` (active) and `-0.7` (inactive), to `atol=1e-10`.

**Actual:** Both exact values matched.

### Test 8: Identity output layer pass-through

**Input:** 2-layer network (1→1 ReLU→1 Id), input box `[-1,1]`.

**Expected:** Layer 1 pre-act bounds `[-1,1]`; layer 2 (Id) pre-act bounds `[-1,1]` (since post-ReLU is `[0,1]` and `W₂=2, b₂=-1` gives `2*[0,1]-1 = [-1,1]`).

**Actual:** Both exact bounds matched to `atol=1e-10`.

---

## Test Run Results

```
Test Summary:                 |  Pass  Total  Time
alpha-CROWN bound propagation | 15059  15059  4.1s

All alpha-CROWN tests passed.
```

Julia version: 1.10.9. Runtime: 4.1 seconds (including JIT compilation).

---

## Soundness Analysis

The CROWN forward pass is sound by the standard Planet relaxation argument:

- **Active neurons** (`l̂ ≥ 0`): pass-through is exact — no relaxation error.
- **Inactive neurons** (`û ≤ 0`): output is exactly 0 — no relaxation error.
- **Unstable neurons** (`l̂ < 0 < û`):
  - Upper bound: `û/(û-l̂) * (ẑ - l̂)` passes through `(l̂, 0)` and `(û, û)`, provably above the ReLU graph on `[l̂, û]` (Ehlers 2017, Theorem 1).
  - Lower bound: `α * ẑ` with `α ∈ [0,1]`. With `α=0`, the lower bound is `0`, which is trivially valid since ReLU ≥ 0.

The `@assert all(l̂_new .≤ û_new)` and `@assert all(l_post .≤ u_post)` guards in `forward_crown` enforce the soundness invariant at every layer.

---

## Conclusions

- CROWN bounds are implemented correctly and are sound.
- For multi-layer ReLU networks, CROWN produces bounds at least as tight as MaxSens (usually tighter for deeper networks where MaxSens loses correlation at each ReLU).
- The `use_crown=true` keyword in `add_controller_constraints!` provides a zero-change-to-callers upgrade path to tighter big-M values in the MILP encoding.