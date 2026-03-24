# Anderson Cuts Test Report — 2026-03-24-14-06

## Summary

All 18 tests passed in 5.5s.

---

## Test 1: `_conditional_bound_min` Analytical LP (6 cases)

**What:** Verifies the analytical parametric LP solver (box + 1 linear constraint) returns correct optimal values on hand-checked examples.

| Case | Problem | Expected | Actual | Pass |
|------|---------|----------|--------|------|
| A | min z₁+z₂ s.t. z₁≥0, z∈[-1,1]² | -1.0 | -1.0 | ✓ |
| B | min z₁+z₂ s.t. z₁+z₂≥1, z∈[-1,1]² | 1.0 | 1.0 | ✓ |
| C | min z₁-z₂ s.t. z₁-z₂≥0, z∈[-1,1]² | 0.0 | 0.0 | ✓ |
| D | infeasible: z₁≥5, z∈[-1,1] | NaN | NaN | ✓ |
| E | constraint always satisfied | -1.0 | -1.0 | ✓ |
| F | max via negation | 2.0 | 2.0 | ✓ |

**Conclusion:** Analytical LP solver is correct on all cases including infeasibility detection.

---

## Test 2: `conditional_preact_bounds` Soundness (500 random samples × 3 pairs)

**What:** For a 3-neuron ReLU layer with 2D input, samples 500 random inputs and verifies that for all inputs where the conditioning event holds (ẑ_i ≥ 0 or ẑ_i ≤ 0), the true pre-activation ẑ_j lies within the computed conditional bounds.

**Network:** W = [[1,1],[-1,1],[1,-1]], b = [-0.3,-0.3,-0.3], input ∈ [-1,1]²

**Pairs tested:** (1,2), (1,3), (2,3)

**Result:** 0 violations across all pairs and all 500 samples. Bounds are sound.

---

## Test 3: `add_anderson_cuts!` Synthetic Network

**What:** Builds a 2-layer network (3 unstable neurons in layer 1), encodes with BoundedMixedIntegerLP, adds Anderson cuts, and verifies the model remains feasible.

**Inputs:** W1 = [[1,1],[-1,1],[1,0]], b1 = [-0.3,-0.3,-0.3], input ∈ [-0.5,0.5]²

**Checks:**
- `n_cuts ≥ 0` ✓
- `n_fixed ≥ 0` ✓
- Constraint count does not decrease ✓
- Model remains feasible (Min 0, status OPTIMAL) ✓

---

## Test 4: No Cuts for Stable Network

**What:** Verifies that no cuts are generated when all neurons are stably active (l̂ ≥ 0 everywhere).

**Result:** `n_cuts = 0`, `n_fixed = 0` ✓

---

## Test 5: TORA Network Integration

**What:** Runs `add_anderson_cuts!` on the ARCH-COMP TORA network (4D input, 3 ReLU layers) with `max_cuts_per_layer=200`. Verifies that cuts are generated and the model is still feasible.

**Network:** `controllerTORA.nnet`, input ∈ [-0.77,0.77]×[-0.45,0.45]³
**Bounds:** CROWN back-substitution

**Result:** 600 cuts added, 0 binaries fixed. Model feasible (Min 0, OPTIMAL) ✓

**Note on scale:** The uncapped run (first attempt) generated 42,394 cuts for TORA. This made the MILP intractable. The `max_cuts_per_layer` cap is essential for practical use. Benchmarking the LP relaxation quality vs. cut count is future work.

---

## Issues Found and Fixed

1. **Double-counting:** Original implementation iterated over ordered pairs (i,j) AND (j,i) separately, generating ~2× the cuts. Fixed to iterate unordered pairs `{i,j}` and handle both directions per pair.
2. **JuMP performance warning:** Rebuilding `ẑ_j_expr` inside the inner O(n²) loop triggered JuMP's addition operator warning. Fixed by precomputing `AffExpr` per neuron using `add_to_expression!`.
3. **Unbounded cut generation:** 42k cuts on TORA made the MILP unsolvable in reasonable time. Added `max_cuts_per_layer` cap (default unlimited, set to 200 in integration test).