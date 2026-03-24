# Report: DLOG Encoding — 2026-03-24-00-00

## Summary

Tests for the DLOG (Disaggregated Logarithmic Convex Combination) encoding implemented in
`src/overtPoly_to_mip.jl`. All 32 assertions passed in 5.1 s.

---

## Tests

### 1. `_dlog_assign_codes`: n=1 edge case

| Item | Value |
|------|-------|
| Input | `xS = [(0,0),(1,0),(0,1)]`, `Tri = [[1,2,3]]` (1 simplex) |
| Expected | `K_total == 0`, `size(codes) == (1, 0)` |
| Actual | `K_total = 0`, `size(codes) = (1, 0)` |
| Result | **PASS** |

Single simplex needs no binary variables.

---

### 2. `_dlog_assign_codes`: n=2 single cell

| Item | Value |
|------|-------|
| Input | Unit square, 2 simplices (`xS_2simplex`, `Tri_2simplex`) |
| Expected | `K_total == 1`, `size(codes) == (2,1)`, codes distinct, all binary |
| Actual | `K_total = 1`, `size(codes) = (2,1)`, codes `[0]` and `[1]` |
| Result | **PASS** |

One bit suffices to distinguish two sub-simplices in a single cell.

---

### 3. `_dlog_assign_codes`: 3×3 grid (8 simplices)

| Item | Value |
|------|-------|
| Input | 9 vertices on `{0,1,2}²`, 8 simplices across 4 cells (`xS_3x3`, `Tri_3x3`) |
| Expected | `K_total == 3`, `size(codes) == (8,3)`, all codes distinct, Gray adjacency |
| Actual | `K_total = 3`, unique codes, correct Gray structure |
| Result | **PASS (7 assertions)** |

Gray adjacency checks verified:
- Simplices in the same cell share dimension bits, differ only in sub-simplex bit.
- Cells adjacent in dim-1 differ in the dim-1 bit but agree in dim-2 bit.
- Cells adjacent in dim-2 differ in the dim-2 bit but agree in dim-1 bit.

---

### 4. `dlogEncoding!` binary variable count

| Item | Value |
|------|-------|
| Input | 3×3 grid, `GraphPolyQuery`, no control (`uCoef=0`), Gurobi |
| Expected | Binary vars = `K_total = 3`, strictly less than `n = 8` |
| Actual | 3 binary variables |
| Result | **PASS** |

Confirms logarithmic reduction: 3 bits vs 8 one-hot binary variables in CC.

---

### 5. `dlogEncoding!` vs `ccEncoding!` — optimal value agreement

| Item | Value |
|------|-------|
| Input | Same 3×3 grid, maximize `y = x₁ + x₂` |
| Expected | Both report `OPTIMAL`, same objective value `4.0`, DLOG uses 3 binaries vs CC's 8 |
| Actual | Both `OPTIMAL`, both `obj = 4.0`, DLOG: 3 bins, CC: 8 bins |
| Result | **PASS (5 assertions)** |

The maximum of `f = x₁ + x₂` over `[0,2]²` is `4.0` at `(2,2)`, achieved by both encodings.

---

### 6. Solution domain validity

| Item | Value |
|------|-------|
| Input | Same maximize problem with DLOG encoding |
| Expected | `x ∈ [0,2]²`, `y ∈ [min(yVals), max(yVals)]` within `1e-6` |
| Actual | `x = [2.0, 2.0]`, `y = 4.0`, all within bounds |
| Result | **PASS (4 assertions)** |

---

## Conclusions

- `_dlog_assign_codes` produces valid, injective, Gray-adjacent codes for all tested grid sizes.
- `dlogEncoding!` (GraphPolyQuery) correctly encodes the piecewise-linear function with logarithmically fewer binary variables: **3 vs 8** for an 8-simplex triangulation.
- DLOG and CC encodings agree exactly on optimal objective value, confirming soundness of the DLOG formulation.
- The DLOG solution lies within the triangulation domain, confirming no spurious relaxation.
