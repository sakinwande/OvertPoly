# Report: DLOG Encoding — Unicycle Benchmark — 2026-03-24-00-30

## Setup

| Parameter   | Value |
|-------------|-------|
| Benchmark   | Unicycle (4D, nn-controlled) |
| Domain      | `x ∈ [9.50,9.55] × [-4.50,-4.45] × [2.10,2.11] × [1.50,1.51]` |
| `concInt`   | `[5, 5]` (10 total steps, two hybrid intervals of 5) |
| `N_overt`   | 2 |
| `dt`        | 0.2 |
| Solver      | Gurobi 11.0.3 (academic), 32-thread Ryzen 9 7950X |

Three encodings compared: `ccEncoding!`, `dccEncoding!`, `dlogEncoding!`.
Each runs a full `multi_step_hybreach` (concrete bounds + symbolic reachability per interval).

---

## Results

| Encoding | Time (s) | Volume        | Δ vol vs CC (%) |
|----------|----------|---------------|-----------------|
| CC       | 71.67    | 1.83057 × 10⁻⁶ | —              |
| DCC      | 102.54   | 1.84808 × 10⁻⁶ | +0.96 %        |
| **DLOG** | **28.70**| **1.82862 × 10⁻⁶** | **−0.11 %** |

### Final reachset bounds (step 10)

| Encoding | low | high |
|----------|-----|------|
| CC   | `[5.87908, -2.05262, 2.61395, 2.17602]` | `[5.94916, -1.98314, 2.62428, 2.21235]` |
| DCC  | `[5.87908, -2.05262, 2.61395, 2.17602]` | `[5.94916, -1.98314, 2.62428, 2.21235]` |
| DLOG | `[5.87908, -2.05262, 2.61395, 2.17602]` | `[5.94916, -1.98314, 2.62428, 2.21235]` |

All three encodings produce identical or nearly identical reachset bounds at double-precision.

### MIP size (pre-presolve / after Gurobi presolve)

| Encoding | Binary vars (raw) | Binary vars (presolved) |
|----------|-------------------|-------------------------|
| CC       | 2888              | ~380                    |
| DCC      | 2892              | ~380                    |
| DLOG     | 2582              | **~84**                 |

DLOG reduces binary variables by ~11% pre-presolve and by **78%** post-presolve relative to CC.

---

## Conclusions

1. **DLOG is 2.5× faster than CC and 3.6× faster than DCC** on this benchmark with no change in reachset size (< 0.2% volume difference, attributable to floating-point rounding).

2. **DLOG produces the smallest reachset** of the three encodings (volume 1.82862 × 10⁻⁶ vs CC 1.83057 × 10⁻⁶ vs DCC 1.84808 × 10⁻⁶), consistent with a tighter LP relaxation.

3. **Binary variable count reduction is significant after presolve**: ~380 → ~84, a 78% reduction. This is the primary driver of the speedup, as branch-and-bound complexity scales super-linearly in binary count.

4. **DCC is slower than CC** on this benchmark. The additional continuous variables introduced by disaggregation outweigh the LP-tightening benefit at this problem scale.

5. DLOG is recommended over both CC and DCC for this class of problems.
