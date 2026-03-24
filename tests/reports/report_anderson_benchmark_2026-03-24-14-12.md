# Anderson Cuts Benchmark Report — 2026-03-24-14-12

## Setup
- Network: `controllerTORA.nnet` (ARCH-COMP-2023)
- Input box: Hyperrectangle{Float64, Vector{Float64}, Vector{Float64}}([0.6499999999999999, -0.6499999999999999, -0.35, 0.55], [0.050000000000000044, 0.04999999999999993, 0.04999999999999999, 0.04999999999999993])
- Bounds: CROWN back-substitution
- Solver: Gurobi, TimeLimit=120s
- Objective: min/max of control output (neurons[end][1])

## Results

| Condition | Cuts | Min obj | Max obj | Min t(s) | Max t(s) | Min nodes | Max nodes |
|-----------|------|---------|---------|----------|----------|-----------|-----------|
| Baseline (no cuts)           |    0 | -0.1787 |  0.2482 |    0.578 |    0.132 |         1 |         1 |
| Anderson, cap=50             |  116 | -0.1787 |  0.2482 |    0.453 |    0.188 |         1 |         1 |
| Anderson, cap=200            |  418 | -0.1787 |  0.2482 |    0.757 |    0.267 |         1 |         1 |
| Anderson, cap=1000           | 2018 | -0.1787 |  0.2482 |    1.551 |    1.602 |         1 |         1 |

## Conclusions

- **Anderson, cap=50**: min speedup 1.27×, max speedup 0.7×
- **Anderson, cap=200**: min speedup 0.76×, max speedup 0.49×
- **Anderson, cap=1000**: min speedup 0.37×, max speedup 0.08×

## Objective consistency check
All conditions should produce the same min/max objective (cuts are valid).
- Min objective consistent: true
- Max objective consistent: true
