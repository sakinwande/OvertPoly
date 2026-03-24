# Anderson Cuts Benchmark Report — 2026-03-24-14-24

## Setup
- Network: `controllerTORA.nnet` (ARCH-COMP-2023)
- Domain: Hyperrectangle{Float64, Vector{Float64}, Vector{Float64}}([0.6499999999999999, -0.6499999999999999, -0.35, 0.55], [0.050000000000000044, 0.04999999999999993, 0.04999999999999999, 0.04999999999999993])
- Encoding: CC dynamics + CROWN-bounded network MIP
- Pipeline: full `concreach!` (8 MILP solves: min/max × 4 state vars)
- N_overt: 2, dt: 0.1

## Results

| Condition | Binaries | Constraints | Time (s) | Volume | Speedup |
|-----------|----------|-------------|----------|--------|---------|
| Baseline (no cuts)         |      342 |         566 |     1.73 | 0.00014 |  1.00× |
| Anderson, cap=50           |      342 |         682 |     1.13 | 0.00014 |  1.53× |
| Anderson, cap=200          |      342 |         984 |     1.92 | 0.00014 |  0.90× |

## Reachset bounds

- **Baseline (no cuts)**: low=[0.5299, -0.77399, -0.3501, 0.50157], high=[0.6399, -0.66286, -0.2399, 0.60308]
- **Anderson, cap=50**: low=[0.5299, -0.77399, -0.3501, 0.50157], high=[0.6399, -0.66286, -0.2399, 0.60308]
- **Anderson, cap=200**: low=[0.5299, -0.77399, -0.3501, 0.50157], high=[0.6399, -0.66286, -0.2399, 0.60305]

## Soundness check
All reachset volumes within 0.01% of baseline: false
