# Anderson Cuts Benchmark Report — 2026-03-24-14-35

## Setup
- Network: `controllerTORA.nnet` (ARCH-COMP-2023)
- Initial domain: Hyperrectangle{Float64, Vector{Float64}, Vector{Float64}}([0.6499999999999999, -0.6499999999999999, -0.35, 0.55], [0.050000000000000044, 0.04999999999999993, 0.04999999999999999, 0.04999999999999993])
- Pipeline: `multi_step_concreach`, 10 steps, CC dynamics encoding
- N_overt: 2, dt: 0.1
- Bounds: CROWN back-substitution

## Results

| Condition | Total (s) | Per step (s) | Final volume | Speedup |
|-----------|-----------|--------------|--------------|---------|
| Baseline (no cuts)         |     38.29 |         3.83 |      0.00465 |  1.00× |
| Anderson, cap=50           |     43.39 |         4.34 |      0.00465 |  0.88× |
| Anderson, cap=200          |     67.54 |         6.75 |      0.00465 |  0.57× |

## Final reachset bounds

- **Baseline (no cuts)**: low=[-0.3452, -1.0921, -0.0064, 0.1511], high=[-0.079, -0.8141, 0.2312, 0.4156]
- **Anderson, cap=50**: low=[-0.3452, -1.0921, -0.0064, 0.151], high=[-0.079, -0.8141, 0.2312, 0.4155]
- **Anderson, cap=200**: low=[-0.3452, -1.0921, -0.0064, 0.151], high=[-0.079, -0.8141, 0.2312, 0.4156]

## Soundness check
All final volumes within 0.1% of baseline: true
