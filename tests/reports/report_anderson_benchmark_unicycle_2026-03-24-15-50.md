# Anderson Cuts Benchmark — Unicycle — 2026-03-24-15-50

## Setup
- Network: `controllerUnicycle.nnet` (ARCH-COMP-2023)
- Initial domain: Hyperrectangle{Float64, Vector{Float64}, Vector{Float64}}([9.525, -4.475, 2.105, 1.505], [0.025000000000000355, 0.024999999999999467, 0.004999999999999893, 0.0050000000000001155])
- Pipeline: `multi_step_hybreach`, concInt=[10, 10, 5], CC dynamics encoding
- N_overt: 2, dt: 0.2
- Bounds: CROWN back-substitution

## Results

| Condition | Time (s) | Final volume | Speedup |
|-----------|----------|--------------|---------|
| Baseline (no cuts)         |   638.97 |      0.00002 |  1.00× |
| Anderson, cap=50           |   663.09 |      0.00002 |  0.96× |
| Anderson, cap=100          |   707.45 |      0.00002 |  0.90× |
| Anderson, cap=200          |   653.13 |      0.00002 |  0.98× |

## Final reachset bounds

- **Baseline (no cuts)**: low=[3.8229, -0.3365, -0.0484, -0.7626], high=[3.9861, -0.2399, -0.0267, -0.7057]
- **Anderson, cap=50**: low=[3.8228, -0.3365, -0.0484, -0.7626], high=[3.9861, -0.2399, -0.0267, -0.7057]
- **Anderson, cap=100**: low=[3.8227, -0.3365, -0.0484, -0.7627], high=[3.9861, -0.2398, -0.0267, -0.7056]
- **Anderson, cap=200**: low=[3.8229, -0.3365, -0.0484, -0.7626], high=[3.9857, -0.2398, -0.0267, -0.7057]

## Soundness check
All final volumes within 0.1% of baseline: false
