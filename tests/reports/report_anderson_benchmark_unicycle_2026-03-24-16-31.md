# Anderson Cuts Benchmark — Unicycle — 2026-03-24-16-31

## Setup
- Network: `controllerUnicycle.nnet` (ARCH-COMP-2023)
- Initial domain: Hyperrectangle{Float64, Vector{Float64}, Vector{Float64}}([9.525, -4.475, 2.105, 1.505], [0.025000000000000355, 0.024999999999999467, 0.004999999999999893, 0.0050000000000001155])
- Pipeline: `multi_step_hybreach`, concInt=[10, 10, 5], CC dynamics encoding
- N_overt: 2, dt: 0.2
- Bounds: CROWN back-substitution

## Results

| Condition | Time (s) | Final volume | Speedup |
|-----------|----------|--------------|---------|
| Baseline (no cuts)         |   642.04 |      0.00002 |  1.00× |
| Anderson, cap=500          |   651.05 |      0.00002 |  0.99× |
| Anderson, cap=1000         |   714.87 |      0.00002 |  0.90× |

## Final reachset bounds

- **Baseline (no cuts)**: low=[3.8229, -0.3365, -0.0484, -0.7626], high=[3.9861, -0.2399, -0.0267, -0.7057]
- **Anderson, cap=500**: low=[3.8225, -0.3365, -0.0484, -0.7626], high=[3.9861, -0.2398, -0.0267, -0.7056]
- **Anderson, cap=1000**: low=[3.8225, -0.3365, -0.0484, -0.7626], high=[3.986, -0.2398, -0.0267, -0.7056]

## Soundness check
All final volumes within 0.1% of baseline: false
