# Anderson Cuts Benchmark — Unicycle — 2026-03-24-14-49

## Setup
- Network: `controllerUnicycle.nnet` (ARCH-COMP-2023)
- Initial domain: Hyperrectangle{Float64, Vector{Float64}, Vector{Float64}}([9.525, -4.475, 2.105, 1.505], [0.025000000000000355, 0.024999999999999467, 0.004999999999999893, 0.0050000000000001155])
- Pipeline: `multi_step_hybreach`, concInt=[5, 5], CC dynamics encoding
- N_overt: 2, dt: 0.2
- Bounds: CROWN back-substitution

## Results

| Condition | Time (s) | Final volume | Speedup |
|-----------|----------|--------------|---------|
| Baseline (no cuts)         |    71.32 |      0.00000 |  1.00× |
| Anderson, cap=50           |    59.78 |      0.00000 |  1.19× |
| Anderson, cap=100          |    50.23 |      0.00000 |  1.42× |
| Anderson, cap=200          |    70.91 |      0.00000 |  1.01× |

## Final reachset bounds

- **Baseline (no cuts)**: low=[5.8791, -2.0526, 2.6139, 2.176], high=[5.9491, -1.9831, 2.6243, 2.2123]
- **Anderson, cap=50**: low=[5.8791, -2.0526, 2.6139, 2.176], high=[5.9492, -1.9831, 2.6243, 2.2124]
- **Anderson, cap=100**: low=[5.8791, -2.0527, 2.6139, 2.176], high=[5.9492, -1.9832, 2.6243, 2.2124]
- **Anderson, cap=200**: low=[5.8791, -2.0527, 2.6139, 2.176], high=[5.9492, -1.9831, 2.6243, 2.2124]

## Soundness check
All final volumes within 0.1% of baseline: false
