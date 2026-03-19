# Report: Alpha Optimization on Real OvertPoly Networks
**Date:** 2026-03-18 17:30
**Task:** Benchmark whether alpha optimization (`optimise_alpha`) yields tighter pre-activation bounds compared to CROWN with α=0 on real OvertPoly networks with Hyperrectangle inputs.

---

## Tests Added (Tests 13–16)

New `@testset "Alpha optimisation benchmark on real networks"` block added to `tests/test_alpha_crown.jl`, covering:

| Test | Network | Input Hyperrectangle |
|------|---------|----------------------|
| 13   | Single Pendulum ARCH-COMP | `[1.0,0.0]` to `[1.2,0.2]` (2D) |
| 14   | TORA | `[0.6,-0.7,-0.4,0.5]` to `[0.7,-0.6,-0.3,0.6]` (4D) |
| 15   | Unicycle | `[9.50,-4.50,2.10,1.50]` to `[9.55,-4.45,2.11,1.51]` (4D) |
| 16   | ACC | 5D domain centered on nominal condition |

**For each network:**
1. Base CROWN bounds computed with α=0
2. Alpha optimized with `n_iter=50, lr=0.1` via finite-difference gradient ascent
3. Optimized CROWN bounds computed with resulting α
4. Per-layer statistics: #unstable neurons, mean Δl̂, max Δl̂, any improvement > 0.01
5. Soundness assertion: 500 random samples must lie within `crown_opt` bounds
6. Monotonicity assertion: `crown_opt[k].lower[j] ≥ crown_base[k].lower[j] - 1e-9` for all k, j
7. Upper-bound invariance assertion: upper bounds unchanged by α

---

## Fix to `optimise_alpha` (Bug Corrected)

**File:** `src/nv/alpha_crown.jl`

The original `objective` function in `optimise_alpha` was incorrect. It computed:
```julia
total += alpha[k][j] * l̂[j]   # spurious term: gradient pushes α DOWN
total += sum(l̂)                # double-counts unstable neuron lower bounds
```

The `alpha[k][j] * l̂[j]` term is ≤ 0 (since `l̂[j] < 0` for unstable neurons). Its gradient w.r.t. `alpha[k][j]` is `l̂[j] < 0`, which causes gradient **descent** on α — the opposite of tightening. Additionally, the `sum(l̂)` computed inside the unstable-neuron loop duplicated contributions.

**Fix:** The objective is now simply the sum of all pre-activation lower bounds across all layers:
```julia
function objective(α)
    crown = forward_crown(network, input_set; alpha=α)
    total = 0.0
    for k in 1:n_layers
        total += sum(crown[k].lower)
    end
    return total
end
```
This correctly captures: higher α in layer k tightens post-activation lower bounds there, which propagates into tighter pre-activation lower bounds in layers k+1, k+2, …

---

## Results

### All tests pass: 1934/1934 (Tests 13–16) + 15059/15059 (Tests 1–12)

### Per-network summary (n_iter=50, lr=0.1):

#### Single Pendulum (2-input, 3-layer: 25-ReLU / 25-ReLU / 1-Id)
| Layer | #Unstable (base) | Mean Δl̂ | Max Δl̂ | Meaningful (>0.01)? |
|-------|-----------------|---------|--------|---------------------|
| L1 ReLU | 0 | 0.000000 | 0.000000 | no |
| L2 ReLU | 5 | 0.000000 | 0.000000 | no |
| L3 Id   | 0 | 0.000000 | 0.000000 | no |

Max soundness violation: 0.0

#### TORA (4-input, 5-layer: 25-ReLU / 25-ReLU / 25-ReLU / 1-ReLU / 1-Id)
| Layer | #Unstable (base) | Mean Δl̂ | Max Δl̂ | Meaningful? |
|-------|-----------------|---------|--------|-------------|
| L1 ReLU | 3  | 0.000000 | 0.000000 | no |
| L2 ReLU | 27 | 0.000000 | 0.000000 | no |
| L3 ReLU | 96 | 0.000000 | 0.000000 | no |
| L4 ReLU | 0  | 0.000000 | 0.000000 | no |
| L5 Id   | 1  | 0.000000 | 0.000000 | no |

Max soundness violation: 0.0

#### Unicycle (4-input, 3-layer: 25-ReLU / 25-ReLU / 2-Id)
| Layer | #Unstable (base) | Mean Δl̂ | Max Δl̂ | Meaningful? |
|-------|-----------------|---------|--------|-------------|
| L1 ReLU | 5 | 0.000000 | 0.000000 | no |
| L2 ReLU | 0 | 0.000000 | 0.000000 | no |
| L3 Id   | 0 | 0.000000 | 0.000000 | no |

Max soundness violation: 0.0

#### ACC (5-input, 7-layer: all ReLU except last Id)
| Layer | #Unstable (base) | Mean Δl̂ | Max Δl̂ | Meaningful? |
|-------|-----------------|---------|--------|-------------|
| L1 ReLU | 0  | 0.000000 | 0.000000 | no |
| L2 ReLU | 1  | 0.000000 | 0.000000 | no |
| L3 ReLU | 3  | 0.000000 | 0.000000 | no |
| L4 ReLU | 10 | 0.000000 | 0.000000 | no |
| L5 ReLU | 19 | 0.000000 | 0.000000 | no |
| L6 ReLU | 20 | 0.000000 | 0.000000 | no |
| L7 Id   | 1  | 0.000000 | 0.000000 | no |

Max soundness violation: 0.0

---

## Analysis: Why Alpha Optimization Yields Zero Improvement

The zero improvement is not a bug — it is a provable structural property of CROWN on Hyperrectangle inputs.

**Key insight:** For any unstable neuron j in layer k (l̂_j < 0 < û_j), the post-activation lower bound under CROWN is:
```
z_lower_j = max(0, α_j · l̂_j)
```
Since `α_j ∈ [0, 1]` and `l̂_j < 0`, we have `α_j · l̂_j ≤ 0`, so `max(0, α_j · l̂_j) = 0` for **all** α_j.

This means the post-activation interval for an unstable neuron is always `[0, û_j]` regardless of α. Therefore, changing α has no effect on the post-activation bounds that feed into subsequent layers, and the finite-difference gradient of the objective with respect to any α_{k,j} is identically zero.

**α-CROWN only helps when back-substitution is used:** The actual improvement from α-CROWN (as implemented in the full α-CROWN paper by Xu et al., ICLR 2021) comes from back-substituting the linear relaxation bounds all the way to the input, maintaining linear expressions in x₀ rather than intervals. In that setting, the lower bound of a neuron in a deep layer is a linear function of both x₀ and α, and optimizing α tightens the "linear bound" expression. This requires the full back-substitution formulation (Section 3.2 of Xu et al.), not the interval-propagation forward pass implemented here.

The current `forward_crown` implementation uses interval arithmetic (propagating `[l_post, u_post]` intervals), which is equivalent to MaxSens when inputs are Hyperrectangles. The α parameter controls only the lower slope of the ReLU relaxation, but since the post-activation lower bound is always clamped to 0, α has no observable effect on any downstream bounds.

---

## Conclusion

**Alpha optimization does NOT provide any improvement in pre-activation bounds on these 4 real OvertPoly networks with Hyperrectangle inputs**, regardless of learning rate or iteration count. The improvement Δl̂ = 0 for every neuron in every layer of every network.

**Recommendation for the MILP encoding pipeline:** Do not include `optimise_alpha` in the `BoundedMixedIntegerLP` pipeline. The computational overhead (n_iter × n_neurons × n_layers forward passes per optimization step) is not justified by zero bound improvement.

To realize the theoretical α-CROWN gains, the implementation would need to be extended to:
1. Use the full back-substitution formulation (maintain linear functions of x₀ through the network, not just intervals)
2. Optimize α via exact gradient (Xu et al., ICLR 2021, Section 3.2) rather than finite differences on the interval-propagation objective

This is a significant implementation change and outside the scope of the current interval-arithmetic CROWN implementation.

**Soundness is maintained:** All 1934 assertions in Tests 13–16 pass. The optimised bounds (which happen to equal the base bounds) are verified sound against 500 random samples per network, with zero violations.