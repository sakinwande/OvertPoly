---
name: alpha-CROWN implementation
description: Status and key decisions for the alpha-CROWN bound propagation feature in OvertPoly
type: project
---

Implemented CROWN / α-CROWN forward bound propagation in `src/nv/alpha_crown.jl` (2026-03-18). All 16,993 tests pass (15,059 original + 1,934 new benchmark tests).

**Why:** Replace MaxSens interval arithmetic (which overapproximates input HPolytope as Hyperrectangle immediately) with CROWN, which propagates tighter bounds through multi-layer ReLU networks by retaining linear relaxation structure across layers.

**Key decisions:**
- `CrownBounds` struct holds pre-activation `lower`/`upper` per layer (what `BoundedMixedIntegerLP` needs as big-M values).
- `get_bounds_crown` returns `Vector{Hyperrectangle}` in the same format as `get_bounds` — drop-in replacement.
- `add_controller_constraints!` gained `use_crown::Bool=false` keyword; default preserves backward compatibility.
- α=0 (zero lower-slope) used as default — always sound, no optimisation overhead.
- `optimise_alpha` implements finite-difference gradient ascent on α; optional refinement.
- HPolytope input overapproximated to Hyperrectangle (same as MaxSens) — documented in docstring.

**Alpha optimization finding (2026-03-18):** `optimise_alpha` yields ZERO improvement on all 4 real networks (Single Pendulum, TORA, Unicycle, ACC) with Hyperrectangle inputs. This is provable: post-activation lower bound of any unstable neuron is max(0, α·l̂) = 0 for all α ∈ [0,1], so α has no effect on downstream interval bounds. The full back-substitution formulation (Xu et al. ICLR 2021, Section 3.2) — maintaining linear functions of x₀ rather than intervals — would be needed to realize gains. Do NOT include `optimise_alpha` in the MILP pipeline.

**Bug fixed in `optimise_alpha`:** The original objective contained a spurious `alpha[k][j] * l̂[j]` term whose gradient (= l̂[j] < 0) pushed α in the wrong direction. Fixed to: `total += sum(crown[k].lower)` for all layers. With this fix the objective gradient is correctly zero (confirming the structural no-gain result).

**How to apply:** When suggesting bound improvements to MILP encoding, note that `use_crown=true` is available. For deeper networks (>2 hidden layers), CROWN (α=0) equals MaxSens on Hyperrectangle inputs. α optimization is not worth the overhead.

---

## CROWN back-substitution (2026-03-18)

Implemented `forward_crown_backsub` and `get_bounds_crown_backsub` in `src/nv/alpha_crown.jl`. These maintain affine functions of x₀ across all layers rather than collapsing to scalar intervals at each ReLU, yielding dramatically tighter bounds for hidden layers 2+.

**Key results on real networks (typical improvement in mean bound width):**
- Single Pendulum L2/L3: ~52% tighter
- TORA L2: 79%, L3: 84% tighter
- Unicycle L2/L3: 90% tighter
- ACC L3: 53%, L4: 75%, L5: 61% tighter

**Soundness:** Zero violations across all sampling tests. Upper bounds use Planet relaxation; lower bounds use α_j·ẑ_j with α_j ∈ [0,1].

**Layer 1 always identical to interval CROWN:** The first layer is a direct linear map of x₀; both methods evaluate the same affine map over the input box.

**Alpha optimisation on top of back-sub:** Zero improvement (same structural reason as interval CROWN — `optimise_alpha` does not propagate α sensitivity through back-sub).

**Important implementation note:** `dot` from LinearAlgebra is NOT imported in the codebase. Use `sum(a .* b)` for inner products instead.

**All 19,235 tests pass** (tests 17–20 are the new back-sub tests, adding 3,242 assertions).
