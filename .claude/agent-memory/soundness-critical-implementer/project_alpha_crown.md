---
name: alpha-CROWN implementation
description: Status and key decisions for the alpha-CROWN bound propagation feature in OvertPoly
type: project
---

Implemented CROWN / α-CROWN forward bound propagation in `src/nv/alpha_crown.jl` (2026-03-18). All 15,059 tests pass.

**Why:** Replace MaxSens interval arithmetic (which overapproximates input HPolytope as Hyperrectangle immediately) with CROWN, which propagates tighter bounds through multi-layer ReLU networks by retaining linear relaxation structure across layers.

**Key decisions:**
- `CrownBounds` struct holds pre-activation `lower`/`upper` per layer (what `BoundedMixedIntegerLP` needs as big-M values).
- `get_bounds_crown` returns `Vector{Hyperrectangle}` in the same format as `get_bounds` — drop-in replacement.
- `add_controller_constraints!` gained `use_crown::Bool=false` keyword; default preserves backward compatibility.
- α=0 (zero lower-slope) used as default — always sound, no optimisation overhead.
- `optimise_alpha` implements finite-difference gradient ascent on α; optional refinement.
- HPolytope input overapproximated to Hyperrectangle (same as MaxSens) — documented in docstring.

**How to apply:** When suggesting bound improvements to MILP encoding, note that `use_crown=true` is available. For deeper networks (>2 hidden layers), CROWN gains are larger.
