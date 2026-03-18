---
name: OvertPoly codebase structure
description: Key types, modules, and conventions in the OvertPoly Julia project
type: project
---

OvertPoly is a Julia reachability analysis tool that encodes neural network controllers as MILPs.

**Key nv/ types:**
- `Network` — `Vector{Layer}` (includes output layer)
- `Layer{F,N}` — `weights::Matrix{N}`, `bias::Vector{N}`, `activation::F`
- `ReLU`, `Id`, `GeneralAct`, `PiecewiseLinear` — activation function types
- `Hyperrectangle`, `HPolytope` — from LazySets
- `BoundedMixedIntegerLP` — main MILP encoding (Tjeng et al.); needs per-neuron bounds as big-M

**Key utility functions (src/nv/util.jl):**
- `get_bounds(network, input::Hyperrectangle)` — MaxSens post-activation bounds, returns `Vector{Hyperrectangle}` of length `n_layers+1`; `bounds[1]` is input set
- `approximate_affine_map(layer, bounds::Hyperrectangle)` — interval affine map
- `interval_map(W, l, u)` — core interval arithmetic: `l_new = max(W,0)*l + min(W,0)*u`
- `affine_map(layer, input::AbstractPolytope)` — exact affine map via LazySets

**src/nn_mip_encoding.jl** — entry point; includes all nv/ files and alpha_crown.jl; `add_controller_constraints!` builds the full MILP model

**Why:** MaxSens overapproximates input HPolytope as Hyperrectangle immediately and uses pure interval arithmetic — loose bounds. CROWN propagates tighter linear relaxation bounds through multi-layer networks.

**Test convention:** Never run tests directly in bash. Add to test script and run that script. Reports go in `tests/reports/report_{task}_{yyyy-mm-dd-hh-mm}.md`.

**Julia binary:** `/home/sam/.julia/juliaup/julia-1.10.9+0.x64.linux.gnu/bin/julia` (julia not on PATH)
