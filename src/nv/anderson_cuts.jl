"""
    anderson_cuts.jl — Pairwise conditional bound cuts for ReLU MILP encoding.

For each pair (i, j) of unstable neurons in the same ReLU layer, we compute
tighter bounds on ẑ_j = W[j,:] · z_prev + b[j] conditional on the activation
status of neuron i (δ_i ∈ {0,1}).  These yield valid linear cuts that tighten
the LP relaxation without adding new binary variables.

### Method

For neurons i, j in layer k with shared input z_prev ∈ [l_prev, u_prev]:

  When δ_i = 1 (ẑ_i ≥ 0):
    l_j⁺ = min { W[j,:]·z + b[j] : W[i,:]·z + b[i] ≥ 0, z ∈ [l,u] }
    u_j⁺ = max { W[j,:]·z + b[j] : W[i,:]·z + b[i] ≥ 0, z ∈ [l,u] }

  When δ_i = 0 (ẑ_i ≤ 0):
    l_j⁻ = min { W[j,:]·z + b[j] : W[i,:]·z + b[i] ≤ 0, z ∈ [l,u] }
    u_j⁻ = max { W[j,:]·z + b[j] : W[i,:]·z + b[i] ≤ 0, z ∈ [l,u] }

Each conditional bound is computed analytically using the parametric LP dual
for the special structure: box + one linear constraint.  No LP solver needed.

If l_j⁺ > l̂_j (tighter lower bound when i active):
  add:  W[j,:]·z_prev + b[j]  ≥  l̂_j + (l_j⁺ - l̂_j) * δ_i

If u_j⁺ < û_j (tighter upper bound when i active):
  add:  W[j,:]·z_prev + b[j]  ≤  û_j - (û_j - u_j⁺) * δ_i

Analogous cuts for the δ_i = 0 case (using 1 - δ_i).

If either condition is LP-infeasible (the activation state cannot occur given
the input box), we fix the binary: δ_i = 0 or δ_i = 1 accordingly.

### Soundness

Each cut is a valid inequality over the feasible set {z_prev ∈ [l,u], δ_i ∈ {0,1}}.
When δ_i = 1, the constraint W[i,:]·z_prev + b[i] ≥ 0 holds (from the standard
BoundedMixedIntegerLP encoding: z_next[i] ≥ W[i,:]·z_prev + b[i] and z_next[i] ≥ 0,
and z_next[i] ≤ û_i·δ_i → δ_i=1 iff z_next[i] > 0 iff ẑ_i > 0).
The LP-based conditional bound is a valid over-approximation over this restricted set.

### Reference

Anderson et al. (2020), "Strong mixed-integer programming formulations for
trained neural networks." Mathematical Programming 183(1), 3–39.
"""

using LinearAlgebra

# ---------------------------------------------------------------------------
# Analytical conditional bound via parametric LP dual
# ---------------------------------------------------------------------------

"""
    _conditional_bound_min(c, b_c, a, b_rhs, l, u)

Compute min c·z + b_c  subject to  a·z ≥ b_rhs,  l ≤ z ≤ u.

Uses the parametric dual for box + one linear constraint:
  The optimal solution sets each z[k] to l[k] (if c[k] ≥ 0) or u[k] (if c[k] < 0),
  then greedily moves the cheapest coordinates (by ratio c[k]/a[k]) toward the
  constraint-satisfying direction until a·z ≥ b_rhs.

Returns NaN if the feasible region {a·z ≥ b_rhs, z ∈ [l,u]} is empty.
"""
function _conditional_bound_min(c::Vector{Float64}, b_c::Float64,
                                 a::Vector{Float64}, b_rhs::Float64,
                                 l::Vector{Float64}, u::Vector{Float64})::Float64
    n = length(c)

    # Unconstrained minimum on box
    z = [c[k] >= 0.0 ? l[k] : u[k] for k in 1:n]
    val = dot(c, z) + b_c
    deficit = b_rhs - dot(a, z)

    deficit <= 1e-10 && return val  # constraint already satisfied

    # Collect coordinates useful for satisfying a·z ≥ b_rhs cheaply:
    #   a[k] > 0 and z[k] = l[k]  (i.e., c[k] ≥ 0): move z[k] toward u[k]
    #   a[k] < 0 and z[k] = u[k]  (i.e., c[k] ≤ 0): move z[k] toward l[k]
    # In both cases cost_per_unit_gain = c[k] / a[k] ≥ 0.
    useful = Tuple{Float64,Float64,Int}[]  # (cost_per_gain, capacity, k)
    for k in 1:n
        if a[k] > 1e-14 && z[k] < u[k]       # a[k]>0, z[k]=l[k]
            push!(useful, (c[k] / a[k], u[k] - l[k], k))
        elseif a[k] < -1e-14 && z[k] > l[k]  # a[k]<0, z[k]=u[k]
            push!(useful, (c[k] / a[k], u[k] - l[k], k))
        end
    end
    sort!(useful)  # ascending cost_per_gain

    for (cpg, cap, k) in useful
        gain = abs(a[k]) * cap
        if gain >= deficit - 1e-12
            # Use this coordinate partially
            frac = deficit / abs(a[k])
            val += (a[k] > 0.0 ? c[k] : -c[k]) * frac
            deficit = 0.0
            break
        else
            # Use fully
            val += (a[k] > 0.0 ? c[k] : -c[k]) * cap
            deficit -= gain
        end
    end

    deficit > 1e-8 && return NaN  # infeasible
    return val
end

"""
    _conditional_bound_max(c, b_c, a, b_rhs, l, u)

Compute max c·z + b_c  subject to  a·z ≥ b_rhs,  l ≤ z ≤ u.
"""
function _conditional_bound_max(c::Vector{Float64}, b_c::Float64,
                                 a::Vector{Float64}, b_rhs::Float64,
                                 l::Vector{Float64}, u::Vector{Float64})::Float64
    neg = _conditional_bound_min(-c, -b_c, a, b_rhs, l, u)
    isnan(neg) && return NaN
    return -neg
end

"""
    conditional_preact_bounds(layer, bounds_prev, i, j)

Compute conditional pre-activation bounds on ẑ_j for each activation state of δ_i.

Returns `(l_active, u_active, l_inactive, u_inactive)` where:
- `l_active`, `u_active`: bounds on ẑ_j when δ_i = 1 (ẑ_i ≥ 0)
- `l_inactive`, `u_inactive`: bounds on ẑ_j when δ_i = 0 (ẑ_i ≤ 0)

NaN entries indicate the corresponding activation state is LP-infeasible given
the input bounds (the state cannot occur).
"""
function conditional_preact_bounds(layer::Layer{ReLU}, bounds_prev::Hyperrectangle,
                                    i::Int, j::Int)
    W  = layer.weights
    b  = layer.bias
    l  = low(bounds_prev)
    u  = high(bounds_prev)

    W_i = W[i, :]
    W_j = W[j, :]
    b_i = b[i]
    b_j = b[j]

    # Active:   ẑ_i ≥ 0  ↔  W_i · z ≥ -b_i
    # Inactive: ẑ_i ≤ 0  ↔  (-W_i) · z ≥ b_i
    l_act  = _conditional_bound_min( W_j, b_j,  W_i,  -b_i, l, u)
    u_act  = _conditional_bound_max( W_j, b_j,  W_i,  -b_i, l, u)
    l_inact = _conditional_bound_min(W_j, b_j, -W_i,   b_i, l, u)
    u_inact = _conditional_bound_max(W_j, b_j, -W_i,   b_i, l, u)

    return (l_act, u_act, l_inact, u_inact)
end

# ---------------------------------------------------------------------------
# Add Anderson cuts to a JuMP model
# ---------------------------------------------------------------------------

"""
    add_anderson_cuts!(model, network, neurons, deltas, bounds;
                       tol=1e-6, max_cuts_per_layer=typemax(Int))

Add pairwise conditional Anderson cuts to `model` for every ReLU layer.

For each pair (i, j) of unstable neurons in the same layer, computes conditional
pre-activation bounds and adds valid linear inequalities tightening the MILP.

### Arguments
- `model`: JuMP model (already encoded with `encode_network!`)
- `network`: the `Network` struct
- `neurons`: `Vector{Vector{VariableRef}}` from `init_neurons` — neurons[k] is the
  post-activation variable vector that is the INPUT to layer k.
- `deltas`: `Vector{Vector{VariableRef}}` from `init_deltas`
- `bounds`: `Vector{Hyperrectangle}` from `get_bounds_crown_backsub` — bounds[k]
  is the post-activation hyperrectangle that is the INPUT to layer k.
- `tol`: minimum improvement in a bound (relative to the unconditional width)
  required before adding a cut (default 1e-6).
- `max_cuts_per_layer`: cap on cuts added per layer (default unlimited).

### Returns
`(n_cuts_added, n_fixed_binaries)`: total cuts added and binaries fixed to constant.
"""
function add_anderson_cuts!(model, network::Network,
                             neurons::Vector,
                             deltas::Vector,
                             bounds::Vector{Hyperrectangle};
                             tol::Float64 = 1e-6,
                             max_cuts_per_layer::Int = typemax(Int))

    n_cuts    = 0
    n_fixed   = 0

    for (k, layer) in enumerate(network.layers)
        layer.activation isa ReLU || continue

        bounds_prev = bounds[k]          # post-activation bounds feeding into layer k
        z_prev      = neurons[k]         # post-activation variables (input to layer k)
        δ           = deltas[k]          # binary variables for layer k

        W  = layer.weights
        b  = layer.bias
        n_k = length(b)

        # Pre-activation bounds for layer k (from box interval arithmetic)
        preact = approximate_affine_map(layer, bounds_prev)
        l̂ = low(preact)
        û = high(preact)

        # Unstable neurons (need binary variables)
        unstable = findall(j -> l̂[j] < 0.0 < û[j], 1:n_k)
        length(unstable) < 2 && continue

        # Precompute ẑ_j affine expressions once per unstable neuron j
        # (avoids rebuilding the sum O(n²) times in the inner loop)
        # Use variable type from z_prev to support both JuMP.VariableRef and Plasmo.NodeVariableRef
        VT = eltype(z_prev)
        ẑ_exprs = Dict{Int, GenericAffExpr{Float64, VT}}()
        for j in unstable  # j is a neuron index (value), not an index into `unstable`
            e = GenericAffExpr{Float64, VT}()
            add_to_expression!(e, b[j])
            for m in eachindex(z_prev)
                add_to_expression!(e, W[j, m], z_prev[m])
            end
            ẑ_exprs[j] = e
        end

        layer_cuts = 0
        # Iterate over unordered pairs {i,j} only (i < j) to avoid double-counting.
        # Cuts for δ_i are added when processing pair (i,j); cuts for δ_j when (j,i)
        # would be processed — we cover both directions by adding cuts for BOTH
        # δ_i and δ_j within the same pair iteration.
        for a_idx in eachindex(unstable)
            layer_cuts >= max_cuts_per_layer && break
            i = unstable[a_idx]
            for b_idx in (a_idx+1):lastindex(unstable)
                layer_cuts >= max_cuts_per_layer && break
                j = unstable[b_idx]

                # Cuts on ẑ_j conditioned on δ_i
                l_act_j, u_act_j, l_inact_j, u_inact_j = conditional_preact_bounds(
                    layer, bounds_prev, i, j)

                ẑ_j = ẑ_exprs[j]

                if isnan(l_act_j) && isnan(u_act_j)
                    @constraint(model, δ[i] == 0)
                    n_fixed += 1
                else
                    if !isnan(l_act_j) && l_act_j > l̂[j] + tol
                        @constraint(model, ẑ_j >= l̂[j] + (l_act_j - l̂[j]) * δ[i])
                        n_cuts += 1; layer_cuts += 1
                    end
                    if !isnan(u_act_j) && u_act_j < û[j] - tol
                        @constraint(model, ẑ_j <= û[j] - (û[j] - u_act_j) * δ[i])
                        n_cuts += 1; layer_cuts += 1
                    end
                end

                if isnan(l_inact_j) && isnan(u_inact_j)
                    @constraint(model, δ[i] == 1)
                    n_fixed += 1
                else
                    if !isnan(l_inact_j) && l_inact_j > l̂[j] + tol
                        @constraint(model, ẑ_j >= l̂[j] + (l_inact_j - l̂[j]) * (1 - δ[i]))
                        n_cuts += 1; layer_cuts += 1
                    end
                    if !isnan(u_inact_j) && u_inact_j < û[j] - tol
                        @constraint(model, ẑ_j <= û[j] - (û[j] - u_inact_j) * (1 - δ[i]))
                        n_cuts += 1; layer_cuts += 1
                    end
                end

                # Cuts on ẑ_i conditioned on δ_j (symmetric direction)
                l_act_i, u_act_i, l_inact_i, u_inact_i = conditional_preact_bounds(
                    layer, bounds_prev, j, i)

                ẑ_i = ẑ_exprs[i]

                if isnan(l_act_i) && isnan(u_act_i)
                    @constraint(model, δ[j] == 0)
                    n_fixed += 1
                else
                    if !isnan(l_act_i) && l_act_i > l̂[i] + tol
                        @constraint(model, ẑ_i >= l̂[i] + (l_act_i - l̂[i]) * δ[j])
                        n_cuts += 1; layer_cuts += 1
                    end
                    if !isnan(u_act_i) && u_act_i < û[i] - tol
                        @constraint(model, ẑ_i <= û[i] - (û[i] - u_act_i) * δ[j])
                        n_cuts += 1; layer_cuts += 1
                    end
                end

                if isnan(l_inact_i) && isnan(u_inact_i)
                    @constraint(model, δ[j] == 1)
                    n_fixed += 1
                else
                    if !isnan(l_inact_i) && l_inact_i > l̂[i] + tol
                        @constraint(model, ẑ_i >= l̂[i] + (l_inact_i - l̂[i]) * (1 - δ[j]))
                        n_cuts += 1; layer_cuts += 1
                    end
                    if !isnan(u_inact_i) && u_inact_i < û[i] - tol
                        @constraint(model, ẑ_i <= û[i] - (û[i] - u_inact_i) * (1 - δ[j]))
                        n_cuts += 1; layer_cuts += 1
                    end
                end
            end
        end
    end

    return (n_cuts, n_fixed)
end

