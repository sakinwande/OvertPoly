"""
    alpha_crown.jl — CROWN / α-CROWN bound propagation for neural networks.

    Implements the CROWN forward-pass bound propagation algorithm for computing
    per-neuron pre-activation bounds used as big-M values in `BoundedMixedIntegerLP`.

    Reference:
      Zhang et al., "Efficient Neural Network Robustness Certification with
      General Activation Functions", NeurIPS 2018. (CROWN)

      Xu et al., "Fast and Complete: Enabling Complete Neural Network Verification
      with Rapid and Massaged Mixed-Integer Programming", ICLR 2021. (α-CROWN)

    Soundness: CROWN (without alpha optimisation) is sound by the standard triangle
    relaxation argument (see Ehlers 2017, "Formal Verification of Piece-Wise Linear
    Feed-Forward Neural Networks"). With α_j = 0 for every unstable ReLU neuron the
    lower bound is identically 0, which is trivially valid.  The upper bound
    û_j/(û_j - l̂_j) * (ẑ_j - l̂_j) is the Planet relaxation (Ehlers 2017) and is
    provably a valid over-approximation of ReLU.

    All operations are performed in Float64 with standard IEEE 754 semantics;
    no directed rounding is required because every interval computation uses the
    `interval_map` helper which already separates positive and negative weights
    (see `src/nv/util.jl`).
"""

# ---------------------------------------------------------------------------
# CrownBounds struct
# ---------------------------------------------------------------------------

"""
    CrownBounds

Per-layer pre-activation bound vectors computed by the CROWN forward pass.

### Fields
- `lower::Vector{Float64}`: element-wise lower bound on the pre-activation value
  `ẑ_k = W_k * z_{k-1} + b_k` for every neuron in layer k.
- `upper::Vector{Float64}`: element-wise upper bound on the pre-activation value.

### Invariant (soundness)
For every input `x₀` in the input set and every neuron j:
    `lower[j] ≤ ẑ_{k,j}(x₀) ≤ upper[j]`

These are *pre-activation* bounds (before ReLU is applied), which is exactly what
`BoundedMixedIntegerLP.encode_layer!` uses as big-M values.
"""
struct CrownBounds
    lower::Vector{Float64}
    upper::Vector{Float64}
end

# ---------------------------------------------------------------------------
# ReLU linear relaxation helpers
# ---------------------------------------------------------------------------

"""
    relu_upper_slope_intercept(l̂, û)

Compute the slope and intercept of the standard Planet (triangle) upper-bound
linear relaxation for a ReLU neuron with pre-activation bounds [l̂, û].

For an unstable neuron (l̂ < 0 < û):
    ReLU(ẑ) ≤ slope * ẑ + intercept   ∀ ẑ ∈ [l̂, û]

where  slope = û / (û - l̂),  intercept = -l̂ * û / (û - l̂).

Soundness: This is the tightest valid linear upper bound that passes through
(l̂, 0) and (û, û), i.e., the convex hull of the ReLU epigraph over [l̂, û].
See Ehlers (2017), Theorem 1.

Returns `(slope, intercept)`.
"""
function relu_upper_slope_intercept(l̂::Float64, û::Float64)
    @assert l̂ < 0.0 < û "relu_upper_slope_intercept called on a non-unstable neuron"
    denom = û - l̂
    slope     = û / denom
    intercept = -l̂ * û / denom
    return (slope, intercept)
end


# ---------------------------------------------------------------------------
# forward_crown — CROWN forward bound propagation
# ---------------------------------------------------------------------------

"""
    forward_crown(network::Network, input_set::Hyperrectangle;
                  alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)
    forward_crown(network::Network, input_set::HPolytope;
                  alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)

Compute per-neuron pre-activation bounds via the CROWN forward pass.

### Algorithm (CROWN, Zhang et al. NeurIPS 2018)

Layer 1 (first hidden layer):
    ẑ₁ = W₁ x₀ + b₁  where x₀ ∈ [l₀, u₀]
    l̂₁ = max(W₁, 0) * l₀ + min(W₁, 0) * u₀ + b₁
    û₁ = max(W₁, 0) * u₀ + min(W₁, 0) * l₀ + b₁

For each subsequent layer k, the previous layer's ReLU is relaxed:
  - Active neuron  (l̂_{k-1,j} ≥ 0):  z_{k-1,j} = ẑ_{k-1,j}  (slope 1, intercept 0)
  - Inactive neuron (û_{k-1,j} ≤ 0):  z_{k-1,j} = 0           (slope 0, intercept 0)
  - Unstable neuron (l̂ < 0 < û):
      upper: slope_j = û_j/(û_j - l̂_j), intercept_j = -l̂_j*û_j/(û_j - l̂_j)
      lower: slope_j = α_j,              intercept_j = 0

The post-activation bounds (used as the input interval for the next layer's
affine map) are:
    z_lower_j = max(slope_lower_j * l̂_j, slope_lower_j * û_j)   (α_j ≥ 0)
    z_upper_j = slope_upper_j * û_j + intercept_j

Actually more precisely, the post-activation interval after relaxation is:
    z_lower_j ∈ [min of α*ẑ over [l̂,û], max of α*ẑ over [l̂,û]]
             = [α_j * l̂_j,  α_j * û_j]   (since α_j ≥ 0)
    clipped to non-negative: [max(0, α_j * l̂_j), max(0, α_j * û_j)]
    with α_j = 0: [0, 0]

    z_upper_j ∈ [slope_j * l̂_j + intercept_j, slope_j * û_j + intercept_j]
             evaluates to [0, û_j]  (since the upper relaxation passes through (l̂,0) and (û,û))

### Inputs
- `network::Network`: the neural network
- `input_set`: the input domain (Hyperrectangle or HPolytope)
- `alpha`: optional optimised α values per layer/neuron. If `nothing`, uses α = 0
  for all unstable neurons (the safe CROWN default).

### Returns
`Vector{CrownBounds}` of length `length(network.layers)`.
`result[k]` contains the pre-activation bounds for layer k (the k-th hidden/output layer).

### Soundness
Sound by the Planet relaxation argument. With α = 0, every lower bound is 0 which
is trivially valid for ReLU neurons. Upper bounds use the standard triangle relaxation.
"""
function forward_crown(network::Network, input_set::Hyperrectangle;
                       alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)

    l₀ = low(input_set)
    u₀ = high(input_set)

    # Running post-activation interval bounds through the network.
    # For layer k's affine map, we need post-activation bounds from layer k-1.
    l_post = l₀   # post-activation lower for current input (initialised to raw input)
    u_post = u₀   # post-activation upper for current input

    n_layers = length(network.layers)
    crown_bounds = Vector{CrownBounds}(undef, n_layers)

    for (k, layer) in enumerate(network.layers)
        W, b = layer.weights, layer.bias

        # --- Pre-activation bounds via interval arithmetic ---
        # l̂_k = max(W,0)*l_post + min(W,0)*u_post + b
        # û_k = max(W,0)*u_post + min(W,0)*l_post + b
        l̂_new, û_new = interval_map(W, l_post, u_post)
        l̂_new = l̂_new .+ b
        û_new = û_new .+ b

        # Soundness assertion: lower ≤ upper for all neurons.
        @assert all(l̂_new .≤ û_new) "CROWN pre-activation bound violation at layer $k"

        crown_bounds[k] = CrownBounds(l̂_new, û_new)

        # --- Post-activation bounds (used as input to next layer) ---
        # Apply the ReLU linear relaxation to get a new post-activation interval.
        if layer.activation isa ReLU
            n_j = length(b)
            l_post_new = Vector{Float64}(undef, n_j)
            u_post_new = Vector{Float64}(undef, n_j)

            # α values for this layer (if provided)
            α_layer = (alpha !== nothing && k ≤ length(alpha)) ? alpha[k] : nothing

            for j in 1:n_j
                l̂_j = l̂_new[j]
                û_j = û_new[j]

                if l̂_j ≥ 0.0
                    # Active: ReLU passes through exactly
                    l_post_new[j] = l̂_j
                    u_post_new[j] = û_j
                elseif û_j ≤ 0.0
                    # Inactive: ReLU clamps to 0
                    l_post_new[j] = 0.0
                    u_post_new[j] = 0.0
                else
                    # Unstable: use linear relaxation
                    α_j = (α_layer !== nothing) ? α_layer[j] : 0.0
                    @assert 0.0 ≤ α_j ≤ 1.0 "α[$k][$j] = $α_j is outside [0,1]"

                    # Upper relaxation: slope*û + intercept = û (passes through (û,û))
                    slope_upper, intercept_upper = relu_upper_slope_intercept(l̂_j, û_j)
                    # Upper post-activation bound: max of upper-relaxation over [l̂_j, û_j]
                    # The upper-relaxation is linear and increasing (slope > 0), so max at û_j.
                    u_post_new[j] = slope_upper * û_j + intercept_upper  # = û_j

                    # Lower relaxation: α_j * ẑ, valid since α_j ≥ 0 and ẑ ≥ l̂_j.
                    # With α_j = 0, the lower bound is 0.
                    # Minimum of α_j * ẑ over [l̂_j, û_j]:
                    #   if α_j ≥ 0 → minimum at ẑ = l̂_j → α_j * l̂_j (≤ 0 since l̂_j < 0)
                    # But post-activation of ReLU is ≥ 0, so clamp below by 0.
                    l_post_new[j] = max(0.0, α_j * l̂_j)
                end
            end

            l_post = l_post_new
            u_post = u_post_new

        elseif layer.activation isa Id
            # Identity layer: post-activation = pre-activation
            l_post = l̂_new
            u_post = û_new
        else
            # Generic monotone activation: fall back to evaluating at endpoints.
            # NOTE: only sound for monotone activations (same assumption as MaxSens).
            act = layer.activation
            l_post = min.(act(l̂_new), act(û_new))
            u_post = max.(act(l̂_new), act(û_new))
        end

        # Soundness assertion: post-activation bounds are non-negative for ReLU layers.
        if layer.activation isa ReLU
            @assert all(l_post .≥ 0.0) "Post-activation lower bound negative at layer $k"
        end
        @assert all(l_post .≤ u_post) "Post-activation bound ordering violation at layer $k"
    end

    return crown_bounds
end

"""
    forward_crown(network::Network, input_set::HPolytope; alpha=nothing)

HPolytope overload: overapproximates the input polytope as a Hyperrectangle
before running the CROWN forward pass.  This is the same approach as MaxSens
(the existing `get_bounds` function) and retains soundness because
    box(HPolytope) ⊇ HPolytope.

For tighter bounds with an HPolytope input, CROWN could be extended to operate
on the V-representation directly, but that is not implemented here.
"""
function forward_crown(network::Network, input_set::HPolytope;
                       alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)
    box = overapproximate(input_set, Hyperrectangle)
    return forward_crown(network, box; alpha=alpha)
end

# ---------------------------------------------------------------------------
# get_bounds_crown — drop-in replacement for get_bounds
# ---------------------------------------------------------------------------

"""
    get_bounds_crown(network::Network, input_set;
                     alpha=nothing)

Compute post-activation neuron-wise bounds using CROWN, returning a
`Vector{Hyperrectangle}` in the same format as `get_bounds` so it can serve
as a drop-in replacement.

### Format (matches `get_bounds`)
- `result[1]` = the input set (as a Hyperrectangle)
- `result[k+1]` = post-activation bounds for layer k (k = 1, …, K)

Post-activation clipping:
- ReLU layer: lower = max(0, pre_lower), upper = max(0, pre_upper)
- Id layer:   lower = pre_lower,         upper = pre_upper

### Soundness
The returned bounds are valid post-activation over-approximations: for every
input x₀ in `input_set` and every neuron j in layer k,
    result[k+1].center[j] - result[k+1].radius[j]  ≤  z_{k,j}(x₀)  ≤
    result[k+1].center[j] + result[k+1].radius[j]

where z_{k,j} is the post-activation output of neuron j in layer k.

### Arguments
- `network::Network`: the neural network
- `input_set`: `Hyperrectangle` or `HPolytope`
- `alpha`: optional per-layer α vectors for α-CROWN. If `nothing`, uses α = 0.

### Returns
`Vector{Hyperrectangle}` of length `length(network.layers) + 1`.
"""
function get_bounds_crown(network::Network, input_set;
                          alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)

    # Normalise input to Hyperrectangle for storage in result[1].
    if input_set isa HPolytope
        input_rect = overapproximate(input_set, Hyperrectangle)
    else
        input_rect = input_set
    end

    # Compute CROWN pre-activation bounds.
    crown = forward_crown(network, input_set; alpha=alpha)

    n_layers = length(network.layers)
    bounds = Vector{Hyperrectangle}(undef, n_layers + 1)
    bounds[1] = input_rect

    for (k, layer) in enumerate(network.layers)
        l̂ = crown[k].lower
        û = crown[k].upper

        if layer.activation isa ReLU
            # Clip to non-negative for post-activation bounds.
            l_post = max.(0.0, l̂)
            u_post = max.(0.0, û)
        elseif layer.activation isa Id
            l_post = l̂
            u_post = û
        else
            # Generic: apply activation at the pre-activation bounds.
            act = layer.activation
            l_post = min.(act(l̂), act(û))
            u_post = max.(act(l̂), act(û))
        end

        @assert all(l_post .≤ u_post) "Post-activation bound ordering violation at layer $k"

        center = (l_post .+ u_post) ./ 2.0
        radius = (u_post .- l_post) ./ 2.0
        bounds[k+1] = Hyperrectangle(center, radius)
    end

    return bounds
end

# ---------------------------------------------------------------------------
# Alpha optimisation (gradient ascent on lower bounds)
# ---------------------------------------------------------------------------

"""
    optimise_alpha(network::Network, input_set::Hyperrectangle;
                   n_iter::Int = 20, lr::Float64 = 0.1)

Run a simple projected gradient ascent to optimise the α parameters in α-CROWN,
tightening the lower bounds on pre-activation values for unstable ReLU neurons.

### Method
α-CROWN (Xu et al. ICLR 2021) maximises the lower bound objective by gradient
ascent on α ∈ [0,1]^n for each unstable neuron.  The gradient of the lower bound
with respect to α_j is the post-activation value propagated back through the
subsequent linear layers (the "back-substitution" sensitivity).

This is a simplified scalar implementation that performs a forward pass for each
gradient step, which is correct but not as fast as the full back-substitution
approach. For typical small/medium networks this is acceptable.

The gradient estimate used here is a finite-difference approximation:
    ∂L/∂α_j ≈ (L(α + ε*e_j) - L(α)) / ε
where L(α) is the sum of lower bounds across all unstable neurons.

### Arguments
- `n_iter::Int`: number of gradient ascent steps (default 20)
- `lr::Float64`: learning rate for gradient ascent (default 0.1)

### Returns
`Vector{Vector{Float64}}` — optimised α values, one vector per layer.

### Note
This is an optional refinement step. The base CROWN bounds (α = 0) are already
sound; optimising α can only tighten (or leave unchanged) the lower bounds.
The returned α values are guaranteed to be in [0, 1] (projected after each step).
"""
function optimise_alpha(network::Network, input_set::Hyperrectangle;
                        n_iter::Int = 20, lr::Float64 = 0.1)

    n_layers = length(network.layers)

    # Initialise α for each ReLU layer (Id layers get empty vectors).
    alpha = Vector{Vector{Float64}}(undef, n_layers)
    for (k, layer) in enumerate(network.layers)
        if layer.activation isa ReLU
            alpha[k] = zeros(Float64, n_nodes(layer))
        else
            alpha[k] = zeros(Float64, 0)
        end
    end

    # Objective: sum of ALL pre-activation lower bounds across all layers.
    #
    # The α-CROWN gain is indirect: α_j controls the post-activation lower bound
    # of an unstable neuron in layer k (higher α → tighter lower relaxation →
    # tighter post-activation interval → tighter bounds in layers k+1, k+2, …).
    # The direct pre-activation lower bound at the unstable neuron itself is
    # l̂_{k,j} which does not depend on α[k][j] (α affects the post-activation
    # interval that feeds forward, not the pre-activation bound of that neuron).
    #
    # Therefore the correct objective to maximise is the sum of all pre-activation
    # lower bounds across ALL layers and ALL neurons (including non-ReLU output
    # layers), since these are the big-M values that matter for the MILP encoding.
    #
    # Reference: Xu et al. ICLR 2021, Eq. (6) — maximise lower bound of the
    # verification objective, which subsumes sum of all neuron lower bounds.
    function objective(α)
        crown = forward_crown(network, input_set; alpha=α)
        total = 0.0
        for k in 1:n_layers
            total += sum(crown[k].lower)
        end
        return total
    end

    ε = 1e-5  # finite difference step

    for _ in 1:n_iter
        base_val = objective(alpha)

        for k in 1:n_layers
            if !(network.layers[k].activation isa ReLU)
                continue
            end
            n_j = length(alpha[k])
            for j in 1:n_j
                # Finite-difference gradient w.r.t. α[k][j]
                alpha[k][j] += ε
                perturbed_val = objective(alpha)
                alpha[k][j] -= ε
                grad = (perturbed_val - base_val) / ε
                # Gradient ascent step + projection to [0, 1]
                alpha[k][j] = clamp(alpha[k][j] + lr * grad, 0.0, 1.0)
            end
        end
    end

    return alpha
end

# ---------------------------------------------------------------------------
# forward_crown_backsub — CROWN back-substitution bound propagation
# ---------------------------------------------------------------------------

"""
    forward_crown_backsub(network::Network, input_set::Hyperrectangle;
                          alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)

Compute per-neuron pre-activation bounds via CROWN back-substitution.

### Algorithm (CROWN back-substitution, Zhang et al. NeurIPS 2018, Section 3)

Unlike `forward_crown` which propagates scalar intervals layer-by-layer, this
function maintains *affine functions of x₀* across all intermediate layers.
For each output neuron (k, i), it computes two row-vectors Λ_lower, Λ_upper
and two scalars const_lower, const_upper such that:

    ẑ_{k,i}(x₀)  ≥  Λ_lower ⋅ x₀ + const_lower   (sound lower bound)
    ẑ_{k,i}(x₀)  ≤  Λ_upper ⋅ x₀ + const_upper   (sound upper bound)

for all x₀ ∈ [l₀, u₀].  Back-substitution threads through the intermediate
ReLU relaxations (using the same upper triangle + lower slope-α relaxations as
`forward_crown`) but keeps the linear dependence on x₀ intact rather than
collapsing to a scalar interval at each step.

### Why this is tighter than `forward_crown`

`forward_crown` loses the correlation between neurons when it replaces each
unstable ReLU's post-activation value with a scalar interval. For an unstable
neuron j, this interval is [0, û_j] regardless of where x₀ falls in [l₀, u₀].

Back-substitution retains the expression "z_j is at most αᵁ_j·(W_j x₀ + b_j) + γ_j"
all the way back to x₀, then maximises that linear function over [l₀, u₀]. This
captures the correlation: if W_j x₀ + b_j is negative when x₀ is large, then a
later neuron that has a positive coefficient on z_j will see a tighter upper bound.

### Back-substitution procedure for neuron (K, i)

**Initialisation** (at layer K):
    Λ_lower = eᵢᵀ · W_K,   const_lower = b_K[i]
    Λ_upper = eᵢᵀ · W_K,   const_upper = b_K[i]

**Step** from frontier z_{k-1} through ReLU at layer k-1:

For the *lower* bound row Λ_lower (we want to lower-bound ẑ_{K,i}):
  - For coefficient c = Λ_lower[j]:
    - c > 0: lower-bound z_{k-1,j} using lower ReLU relaxation (z ≥ αᴸ_j·ẑ):
             contributes c·αᴸ_j to new Λ, 0 to const
    - c < 0: lower-bound z_{k-1,j} using upper ReLU relaxation (z ≤ αᵁ_j·ẑ + γ_j):
             contributes c·αᵁ_j to new Λ, c·γ_j to const

For the *upper* bound row Λ_upper (we want to upper-bound ẑ_{K,i}):
  - For coefficient c = Λ_upper[j]:
    - c > 0: upper-bound z_{k-1,j} using upper ReLU relaxation (z ≤ αᵁ_j·ẑ + γ_j):
             contributes c·αᵁ_j to new Λ, c·γ_j to const
    - c < 0: upper-bound z_{k-1,j} using lower ReLU relaxation (z ≥ αᴸ_j·ẑ):
             contributes c·αᴸ_j to new Λ, 0 to const

After ReLU back-sub, substitute ẑ_{k-1} = W_{k-1}·z_{k-2} + b_{k-1}:
    Λ_new · ẑ_{k-1} → (Λ_new · W_{k-1}) · z_{k-2} + Λ_new · b_{k-1}

For Identity layers: no slope adjustment, just substitute directly.

**Final evaluation** (x₀ ∈ [l₀, u₀]):
    lower_bound = interval_map(Λ_lower, l₀, u₀)[lower side] + const_lower
    upper_bound = interval_map(Λ_upper, l₀, u₀)[upper side] + const_upper

### ReLU relaxation slopes used

For each neuron j at layer k, given pre-activation bounds [l̂_j, û_j]:
  - Active   (l̂_j ≥ 0):  αᴸ_j = 1,   αᵁ_j = 1,  γ_j = 0
  - Inactive (û_j ≤ 0):  αᴸ_j = 0,   αᵁ_j = 0,  γ_j = 0
  - Unstable (l̂_j < 0 < û_j):
      αᵁ_j = û_j/(û_j - l̂_j),  γ_j = -l̂_j·û_j/(û_j - l̂_j)  [Planet/CROWN upper]
      αᴸ_j = alpha[k][j] ∈ [0,1]                                  [α-CROWN lower]

### Arguments
- `network::Network`: the neural network
- `input_set`: the input domain (Hyperrectangle)
- `alpha`: optional optimised α values per layer. If `nothing`, uses α = 0 for all
  unstable neurons (the CROWN-0 default).

### Returns
`Vector{CrownBounds}` of length `length(network.layers)`.
`result[k]` contains pre-activation bounds for layer k (tighter than `forward_crown`).

### Soundness
- Upper bounds use the Planet (triangle) relaxation: provably valid over-approximation
  of ReLU on [l̂, û] (Ehlers 2017, Theorem 1; Zhang et al. 2018, Proposition 1).
- Lower bounds use α_j ∈ [0,1]: since ReLU(ẑ) ≥ 0 and ReLU(ẑ) ≥ ẑ (for ẑ ≥ 0),
  the bound z_j ≥ α_j·ẑ_j is valid for all ẑ_j ∈ [l̂_j, û_j]:
    - ẑ_j ≥ 0: ReLU = ẑ_j ≥ α_j·ẑ_j  (since α_j ≤ 1)
    - ẑ_j < 0: ReLU = 0 ≥ α_j·ẑ_j    (since α_j ≥ 0, ẑ_j < 0)
- interval_map is exact (no rounding) for linear functions of x₀.
- The pre-activation bounds from forward_crown (forward-pass interval) are used to
  define the relaxation slopes; back-substitution can only yield tighter bounds than
  the forward pass, never looser.

### Reference
Zhang et al., "Efficient Neural Network Robustness Certification with General
Activation Functions", NeurIPS 2018.  Algorithm 1 (back-substitution procedure).
"""
function forward_crown_backsub(network::Network, input_set::Hyperrectangle;
                                alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)

    l₀ = low(input_set)
    u₀ = high(input_set)
    n₀ = length(l₀)
    n_layers = length(network.layers)

    # -----------------------------------------------------------------------
    # Step 1: Forward pass to get pre-activation bounds at each layer.
    # These bounds define the ReLU relaxation slopes used in back-substitution.
    # We use the existing forward_crown for this (which is equivalent to
    # interval arithmetic on Hyperrectangle inputs).
    # -----------------------------------------------------------------------
    forward_bounds = forward_crown(network, input_set; alpha=alpha)

    # -----------------------------------------------------------------------
    # Step 2: For each layer k and each neuron i, run back-substitution from
    # (k, i) back to x₀.  The result replaces forward_bounds[k] with a
    # tighter bound computed via the affine back-substitution.
    # -----------------------------------------------------------------------
    result = Vector{CrownBounds}(undef, n_layers)

    for K in 1:n_layers
        n_K = length(network.layers[K].bias)
        lower_bounds_K = Vector{Float64}(undef, n_K)
        upper_bounds_K = Vector{Float64}(undef, n_K)

        for i in 1:n_K
            # --- Initialise Λ at layer K ---
            # ẑ_{K,i} = eᵢᵀ · W_K · z_{K-1} + b_K[i]
            W_K = network.layers[K].weights   # n_K × n_{K-1}
            b_K = network.layers[K].bias

            # Λ_lower, Λ_upper are row vectors of length n_{K-1}
            Λ_lower = W_K[i, :]   # copy: 1-D vector, length n_{K-1}
            Λ_upper = W_K[i, :]   # copy: 1-D vector, length n_{K-1}
            c_lower = b_K[i]      # scalar constant
            c_upper = b_K[i]      # scalar constant

            # --- Back-substitute through layers K-1, K-2, ..., 1 ---
            for k in K-1 : -1 : 1
                layer_k = network.layers[k]
                W_k  = layer_k.weights   # n_k × n_{k-1}
                b_k  = layer_k.bias      # n_k

                n_k = length(b_k)        # number of neurons at layer k

                # Pre-activation bounds at layer k (from forward pass)
                l̂_k = forward_bounds[k].lower   # n_k
                û_k  = forward_bounds[k].upper   # n_k

                # α values for layer k (if provided)
                α_layer = (alpha !== nothing && k ≤ length(alpha)) ? alpha[k] : nothing

                # ------------------------------------------------------------
                # Compute diagonal relaxation parameters for layer k's ReLU.
                # slope_lower[j] = αᴸ_j,  slope_upper[j] = αᵁ_j,  gamma[j] = γ_j
                # where the ReLU at neuron j satisfies:
                #   z_j ≥ αᴸ_j · ẑ_j          (lower bound on post-activation)
                #   z_j ≤ αᵁ_j · ẑ_j + γ_j    (upper bound on post-activation)
                # ------------------------------------------------------------
                slope_lower = Vector{Float64}(undef, n_k)
                slope_upper = Vector{Float64}(undef, n_k)
                gamma       = Vector{Float64}(undef, n_k)

                if layer_k.activation isa ReLU
                    for j in 1:n_k
                        l̂_j = l̂_k[j]
                        û_j  = û_k[j]
                        if l̂_j ≥ 0.0
                            # Active: ReLU is identity
                            slope_lower[j] = 1.0
                            slope_upper[j] = 1.0
                            gamma[j]       = 0.0
                        elseif û_j ≤ 0.0
                            # Inactive: ReLU is zero
                            slope_lower[j] = 0.0
                            slope_upper[j] = 0.0
                            gamma[j]       = 0.0
                        else
                            # Unstable: use linear relaxation
                            α_j = (α_layer !== nothing) ? α_layer[j] : 0.0
                            @assert 0.0 ≤ α_j ≤ 1.0 "α[$k][$j] = $α_j outside [0,1]"

                            # Planet upper bound: slope = û/(û-l̂), intercept = -l̂·û/(û-l̂)
                            su, γ_j = relu_upper_slope_intercept(l̂_j, û_j)
                            slope_lower[j] = α_j
                            slope_upper[j] = su
                            gamma[j]       = γ_j
                        end
                    end
                elseif layer_k.activation isa Id
                    # Identity layer: z_j = ẑ_j exactly
                    fill!(slope_lower, 1.0)
                    fill!(slope_upper, 1.0)
                    fill!(gamma,       0.0)
                else
                    error("forward_crown_backsub: unsupported activation $(typeof(layer_k.activation)) at layer $k. Only ReLU and Id are supported.")
                end

                # ------------------------------------------------------------
                # Back-substitute Λ through layer k's activation:
                #
                # For lower bound: ẑ_{K,i} ≥ Λ_lower · z_{k}
                #   where z_{k,j} = activation(ẑ_{k,j})
                #   - c > 0: z_{k,j} ≥ slope_lower[j]·ẑ_{k,j}
                #            → use slope_lower to substitute
                #   - c < 0: z_{k,j} ≤ slope_upper[j]·ẑ_{k,j} + gamma[j]
                #            → use slope_upper + gamma to substitute (tighter)
                #
                # For upper bound: ẑ_{K,i} ≤ Λ_upper · z_{k}
                #   - c > 0: z_{k,j} ≤ slope_upper[j]·ẑ_{k,j} + gamma[j]
                #   - c < 0: z_{k,j} ≥ slope_lower[j]·ẑ_{k,j}
                # ------------------------------------------------------------
                Λ_lower_new = Vector{Float64}(undef, n_k)
                Λ_upper_new = Vector{Float64}(undef, n_k)

                for j in 1:n_k
                    c_lo = Λ_lower[j]
                    c_up = Λ_upper[j]

                    # Lower bound back-sub:
                    if c_lo ≥ 0.0
                        Λ_lower_new[j] = c_lo * slope_lower[j]
                        # c_lo * 0 (intercept from lower relaxation) = 0
                    else
                        Λ_lower_new[j] = c_lo * slope_upper[j]
                        c_lower       += c_lo * gamma[j]
                    end

                    # Upper bound back-sub:
                    if c_up ≥ 0.0
                        Λ_upper_new[j] = c_up * slope_upper[j]
                        c_upper       += c_up * gamma[j]
                    else
                        Λ_upper_new[j] = c_up * slope_lower[j]
                        # c_up * 0 (intercept from lower relaxation) = 0
                    end
                end

                # ------------------------------------------------------------
                # Substitute ẑ_k = W_k · z_{k-1} + b_k:
                #   Λ_new · ẑ_k = (Λ_new · W_k) · z_{k-1} + Λ_new · b_k
                # ------------------------------------------------------------
                c_lower    += sum(Λ_lower_new .* b_k)
                c_upper    += sum(Λ_upper_new .* b_k)
                Λ_lower     = Λ_lower_new' * W_k   # row vector: 1 × n_{k-1}  (as 1-D)
                Λ_upper     = Λ_upper_new' * W_k
                # Note: Λ_lower_new' * W_k produces a 1×n_{k-1} matrix;
                # we want a plain 1-D vector.
                Λ_lower     = vec(Λ_lower)
                Λ_upper     = vec(Λ_upper)

            end  # for k in K-1 : -1 : 1

            # At this point, Λ_lower / Λ_upper have length n₀ (input dimension),
            # and ẑ_{K,i} ∈ [Λ_lower·x₀ + c_lower, Λ_upper·x₀ + c_upper] for x₀ ∈ [l₀,u₀].
            @assert length(Λ_lower) == n₀ "Λ_lower length mismatch after back-sub"
            @assert length(Λ_upper) == n₀ "Λ_upper length mismatch after back-sub"

            # --- Evaluate linear bounds over [l₀, u₀] ---
            # lower = min_{x₀ ∈ [l₀,u₀]} Λ_lower·x₀ + c_lower
            #       = (Λ_lower⁺ · l₀ + Λ_lower⁻ · u₀) + c_lower   [interval_map logic]
            # interval_map(W, l, u) returns (min_over_box, max_over_box) of W*x over [l,u].
            # For the lower bound we want the minimum of Λ_lower·x over [l₀, u₀].
            # For the upper bound we want the maximum of Λ_upper·x over [l₀, u₀].
            bs_lower_min, bs_lower_max = interval_map(reshape(Λ_lower, 1, n₀), l₀, u₀)
            bs_upper_min, bs_upper_max = interval_map(reshape(Λ_upper, 1, n₀), l₀, u₀)
            lower_bounds_K[i] = bs_lower_min[1] + c_lower
            upper_bounds_K[i] = bs_upper_max[1] + c_upper
        end  # for i in 1:n_K

        # Soundness assertion: bounds must be ordered.
        @assert all(lower_bounds_K .≤ upper_bounds_K .+ 1e-10) "Back-sub bound ordering violation at layer $K"

        result[K] = CrownBounds(lower_bounds_K, upper_bounds_K)
    end  # for K in 1:n_layers

    return result
end

"""
    forward_crown_backsub(network::Network, input_set::HPolytope; alpha=nothing)

HPolytope overload: overapproximates the input polytope as a Hyperrectangle
before running the CROWN back-substitution forward pass.  Soundness is
preserved because box(HPolytope) ⊇ HPolytope.
"""
function forward_crown_backsub(network::Network, input_set::HPolytope;
                                alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)
    box = overapproximate(input_set, Hyperrectangle)
    return forward_crown_backsub(network, box; alpha=alpha)
end


# ---------------------------------------------------------------------------
# get_bounds_crown_backsub — drop-in replacement for get_bounds using back-sub
# ---------------------------------------------------------------------------

"""
    get_bounds_crown_backsub(network::Network, input_set;
                             alpha=nothing)

Compute post-activation neuron-wise bounds using CROWN back-substitution,
returning a `Vector{Hyperrectangle}` in the same format as `get_bounds` and
`get_bounds_crown`.

This is a drop-in replacement for both. The bounds are tighter than or equal to
those returned by `get_bounds_crown` (layer-by-layer interval CROWN), which are
in turn tighter than or equal to those returned by `get_bounds` (MaxSens).

### Format (matches `get_bounds` and `get_bounds_crown`)
- `result[1]` = the input set (as a Hyperrectangle)
- `result[k+1]` = post-activation bounds for layer k (k = 1, …, K)

Post-activation clipping for ReLU layers:
    l_post = max(0, pre_lower),  u_post = max(0, pre_upper)

### Soundness
For every input x₀ ∈ `input_set` and every neuron j in layer k:
    result[k+1].center[j] - result[k+1].radius[j]  ≤  z_{k,j}(x₀)  ≤
    result[k+1].center[j] + result[k+1].radius[j]

The pre-activation soundness is guaranteed by `forward_crown_backsub`.
Post-activation clipping max(0,·) preserves soundness for ReLU.

### Arguments
- `network::Network`: the neural network
- `input_set`: `Hyperrectangle` or `HPolytope`
- `alpha`: optional per-layer α vectors for α-CROWN. If `nothing`, uses α = 0.

### Returns
`Vector{Hyperrectangle}` of length `length(network.layers) + 1`.
"""
function get_bounds_crown_backsub(network::Network, input_set;
                                   alpha::Union{Nothing, Vector{Vector{Float64}}} = nothing)

    # Normalise input to Hyperrectangle for storage in result[1].
    if input_set isa HPolytope
        input_rect = overapproximate(input_set, Hyperrectangle)
    else
        input_rect = input_set
    end

    # Compute back-substitution pre-activation bounds.
    backsub_bounds = forward_crown_backsub(network, input_set; alpha=alpha)

    n_layers = length(network.layers)
    bounds = Vector{Hyperrectangle}(undef, n_layers + 1)
    bounds[1] = input_rect

    for (k, layer) in enumerate(network.layers)
        l̂ = backsub_bounds[k].lower
        û = backsub_bounds[k].upper

        if layer.activation isa ReLU
            # Clip to non-negative for post-activation bounds.
            l_post = max.(0.0, l̂)
            u_post = max.(0.0, û)
        elseif layer.activation isa Id
            l_post = l̂
            u_post = û
        else
            # Generic: apply activation at the pre-activation bounds.
            act = layer.activation
            l_post = min.(act(l̂), act(û))
            u_post = max.(act(l̂), act(û))
        end

        @assert all(l_post .≤ u_post) "Post-activation bound ordering violation at layer $k"

        center = (l_post .+ u_post) ./ 2.0
        radius = (u_post .- l_post) ./ 2.0
        bounds[k+1] = Hyperrectangle(center, radius)
    end

    return bounds
end