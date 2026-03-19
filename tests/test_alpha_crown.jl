"""
    test_alpha_crown.jl

Tests for the α-CROWN / CROWN bound propagation implementation in
src/nv/alpha_crown.jl.

Run from the repo root with:
    julia --project=. tests/test_alpha_crown.jl

Tests cover:
  1. CROWN bounds are no wider than MaxSens bounds on a small multi-layer network.
  2. Soundness: for random inputs in the input set, the true pre-activation value
     is always within the computed bounds.
  3. All three ReLU neuron cases: active, inactive, unstable.
  4. get_bounds_crown returns a Vector{Hyperrectangle} matching get_bounds format.
  5. HPolytope input is handled (overapproximated to Hyperrectangle).
  6. Alpha optimisation only tightens (or preserves) bounds relative to CROWN default.
"""

using Test
using LazySets
import Printf

# Activate project and load the nv utilities.
# nn_mip_encoding.jl pulls in all nv/ includes plus alpha_crown.jl.
include(joinpath(@__DIR__, "..", "src", "nn_mip_encoding.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""Build a small 2-layer network with specified weights/biases."""
function make_network(W1, b1, act1, W2, b2, act2)
    return Network([Layer(W1, b1, act1), Layer(W2, b2, act2)])
end

"""Sample n_samples uniform random points from a Hyperrectangle."""
function random_inputs(box::Hyperrectangle, n_samples::Int; rng=nothing)
    lo = low(box)
    hi = high(box)
    n  = length(lo)
    if rng === nothing
        return [lo .+ rand(n) .* (hi .- lo) for _ in 1:n_samples]
    else
        return [lo .+ rand(rng, n) .* (hi .- lo) for _ in 1:n_samples]
    end
end

"""Compute per-layer pre-activation values for a concrete input vector."""
function true_preactivations(network::Network, x0::Vector{Float64})
    pre_acts = Vector{Vector{Float64}}(undef, length(network.layers))
    z = x0
    for (k, layer) in enumerate(network.layers)
        z_hat = layer.weights * z + layer.bias
        pre_acts[k] = z_hat
        z = layer.activation(z_hat)
    end
    return pre_acts
end

# ──────────────────────────────────────────────────────────────────────────────
# Test suite
# ──────────────────────────────────────────────────────────────────────────────

@testset "alpha-CROWN bound propagation" begin

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: All three ReLU neuron cases (single hidden layer, 3 neurons)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "ReLU neuron cases: active / inactive / unstable" begin
        # Design weights so that over the input box [0.5, 1.5]:
        #   neuron 1:   W=[1], b=1  → ẑ ∈ [1.5, 2.5]   active (l̂>0)
        #   neuron 2:   W=[1], b=-2 → ẑ ∈ [-1.5, -0.5]  inactive (û<0)
        #   neuron 3:   W=[1], b=-1 → ẑ ∈ [-0.5,  0.5]  unstable
        W1 = reshape([1.0, 1.0, 1.0], 3, 1)  # 3×1 weight matrix
        b1 = [1.0, -2.0, -1.0]
        input_box = Hyperrectangle(low=[0.5], high=[1.5])

        net1 = Network([Layer(W1, b1, ReLU())])
        crown = forward_crown(net1, input_box)

        @test length(crown) == 1

        # Active neuron (j=1): l̂ ≥ 0
        @test crown[1].lower[1] ≥ 0.0
        @test crown[1].upper[1] ≥ 0.0
        @test crown[1].lower[1] ≤ crown[1].upper[1]

        # Inactive neuron (j=2): û ≤ 0
        @test crown[1].upper[2] ≤ 0.0
        @test crown[1].lower[2] ≤ crown[1].upper[2]

        # Unstable neuron (j=3): l̂ < 0 < û
        @test crown[1].lower[3] < 0.0
        @test crown[1].upper[3] > 0.0

        # Verify exact values (interval arithmetic on single-layer linear network)
        # W=[1], b=1, input [0.5,1.5] → pre-act [1.5, 2.5]
        @test isapprox(crown[1].lower[1], 1.5; atol=1e-10)
        @test isapprox(crown[1].upper[1], 2.5; atol=1e-10)
        # W=[1], b=-2, input [0.5,1.5] → pre-act [-1.5, -0.5]
        @test isapprox(crown[1].lower[2], -1.5; atol=1e-10)
        @test isapprox(crown[1].upper[2], -0.5; atol=1e-10)
        # W=[1], b=-1, input [0.5,1.5] → pre-act [-0.5, 0.5]
        @test isapprox(crown[1].lower[3], -0.5; atol=1e-10)
        @test isapprox(crown[1].upper[3],  0.5; atol=1e-10)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Soundness — true pre-activation values lie within CROWN bounds
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Soundness: true pre-activations within CROWN bounds" begin
        # 2-layer network: 2 inputs → 4 hidden (ReLU) → 2 outputs (Id)
        W1 = [ 1.0  2.0;
              -1.0  1.0;
               0.5 -0.5;
               2.0  0.5]
        b1 = [0.1, -0.3, 0.2, -0.5]
        W2 = [1.0 -1.0  0.5  0.2;
              0.3  0.7 -0.4  1.0]
        b2 = [0.0, 0.1]

        net2 = make_network(W1, b1, ReLU(), W2, b2, Id())
        input_box = Hyperrectangle(low=[-1.0, -1.0], high=[1.0, 1.0])

        crown = forward_crown(net2, input_box)

        # Sample random inputs and check containment
        n_samples = 1000
        lo = low(input_box)
        hi = high(input_box)
        n_in = length(lo)

        for _ in 1:n_samples
            x0 = lo .+ rand(n_in) .* (hi .- lo)
            pre_acts = true_preactivations(net2, x0)

            for (k, layer) in enumerate(net2.layers)
                for j in eachindex(pre_acts[k])
                    # Each true pre-activation must be within computed bounds.
                    lb = crown[k].lower[j]
                    ub = crown[k].upper[j]
                    # Allow small floating-point tolerance.
                    @test pre_acts[k][j] ≥ lb - 1e-9
                    @test pre_acts[k][j] ≤ ub + 1e-9
                end
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: CROWN bounds are no wider than MaxSens bounds
    # ─────────────────────────────────────────────────────────────────────────
    @testset "CROWN bounds tighter than or equal to MaxSens bounds" begin
        # Multi-layer network where correlation between neurons matters.
        # MaxSens loses correlation at each ReLU; CROWN retains it.
        W1 = [ 1.0  0.5;
              -0.5  1.0;
               1.0 -1.0]
        b1 = [0.0, 0.1, -0.2]
        W2 = [ 1.0  1.0 -0.5;
              -1.0  0.5  1.0]
        b2 = [0.05, -0.05]
        W3 = [2.0 -1.0]
        b3 = [0.0]

        net3 = make_network(W1, b1, ReLU(), W2, b2, ReLU())
        # Add a third layer
        net3_full = Network([net3.layers..., Layer(W3, b3, Id())])
        input_box = Hyperrectangle(low=[-0.5, -0.5], high=[0.5, 0.5])

        # CROWN post-activation bounds (same format as get_bounds)
        bounds_crown  = get_bounds_crown(net3_full, input_box)
        # MaxSens post-activation bounds
        bounds_maxsens = get_bounds(net3_full, input_box)

        @test length(bounds_crown)   == length(bounds_maxsens)
        @test length(bounds_crown)   == length(net3_full.layers) + 1

        # For every layer, CROWN width ≤ MaxSens width (or within tolerance).
        # "Width" of a Hyperrectangle is 2 * radius (= high - low per dimension).
        tighter_or_equal = true
        for k in axes(bounds_crown, 1)[2:end]
            w_crown   = 2.0 .* bounds_crown[k].radius
            w_maxsens = 2.0 .* bounds_maxsens[k].radius
            # Allow CROWN to be slightly wider due to interval-arithmetic differences
            # at the first layer (same algorithm), but for hidden layers it should
            # not be worse.  Use a generous tolerance.
            if any(w_crown .> w_maxsens .+ 1e-8)
                tighter_or_equal = false
                @warn "CROWN wider than MaxSens at layer $k: crown=$(w_crown) maxsens=$(w_maxsens)"
            end
        end
        @test tighter_or_equal

        # Bounds[1] (input set) should match for both.
        @test isapprox(low(bounds_crown[1]),  low(input_box);  atol=1e-12)
        @test isapprox(high(bounds_crown[1]), high(input_box); atol=1e-12)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: get_bounds_crown returns Vector{Hyperrectangle} matching format
    # ─────────────────────────────────────────────────────────────────────────
    @testset "get_bounds_crown return format" begin
        W1 = [1.0 0.0; 0.0 1.0]
        b1 = [0.0, 0.0]
        W2 = [1.0 1.0]
        b2 = [0.0]
        net = make_network(W1, b1, ReLU(), W2, b2, Id())
        input_box = Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0])

        bounds = get_bounds_crown(net, input_box)

        @test bounds isa Vector{Hyperrectangle}
        @test length(bounds) == length(net.layers) + 1
        # First element is the input set
        @test isapprox(bounds[1].center, input_box.center; atol=1e-12)
        @test isapprox(bounds[1].radius, input_box.radius; atol=1e-12)
        # All bounds are non-negative width
        for k in eachindex(bounds)
            @test all(bounds[k].radius .≥ 0.0)
            @test all(low(bounds[k]) .≤ high(bounds[k]))
        end
        # ReLU post-activation lower bounds are non-negative for hidden layers
        for k in axes(bounds, 1)[2:end-1]  # skip input (k=1) and output Id layer (last)
            @test all(low(bounds[k]) .≥ -1e-12)
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: HPolytope input is handled
    # ─────────────────────────────────────────────────────────────────────────
    @testset "HPolytope input handled via overapproximation" begin
        # Unit simplex: x1 ≥ 0, x2 ≥ 0, x1 + x2 ≤ 1
        # Bounding box: [0,1] × [0,1]
        A = [-1.0  0.0;
              0.0 -1.0;
              1.0  1.0]
        b_poly = [0.0, 0.0, 1.0]
        poly = HPolytope(A, b_poly)

        W1 = [1.0 1.0; -1.0 0.5]
        b1 = [0.0, 0.1]
        net = Network([Layer(W1, b1, ReLU())])

        # Should not throw
        crown  = forward_crown(net, poly)
        bounds = get_bounds_crown(net, poly)

        @test crown isa Vector{CrownBounds}
        @test bounds isa Vector{Hyperrectangle}
        @test length(bounds) == 2

        # Verify soundness: all vertices of the polytope should have their
        # true pre-activation within the computed bounds.
        verts = vertices_list(poly)
        for v in verts
            pre = net.layers[1].weights * v + net.layers[1].bias
            for j in eachindex(pre)
                @test pre[j] ≥ crown[1].lower[j] - 1e-9
                @test pre[j] ≤ crown[1].upper[j] + 1e-9
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Alpha optimisation does not loosen bounds
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Alpha optimisation: tighter or equal lower bounds" begin
        W1 = [ 1.0  1.0;
              -1.0  1.0]
        b1 = [-0.3, -0.3]
        W2 = [1.0 -1.0]
        b2 = [0.0]
        net = make_network(W1, b1, ReLU(), W2, b2, Id())
        input_box = Hyperrectangle(low=[-1.0, -1.0], high=[1.0, 1.0])

        # Default α = 0 (base CROWN)
        crown_base = forward_crown(net, input_box)

        # Optimised α
        alpha_opt = optimise_alpha(net, input_box; n_iter=30, lr=0.1)
        crown_opt  = forward_crown(net, input_box; alpha=alpha_opt)

        # α optimisation should tighten or preserve lower bounds (never loosen them).
        # Upper bounds are unaffected by α.
        for k in 1:length(net.layers)
            for j in eachindex(crown_base[k].lower)
                # Lower bound with optimised α should be ≥ lower bound with α=0.
                @test crown_opt[k].lower[j] ≥ crown_base[k].lower[j] - 1e-9
                # Upper bounds should be identical (α does not affect upper bound).
                @test isapprox(crown_opt[k].upper[j], crown_base[k].upper[j]; atol=1e-9)
            end
        end

        # Soundness still holds after α optimisation.
        lo = low(input_box)
        hi = high(input_box)
        n_in = length(lo)
        for _ in 1:500
            x0 = lo .+ rand(n_in) .* (hi .- lo)
            pre_acts = true_preactivations(net, x0)
            for (k, _) in enumerate(net.layers)
                for j in eachindex(pre_acts[k])
                    @test pre_acts[k][j] ≥ crown_opt[k].lower[j] - 1e-9
                    @test pre_acts[k][j] ≤ crown_opt[k].upper[j] + 1e-9
                end
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 7: Single-neuron edge cases
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Edge cases: single neuron, zero-width input" begin
        # Single neuron, single input, zero-width box (point input)
        W = reshape([1.0], 1, 1)
        b = [0.5]
        net = Network([Layer(W, b, ReLU())])
        pt = Hyperrectangle([0.3], [0.0])  # point {0.3}

        crown = forward_crown(net, pt)
        # Pre-activation is exactly 1*0.3 + 0.5 = 0.8
        @test isapprox(crown[1].lower[1], 0.8; atol=1e-10)
        @test isapprox(crown[1].upper[1], 0.8; atol=1e-10)

        # Inactive point: pre-activation is negative
        b_neg = [-1.0]
        net_neg = Network([Layer(W, b_neg, ReLU())])
        crown_neg = forward_crown(net_neg, pt)
        # Pre-activation: 1*0.3 - 1.0 = -0.7
        @test isapprox(crown_neg[1].lower[1], -0.7; atol=1e-10)
        @test isapprox(crown_neg[1].upper[1], -0.7; atol=1e-10)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 8: Identity (output) layer handled correctly
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Identity output layer pass-through" begin
        W1 = [1.0;; ]   # 1×1
        W1 = reshape([1.0], 1, 1)
        b1 = [0.0]
        W2 = reshape([2.0], 1, 1)
        b2 = [-1.0]

        net = make_network(W1, b1, ReLU(), W2, b2, Id())
        input_box = Hyperrectangle(low=[-1.0], high=[1.0])

        crown = forward_crown(net, input_box)

        # Layer 1 pre-activation: W1*x + b1 = x, bounds [-1, 1]
        @test isapprox(crown[1].lower[1], -1.0; atol=1e-10)
        @test isapprox(crown[1].upper[1],  1.0; atol=1e-10)

        # Post-activation layer 1: max(0, [-1,1]) = [0, 1]
        # Layer 2 (Id): W2*z + b2 = 2*z - 1, z ∈ [0,1] → pre-act ∈ [-1, 1]
        @test isapprox(crown[2].lower[1], -1.0; atol=1e-10)
        @test isapprox(crown[2].upper[1],  1.0; atol=1e-10)
    end

end  # @testset "alpha-CROWN bound propagation"

# ──────────────────────────────────────────────────────────────────────────────
# Real-network tests: CROWN vs MaxSens on production .nnet files
# ──────────────────────────────────────────────────────────────────────────────

# Helper: mean of a vector (avoid Statistics dependency)
mean_vec(v) = sum(v) / length(v)

"""
    compare_crown_maxsens(net, input_box; n_samples=500, tol=1e-8)

Run CROWN and MaxSens bound propagation on `net` over `input_box`.
Returns a named tuple with per-layer width statistics and comparison results.

Soundness check: sample `n_samples` random inputs and verify that true
pre-activation values lie within the CROWN pre-activation bounds (from
`forward_crown`).  Note: `get_bounds_crown` returns *post*-activation bounds;
soundness must be checked against *pre*-activation bounds from `forward_crown`.

Tightness check: CROWN post-activation width ≤ MaxSens post-activation width per
neuron (up to `tol`) for every layer, using `get_bounds_crown` vs `get_bounds`.
"""
function compare_crown_maxsens(net::Network, input_box::Hyperrectangle;
                                n_samples::Int=500, tol::Float64=1e-8)
    # Post-activation bounds for tightness comparison (same format as get_bounds)
    bounds_crown_post   = get_bounds_crown(net, input_box)
    bounds_maxsens_post = get_bounds(net, input_box)

    # Pre-activation bounds for soundness check
    crown_pre = forward_crown(net, input_box)  # Vector{CrownBounds}

    n_layers = length(net.layers)
    @assert length(bounds_crown_post)   == n_layers + 1
    @assert length(bounds_maxsens_post) == n_layers + 1
    @assert length(crown_pre)           == n_layers

    # Per-layer average width (high - low per dimension, then mean)
    avg_width_crown   = Float64[]
    avg_width_maxsens = Float64[]
    tighter           = Bool[]  # true if crown ≤ maxsens for ALL neurons at this layer

    for k in 2:n_layers+1   # skip k=1 (input set, identical for both)
        w_c = high(bounds_crown_post[k])   .- low(bounds_crown_post[k])
        w_m = high(bounds_maxsens_post[k]) .- low(bounds_maxsens_post[k])
        push!(avg_width_crown,   mean_vec(w_c))
        push!(avg_width_maxsens, mean_vec(w_m))
        push!(tighter,           all(w_c .- w_m .<= tol))
    end

    # Soundness: sample random inputs and check true pre-activation values
    # lie within the CROWN pre-activation bounds (forward_crown output).
    lo = low(input_box);  hi = high(input_box);  n_in = length(lo)
    max_violation = 0.0
    for _ in 1:n_samples
        x0 = lo .+ rand(n_in) .* (hi .- lo)
        pre_acts = true_preactivations(net, x0)
        for k in 1:n_layers
            lb_k = crown_pre[k].lower   # pre-activation lower bound, layer k
            ub_k = crown_pre[k].upper   # pre-activation upper bound, layer k
            for j in eachindex(pre_acts[k])
                viol_lo = lb_k[j] - pre_acts[k][j]   # > 0 means unsound (LB too high)
                viol_hi = pre_acts[k][j] - ub_k[j]   # > 0 means unsound (UB too low)
                max_violation = max(max_violation, viol_lo, viol_hi)
            end
        end
    end

    return (
        avg_width_crown   = avg_width_crown,
        avg_width_maxsens = avg_width_maxsens,
        tighter           = tighter,
        max_soundness_violation = max_violation,
    )
end

@testset "Real networks: CROWN vs MaxSens bound comparison" begin

    # Absolute path of this file — used to build repo-relative paths robustly.
    repo_root = joinpath(@__DIR__, "..")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 9: Single Pendulum ARCH-COMP — 2-input controller
    # Network: Networks/ARCH-COMP-2023/nnet/controllerSinglePendulum.nnet
    # Input domain from single_pend_overtPoly_graph.jl:
    #   Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Single Pendulum ARCH-COMP controller" begin
        net_path  = joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                             "controllerSinglePendulum.nnet")
        @assert isfile(net_path) "Expected network file not found: $net_path"
        net       = read_nnet(net_path; last_layer_activation=Id())
        input_box = Hyperrectangle(low=[1.0, 0.0], high=[1.2, 0.2])

        result = compare_crown_maxsens(net, input_box; n_samples=500, tol=1e-8)

        println("\n--- Single Pendulum ARCH-COMP ---")
        for (k, (wc, wm, t)) in enumerate(zip(result.avg_width_crown,
                                               result.avg_width_maxsens,
                                               result.tighter))
            println("  Layer $(k): avg CROWN post-act width = $(round(wc, digits=6)), " *
                    "avg MaxSens post-act width = $(round(wm, digits=6)), " *
                    "CROWN tighter = $t")
        end
        println("  Max pre-activation soundness violation (should be ≤ 0): " *
                "$(result.max_soundness_violation)")

        # Soundness: no CROWN pre-activation bound violated by any sampled input
        @test result.max_soundness_violation ≤ 1e-9

        # Tightness: CROWN post-activation width ≤ MaxSens per neuron (within tol)
        @test all(result.tighter)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Test 10: TORA controller — 4-input controller
    # Network: Networks/ARCH-COMP-2023/nnet/controllerTORA.nnet
    # Input domain from tora_overtPoly_distrOpt.jl:
    #   Hyperrectangle(low=[0.6,-0.7,-0.4,0.5], high=[0.7,-0.6,-0.3,0.6])
    # ──────────────────────────────────────────────────────────────────────────
    @testset "TORA controller" begin
        net_path  = joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                             "controllerTORA.nnet")
        @assert isfile(net_path) "Expected network file not found: $net_path"
        net       = read_nnet(net_path; last_layer_activation=Id())
        input_box = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5],
                                   high=[0.7, -0.6, -0.3, 0.6])

        result = compare_crown_maxsens(net, input_box; n_samples=500, tol=1e-8)

        println("\n--- TORA ---")
        for (k, (wc, wm, t)) in enumerate(zip(result.avg_width_crown,
                                               result.avg_width_maxsens,
                                               result.tighter))
            println("  Layer $(k): avg CROWN post-act width = $(round(wc, digits=6)), " *
                    "avg MaxSens post-act width = $(round(wm, digits=6)), " *
                    "CROWN tighter = $t")
        end
        println("  Max pre-activation soundness violation (should be ≤ 0): " *
                "$(result.max_soundness_violation)")

        @test result.max_soundness_violation ≤ 1e-9
        @test all(result.tighter)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Test 11: Unicycle controller — 4-input, 2-output controller
    # Network: Networks/ARCH-COMP-2023/nnet/controllerUnicycle.nnet
    # Input domain from unicycle_overtPoly_distrOpt.jl:
    #   Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high=[9.55,-4.45,2.11,1.51])
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Unicycle controller" begin
        net_path  = joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                             "controllerUnicycle.nnet")
        @assert isfile(net_path) "Expected network file not found: $net_path"
        net       = read_nnet(net_path; last_layer_activation=Id())
        input_box = Hyperrectangle(low=[9.50, -4.50, 2.10, 1.50],
                                   high=[9.55, -4.45, 2.11, 1.51])

        result = compare_crown_maxsens(net, input_box; n_samples=500, tol=1e-8)

        println("\n--- Unicycle ---")
        for (k, (wc, wm, t)) in enumerate(zip(result.avg_width_crown,
                                               result.avg_width_maxsens,
                                               result.tighter))
            println("  Layer $(k): avg CROWN post-act width = $(round(wc, digits=6)), " *
                    "avg MaxSens post-act width = $(round(wm, digits=6)), " *
                    "CROWN tighter = $t")
        end
        println("  Max pre-activation soundness violation (should be ≤ 0): " *
                "$(result.max_soundness_violation)")

        @test result.max_soundness_violation ≤ 1e-9
        @test all(result.tighter)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Test 12: ACC controller — 5-input controller
    # Network: Networks/ARCH-COMP-2023/nnet/controllerACC.nnet
    # Network inputs (from acc_overtPoly_graph.jl acc_control function):
    #   [vSet, tGap, vEgo, dRel, vRel]
    # Using representative domain around initial condition:
    #   domain = Hyperrectangle(low=[90,32,-ϵ,10,30,-ϵ], high=[110,32.2,ϵ,11,30.2,ϵ])
    # Network input domain derived via acc_control():
    #   vSet=30 (constant), tGap=1.4 (constant),
    #   vEgo ∈ [30-ϵ, 30.2+ϵ],
    #   dRel ∈ [90-11, 110-10] = [79, 100],
    #   vRel ∈ [32-30.2, 32.2-30] = [1.8, 2.2]
    # ──────────────────────────────────────────────────────────────────────────
    @testset "ACC controller" begin
        net_path  = joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                             "controllerACC.nnet")
        @assert isfile(net_path) "Expected network file not found: $net_path"
        net       = read_nnet(net_path; last_layer_activation=Id())

        # Network inputs: [vSet, tGap, vEgo, dRel, vRel]
        # Derived from acc_control() applied to the example domain
        ϵ_acc = 1e-8
        input_box = Hyperrectangle(
            low  = [30.0 - ϵ_acc, 1.40 - ϵ_acc, 30.0 - ϵ_acc, 79.0,  1.8],
            high = [30.0 + ϵ_acc, 1.40 + ϵ_acc, 30.2 + ϵ_acc, 100.0, 2.2],
        )

        result = compare_crown_maxsens(net, input_box; n_samples=500, tol=1e-8)

        println("\n--- ACC ---")
        for (k, (wc, wm, t)) in enumerate(zip(result.avg_width_crown,
                                               result.avg_width_maxsens,
                                               result.tighter))
            println("  Layer $(k): avg CROWN post-act width = $(round(wc, digits=6)), " *
                    "avg MaxSens post-act width = $(round(wm, digits=6)), " *
                    "CROWN tighter = $t")
        end
        println("  Max pre-activation soundness violation (should be ≤ 0): " *
                "$(result.max_soundness_violation)")

        @test result.max_soundness_violation ≤ 1e-9
        @test all(result.tighter)
    end

end  # @testset "Real networks: CROWN vs MaxSens bound comparison"

# ──────────────────────────────────────────────────────────────────────────────
# Tests 13–16: Alpha optimisation benchmark on real networks
#
# For each of the 4 real networks (single pendulum, TORA, unicycle, ACC) with
# their Hyperrectangle input domains, we:
#   1. Compute CROWN bounds with α=0 (base)
#   2. Optimise α and compute CROWN bounds with optimised α
#   3. Check that optimised bounds never loosen (soundness monotonicity)
#   4. Check soundness of optimised bounds via random sampling
#   5. Print per-layer improvement statistics for the report
#
# The key metric is the improvement in PRE-ACTIVATION lower bounds:
#   Δlower_{k,j} = crown_opt[k].lower[j] - crown_base[k].lower[j]  (≥ 0)
# Positive Δlower means tighter (higher) lower bound → smaller big-M → tighter MILP.
# ──────────────────────────────────────────────────────────────────────────────

"""
    benchmark_alpha_opt(net, input_box, net_name; n_samples=500, n_iter=50, lr=0.1)

Run alpha-optimisation benchmark for a single network and input box.
Returns a named tuple with per-layer statistics for reporting.

Prints a summary table to stdout.
"""
function benchmark_alpha_opt(net::Network, input_box::Hyperrectangle, net_name::String;
                              n_samples::Int=500, n_iter::Int=50, lr::Float64=0.1)
    n_layers = length(net.layers)

    # Step 1: base CROWN bounds (α = 0)
    crown_base = forward_crown(net, input_box)

    # Step 2: optimise α
    alpha_opt = optimise_alpha(net, input_box; n_iter=n_iter, lr=lr)

    # Step 3: CROWN bounds with optimised α
    crown_opt = forward_crown(net, input_box; alpha=alpha_opt)

    # Per-layer statistics
    # n_unstable[k]   : number of unstable neurons at layer k (in base CROWN)
    # mean_δlower[k]  : mean improvement in lower bound across all neurons at layer k
    # max_δlower[k]   : max improvement in lower bound at layer k
    # any_meaningful[k]: true if max improvement > 0.01
    n_unstable    = Int[]
    mean_δlower   = Float64[]
    max_δlower    = Float64[]
    any_meaningful = Bool[]

    for k in 1:n_layers
        l̂_base = crown_base[k].lower
        û_base  = crown_base[k].upper
        l̂_opt  = crown_opt[k].lower

        unstable_count = count(j -> l̂_base[j] < 0.0 < û_base[j], eachindex(l̂_base))
        push!(n_unstable, unstable_count)

        δlower = l̂_opt .- l̂_base   # element-wise improvement (≥ 0 if sound)
        push!(mean_δlower,    mean_vec(δlower))
        push!(max_δlower,     maximum(δlower))
        push!(any_meaningful, maximum(δlower) > 0.01)
    end

    # Print summary table
    println("\n─── $net_name ───")
    println("  Alpha optimisation: n_iter=$n_iter, lr=$lr")
    println("  $(Printf.@sprintf("%-8s %-12s %-10s %-10s %-12s", "Layer", "#Unstable", "Mean Δl̂", "Max Δl̂", "Meaningful?"))")
    for k in 1:n_layers
        layer_type = net.layers[k].activation isa ReLU ? "ReLU" : "Id"
        println("  $(Printf.@sprintf("%-8s %-12d %-10.6f %-10.6f %-12s",
                 "L$k($layer_type)", n_unstable[k],
                 mean_δlower[k], max_δlower[k],
                 any_meaningful[k] ? "YES" : "no"))")
    end

    # Soundness check: sample random inputs, verify true pre-activations ≤ crown_opt bounds
    lo = low(input_box);  hi = high(input_box);  n_in = length(lo)
    max_lb_violation = 0.0
    max_ub_violation = 0.0
    for _ in 1:n_samples
        x0 = lo .+ rand(n_in) .* (hi .- lo)
        pre_acts = true_preactivations(net, x0)
        for k in 1:n_layers
            for j in eachindex(pre_acts[k])
                lb_viol = crown_opt[k].lower[j] - pre_acts[k][j]   # > 0 → unsound LB
                ub_viol = pre_acts[k][j] - crown_opt[k].upper[j]   # > 0 → unsound UB
                max_lb_violation = max(max_lb_violation, lb_viol)
                max_ub_violation = max(max_ub_violation, ub_viol)
            end
        end
    end
    println("  Max LB soundness violation (must be ≤ 0): $max_lb_violation")
    println("  Max UB soundness violation (must be ≤ 0): $max_ub_violation")

    return (
        crown_base     = crown_base,
        crown_opt      = crown_opt,
        alpha_opt      = alpha_opt,
        n_unstable     = n_unstable,
        mean_δlower    = mean_δlower,
        max_δlower     = max_δlower,
        any_meaningful = any_meaningful,
        max_lb_violation = max_lb_violation,
        max_ub_violation = max_ub_violation,
    )
end

@testset "Alpha optimisation benchmark on real networks" begin

    repo_root = joinpath(@__DIR__, "..")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 13: Single Pendulum — alpha optimisation benchmark
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Alpha opt benchmark: Single Pendulum" begin
        net_path  = joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                             "controllerSinglePendulum.nnet")
        @assert isfile(net_path) "Expected network file not found: $net_path"
        net       = read_nnet(net_path; last_layer_activation=Id())
        input_box = Hyperrectangle(low=[1.0, 0.0], high=[1.2, 0.2])

        result = benchmark_alpha_opt(net, input_box, "Single Pendulum";
                                     n_samples=500, n_iter=50, lr=0.1)

        # Soundness: optimised bounds must be valid for all sampled inputs
        @test result.max_lb_violation ≤ 1e-9
        @test result.max_ub_violation ≤ 1e-9

        # Monotonicity: optimised lower bounds must be ≥ base lower bounds (within tol)
        for k in eachindex(result.crown_base)
            for j in eachindex(result.crown_base[k].lower)
                δ = result.crown_opt[k].lower[j] - result.crown_base[k].lower[j]
                @test δ ≥ -1e-9
            end
        end

        # Upper bounds must be unaffected by α (α only affects lower relaxation slope)
        for k in eachindex(result.crown_base)
            for j in eachindex(result.crown_base[k].upper)
                @test isapprox(result.crown_opt[k].upper[j],
                               result.crown_base[k].upper[j]; atol=1e-9)
            end
        end
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Test 14: TORA — alpha optimisation benchmark
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Alpha opt benchmark: TORA" begin
        net_path  = joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                             "controllerTORA.nnet")
        @assert isfile(net_path) "Expected network file not found: $net_path"
        net       = read_nnet(net_path; last_layer_activation=Id())
        input_box = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5],
                                   high=[0.7, -0.6, -0.3, 0.6])

        result = benchmark_alpha_opt(net, input_box, "TORA";
                                     n_samples=500, n_iter=50, lr=0.1)

        @test result.max_lb_violation ≤ 1e-9
        @test result.max_ub_violation ≤ 1e-9

        for k in eachindex(result.crown_base)
            for j in eachindex(result.crown_base[k].lower)
                δ = result.crown_opt[k].lower[j] - result.crown_base[k].lower[j]
                @test δ ≥ -1e-9
            end
        end

        for k in eachindex(result.crown_base)
            for j in eachindex(result.crown_base[k].upper)
                @test isapprox(result.crown_opt[k].upper[j],
                               result.crown_base[k].upper[j]; atol=1e-9)
            end
        end
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Test 15: Unicycle — alpha optimisation benchmark
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Alpha opt benchmark: Unicycle" begin
        net_path  = joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                             "controllerUnicycle.nnet")
        @assert isfile(net_path) "Expected network file not found: $net_path"
        net       = read_nnet(net_path; last_layer_activation=Id())
        input_box = Hyperrectangle(low=[9.50, -4.50, 2.10, 1.50],
                                   high=[9.55, -4.45, 2.11, 1.51])

        result = benchmark_alpha_opt(net, input_box, "Unicycle";
                                     n_samples=500, n_iter=50, lr=0.1)

        @test result.max_lb_violation ≤ 1e-9
        @test result.max_ub_violation ≤ 1e-9

        for k in eachindex(result.crown_base)
            for j in eachindex(result.crown_base[k].lower)
                δ = result.crown_opt[k].lower[j] - result.crown_base[k].lower[j]
                @test δ ≥ -1e-9
            end
        end

        for k in eachindex(result.crown_base)
            for j in eachindex(result.crown_base[k].upper)
                @test isapprox(result.crown_opt[k].upper[j],
                               result.crown_base[k].upper[j]; atol=1e-9)
            end
        end
    end

    # ──────────────────────────────────────────────────────────────────────────
    # Test 16: ACC — alpha optimisation benchmark
    # ──────────────────────────────────────────────────────────────────────────
    @testset "Alpha opt benchmark: ACC" begin
        net_path  = joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                             "controllerACC.nnet")
        @assert isfile(net_path) "Expected network file not found: $net_path"
        net       = read_nnet(net_path; last_layer_activation=Id())
        ϵ_acc = 1e-8
        input_box = Hyperrectangle(
            low  = [30.0 - ϵ_acc, 1.40 - ϵ_acc, 30.0 - ϵ_acc, 79.0,  1.8],
            high = [30.0 + ϵ_acc, 1.40 + ϵ_acc, 30.2 + ϵ_acc, 100.0, 2.2],
        )

        result = benchmark_alpha_opt(net, input_box, "ACC";
                                     n_samples=500, n_iter=50, lr=0.1)

        @test result.max_lb_violation ≤ 1e-9
        @test result.max_ub_violation ≤ 1e-9

        for k in eachindex(result.crown_base)
            for j in eachindex(result.crown_base[k].lower)
                δ = result.crown_opt[k].lower[j] - result.crown_base[k].lower[j]
                @test δ ≥ -1e-9
            end
        end

        for k in eachindex(result.crown_base)
            for j in eachindex(result.crown_base[k].upper)
                @test isapprox(result.crown_opt[k].upper[j],
                               result.crown_base[k].upper[j]; atol=1e-9)
            end
        end
    end

end  # @testset "Alpha optimisation benchmark on real networks"

# ──────────────────────────────────────────────────────────────────────────────
# Tests 17–20: CROWN back-substitution
#
# Test 17: Correctness on small synthetic network (tightness + soundness).
# Test 18: Provably-tighter example — back-sub must strictly beat interval CROWN.
# Test 19: Real networks — back-sub ≤ interval CROWN per neuron (tightness).
# Test 20: Back-sub + alpha optimisation — result tighter than back-sub with α=0.
# ──────────────────────────────────────────────────────────────────────────────

@testset "CROWN back-substitution" begin

    # ─────────────────────────────────────────────────────────────────────────
    # Test 17: Small 2-layer synthetic network — tightness vs interval CROWN
    #          and soundness via 1000 random samples.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Back-sub tighter/equal than interval CROWN on 2-layer net" begin
        # Same 2-layer network as Test 2.
        W1 = [ 1.0  2.0;
              -1.0  1.0;
               0.5 -0.5;
               2.0  0.5]
        b1 = [0.1, -0.3, 0.2, -0.5]
        W2 = [1.0 -1.0  0.5  0.2;
              0.3  0.7 -0.4  1.0]
        b2 = [0.0, 0.1]

        net = make_network(W1, b1, ReLU(), W2, b2, Id())
        input_box = Hyperrectangle(low=[-1.0, -1.0], high=[1.0, 1.0])

        crown_interval = forward_crown(net, input_box)
        crown_backsub  = forward_crown_backsub(net, input_box)

        @test length(crown_backsub) == length(net.layers)

        # Tightness: back-sub bounds must be at least as tight as interval CROWN.
        # Width of [l, u] is u - l; smaller is tighter.
        for k in 1:length(net.layers)
            for j in eachindex(crown_interval[k].lower)
                width_interval = crown_interval[k].upper[j] - crown_interval[k].lower[j]
                width_backsub  = crown_backsub[k].upper[j]  - crown_backsub[k].lower[j]
                # Back-sub width must not exceed interval CROWN width (tol for fp rounding).
                @test width_backsub ≤ width_interval + 1e-8
            end
        end

        # Soundness: 1000 random samples must lie within back-sub bounds.
        n_samples = 1000
        lo = low(input_box);  hi = high(input_box);  n_in = length(lo)
        for _ in 1:n_samples
            x0 = lo .+ rand(n_in) .* (hi .- lo)
            pre_acts = true_preactivations(net, x0)
            for k in 1:length(net.layers)
                for j in eachindex(pre_acts[k])
                    @test pre_acts[k][j] ≥ crown_backsub[k].lower[j] - 1e-9
                    @test pre_acts[k][j] ≤ crown_backsub[k].upper[j] + 1e-9
                end
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 18: Provably-tighter example
    #
    # Network: 1 input x₀ ∈ [0, 1]
    #   Layer 1 (ReLU): z₁ = ReLU(x₀)    (W1 = [1], b1 = [0])
    #                   z₂ = ReLU(-x₀ + 0.5)  (W1 second row: [-1], b1 = [0.5])
    #   Layer 2 (Id):   output = z₂   (W2 = [0, 1], b2 = [0])
    #
    # For the output neuron (layer 2, neuron 1 = z₂):
    #   Interval CROWN: z₁ ∈ [0, 1], z₂ ∈ [0, 0.5] → pre-act = z₂ → upper = 0.5
    #   Back-sub: ẑ₂ = -x₀ + 0.5. Neuron 2 at layer 1: l̂ = -1+0.5 = -0.5, û = 0+0.5 = 0.5
    #     Upper relaxation: slope = 0.5/(0.5-(-0.5)) = 0.5, intercept = 0.5*0.5/1.0 = 0.25
    #     z₂ ≤ 0.5·ẑ₂ + 0.25 = 0.5·(-x₀+0.5) + 0.25 = -0.5x₀ + 0.5
    #     max over x₀ ∈ [0,1]: at x₀=0 → 0.5.  Wait — same bound.
    #
    # Use a different construction where back-sub provably wins:
    #   Layer 1 (Id):  z₁ = x₀, z₂ = -x₀ + 0.5   (no ReLU → exact identity)
    #   Layer 2 (Id):  output_1 = z₁ + z₂ = x₀ + (-x₀ + 0.5) = 0.5   (exact)
    #
    # Interval CROWN (without back-sub):
    #   z₁ ∈ [0, 1], z₂ ∈ [-0.5, 0.5]  → output_1 ∈ [-0.5, 1.5]  (width 2)
    # Back-sub (propagates through Identity layers exactly):
    #   output_1 = 0.5 exactly → bounds [0.5, 0.5]  (width 0)
    #
    # This is a clear win for back-sub because Id layers don't widen bounds in
    # back-sub (Λ passes through exactly), but interval CROWN accumulates width
    # from the two separate interval inputs [0,1] and [-0.5, 0.5].
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Back-sub strictly tighter: cancellation through Id layers" begin
        # x₀ ∈ [0, 1], 1D input
        # Layer 1 (Id, 2 neurons):  z₁ = x₀,  z₂ = -x₀ + 0.5
        W1_canc = reshape([1.0; -1.0], 2, 1)   # 2×1
        b1_canc = [0.0, 0.5]
        # Layer 2 (Id, 1 neuron): output = z₁ + z₂
        W2_canc = reshape([1.0, 1.0], 1, 2)    # 1×2
        b2_canc = [0.0]

        net_canc  = make_network(W1_canc, b1_canc, Id(), W2_canc, b2_canc, Id())
        input_box = Hyperrectangle(low=[0.0], high=[1.0])

        crown_int = forward_crown(net_canc, input_box)
        crown_bs  = forward_crown_backsub(net_canc, input_box)

        # Interval CROWN at layer 2 should give a wide bound (width ≥ 1).
        width_interval_L2 = crown_int[2].upper[1] - crown_int[2].lower[1]
        @test width_interval_L2 ≥ 1.0

        # Back-sub at layer 2 should give the exact value 0.5 (width = 0).
        @test isapprox(crown_bs[2].lower[1], 0.5; atol=1e-10)
        @test isapprox(crown_bs[2].upper[1], 0.5; atol=1e-10)

        # Soundness: must contain all realisable values (which is exactly {0.5}).
        lo = low(input_box);  hi = high(input_box)
        for _ in 1:200
            x0 = [lo[1] + rand() * (hi[1] - lo[1])]
            pre_acts = true_preactivations(net_canc, x0)
            for k in 1:2
                for j in eachindex(pre_acts[k])
                    @test pre_acts[k][j] ≥ crown_bs[k].lower[j] - 1e-9
                    @test pre_acts[k][j] ≤ crown_bs[k].upper[j] + 1e-9
                end
            end
        end

        # Back-sub strictly tighter than interval CROWN at layer 2.
        @test crown_bs[2].upper[1] - crown_bs[2].lower[1]  <
              crown_int[2].upper[1] - crown_int[2].lower[1] - 0.5
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 19: Real networks — back-sub ≤ interval CROWN per neuron (tightness)
    #          and soundness via 500 random samples.
    # ─────────────────────────────────────────────────────────────────────────

    """
        compare_backsub_vs_interval(net, input_box; n_samples=500, tol=1e-8)

    Compare back-substitution CROWN, interval CROWN, and MaxSens on `net`.
    Returns per-layer average bound widths (post-activation) and soundness violation.
    """
    function compare_backsub_vs_interval(net::Network, input_box::Hyperrectangle;
                                          n_samples::Int=500, tol::Float64=1e-8)
        n_layers = length(net.layers)

        # Post-activation bounds (same format as get_bounds)
        bounds_backsub   = get_bounds_crown_backsub(net, input_box)
        bounds_interval  = get_bounds_crown(net, input_box)
        bounds_maxsens   = get_bounds(net, input_box)

        # Pre-activation bounds for soundness check
        pre_backsub = forward_crown_backsub(net, input_box)

        # Per-layer average widths
        avg_width_backsub  = Float64[]
        avg_width_interval = Float64[]
        avg_width_maxsens  = Float64[]
        tighter_vs_interval = Bool[]
        tighter_vs_maxsens  = Bool[]

        for k in 2:n_layers+1
            w_bs = high(bounds_backsub[k])   .- low(bounds_backsub[k])
            w_iv = high(bounds_interval[k])  .- low(bounds_interval[k])
            w_ms = high(bounds_maxsens[k])   .- low(bounds_maxsens[k])
            push!(avg_width_backsub,  mean_vec(w_bs))
            push!(avg_width_interval, mean_vec(w_iv))
            push!(avg_width_maxsens,  mean_vec(w_ms))
            push!(tighter_vs_interval, all(w_bs .- w_iv .<= tol))
            push!(tighter_vs_maxsens,  all(w_bs .- w_ms .<= tol))
        end

        # Soundness check via sampling
        lo = low(input_box);  hi = high(input_box);  n_in = length(lo)
        max_violation = 0.0
        for _ in 1:n_samples
            x0 = lo .+ rand(n_in) .* (hi .- lo)
            pre_acts = true_preactivations(net, x0)
            for k in 1:n_layers
                for j in eachindex(pre_acts[k])
                    viol_lo = pre_backsub[k].lower[j] - pre_acts[k][j]
                    viol_hi = pre_acts[k][j] - pre_backsub[k].upper[j]
                    max_violation = max(max_violation, viol_lo, viol_hi)
                end
            end
        end

        return (
            avg_width_backsub   = avg_width_backsub,
            avg_width_interval  = avg_width_interval,
            avg_width_maxsens   = avg_width_maxsens,
            tighter_vs_interval = tighter_vs_interval,
            tighter_vs_maxsens  = tighter_vs_maxsens,
            max_soundness_violation = max_violation,
        )
    end

    repo_root = joinpath(@__DIR__, "..")

    @testset "Back-sub vs interval CROWN on real networks" begin
        # Network configurations (same as Tests 9-12)
        networks_cfg = [
            ("Single Pendulum",
             joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                      "controllerSinglePendulum.nnet"),
             Hyperrectangle(low=[1.0, 0.0], high=[1.2, 0.2])),
            ("TORA",
             joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                      "controllerTORA.nnet"),
             Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5],
                            high=[0.7, -0.6, -0.3, 0.6])),
            ("Unicycle",
             joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                      "controllerUnicycle.nnet"),
             Hyperrectangle(low=[9.50, -4.50, 2.10, 1.50],
                            high=[9.55, -4.45, 2.11, 1.51])),
            ("ACC",
             joinpath(repo_root, "Networks", "ARCH-COMP-2023", "nnet",
                      "controllerACC.nnet"),
             let ϵ = 1e-8
                 Hyperrectangle(
                     low  = [30.0 - ϵ, 1.40 - ϵ, 30.0 - ϵ, 79.0,  1.8],
                     high = [30.0 + ϵ, 1.40 + ϵ, 30.2 + ϵ, 100.0, 2.2])
             end),
        ]

        for (name, net_path, input_box) in networks_cfg
            @testset "$name" begin
                @assert isfile(net_path) "Expected network file not found: $net_path"
                net = read_nnet(net_path; last_layer_activation=Id())

                result = compare_backsub_vs_interval(net, input_box;
                                                      n_samples=500, tol=1e-8)

                println("\n--- $name (back-sub vs interval CROWN vs MaxSens) ---")
                for (k, (wbs, wiv, wms, t_iv, t_ms)) in enumerate(zip(
                        result.avg_width_backsub, result.avg_width_interval,
                        result.avg_width_maxsens, result.tighter_vs_interval,
                        result.tighter_vs_maxsens))
                    pct_vs_interval = wiv > 0.0 ? 100.0 * (wiv - wbs) / wiv : 0.0
                    println("  Layer $k: backsub=$(round(wbs, digits=6))  " *
                            "interval=$(round(wiv, digits=6))  " *
                            "maxsens=$(round(wms, digits=6))  " *
                            "improvement_vs_interval=$(round(pct_vs_interval, digits=2))%  " *
                            "tighter_vs_interval=$t_iv")
                end
                println("  Max soundness violation: $(result.max_soundness_violation)")

                # Soundness: no back-sub bound violated by any sampled input
                @test result.max_soundness_violation ≤ 1e-9

                # Tightness: back-sub ≤ interval CROWN per neuron (within tolerance)
                @test all(result.tighter_vs_interval)

                # Back-sub is also tighter than or equal to MaxSens
                @test all(result.tighter_vs_maxsens)
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Test 20: Back-sub + alpha optimisation is tighter than back-sub with α=0
    #          Uses the same 2-layer synthetic network as Test 2.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Back-sub with optimised alpha vs alpha=0" begin
        W1 = [ 1.0  2.0;
              -1.0  1.0;
               0.5 -0.5;
               2.0  0.5]
        b1 = [0.1, -0.3, 0.2, -0.5]
        W2 = [1.0 -1.0  0.5  0.2;
              0.3  0.7 -0.4  1.0]
        b2 = [0.0, 0.1]

        net = make_network(W1, b1, ReLU(), W2, b2, Id())
        input_box = Hyperrectangle(low=[-1.0, -1.0], high=[1.0, 1.0])

        # Back-sub with α=0 (default)
        bs_base = forward_crown_backsub(net, input_box)

        # Optimise α then run back-sub with optimised α
        alpha_opt = optimise_alpha(net, input_box; n_iter=30, lr=0.1)
        bs_opt    = forward_crown_backsub(net, input_box; alpha=alpha_opt)

        # α optimisation should tighten or preserve lower bounds (never loosen).
        for k in 1:length(net.layers)
            for j in eachindex(bs_base[k].lower)
                @test bs_opt[k].lower[j] ≥ bs_base[k].lower[j] - 1e-9
                # Upper bounds must be unaffected by α.
                @test isapprox(bs_opt[k].upper[j], bs_base[k].upper[j]; atol=1e-9)
            end
        end

        # Soundness of back-sub + alpha_opt via 500 random samples.
        lo = low(input_box);  hi = high(input_box);  n_in = length(lo)
        for _ in 1:500
            x0 = lo .+ rand(n_in) .* (hi .- lo)
            pre_acts = true_preactivations(net, x0)
            for k in 1:length(net.layers)
                for j in eachindex(pre_acts[k])
                    @test pre_acts[k][j] ≥ bs_opt[k].lower[j] - 1e-9
                    @test pre_acts[k][j] ≤ bs_opt[k].upper[j] + 1e-9
                end
            end
        end

        println("\n--- Back-sub + alpha opt vs back-sub baseline (2-layer synthetic) ---")
        for k in 1:length(net.layers)
            delta_lower = bs_opt[k].lower .- bs_base[k].lower
            println("  Layer $k: max Δlower = $(maximum(delta_lower)), " *
                    "mean Δlower = $(mean_vec(delta_lower))")
        end
    end

end  # @testset "CROWN back-substitution"

println("\nAll alpha-CROWN tests passed.")
