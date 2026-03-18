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

println("\nAll alpha-CROWN tests passed.")
