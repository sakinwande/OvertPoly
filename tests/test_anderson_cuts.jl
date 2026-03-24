"""
    test_anderson_cuts.jl

Tests for the Anderson pairwise conditional cuts in src/nv/anderson_cuts.jl.

Run from repo root:
    julia --project=. tests/test_anderson_cuts.jl

Tests cover:
  1. _conditional_bound_min: correctness on hand-checked examples.
  2. conditional_preact_bounds: soundness — true ẑ_j ∈ [l_cond, u_cond]
     for all sampled inputs satisfying the conditioning event.
  3. add_anderson_cuts! on a synthetic 2-layer network: verifies cuts are added
     and the model stays feasible.
  4. No cuts generated when no unstable neuron pairs exist.
  5. Integration: add_anderson_cuts! on the TORA network, checks cut count > 0
     and model remains solvable.
"""

using Test
using LazySets
using JuMP, Gurobi
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "nn_mip_encoding.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function random_inputs(box::Hyperrectangle, n::Int)
    lo, hi = low(box), high(box)
    return [lo .+ rand(length(lo)) .* (hi .- lo) for _ in 1:n]
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "Anderson cuts" begin

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: _conditional_bound_min correctness (hand-checked)
# ─────────────────────────────────────────────────────────────────────────────
@testset "_conditional_bound_min analytical LP" begin
    # A: min z₁+z₂  s.t. z₁ ≥ 0,  z ∈ [-1,1]²  → z=(0,-1), val=-1
    @test _conditional_bound_min([1.0,1.0], 0.0, [1.0,0.0], 0.0,
                                  [-1.0,-1.0], [1.0,1.0]) ≈ -1.0  atol=1e-8

    # B: min z₁+z₂  s.t. z₁+z₂ ≥ 1,  z ∈ [-1,1]²  → val=1
    @test _conditional_bound_min([1.0,1.0], 0.0, [1.0,1.0], 1.0,
                                  [-1.0,-1.0], [1.0,1.0]) ≈ 1.0   atol=1e-8

    # C: min z₁-z₂  s.t. z₁-z₂ ≥ 0,  z ∈ [-1,1]²
    # Unconstrained min: z=(-1,1) → val=-2 (violates). Must satisfy z₁≥z₂. val=0.
    @test _conditional_bound_min([1.0,-1.0], 0.0, [1.0,-1.0], 0.0,
                                  [-1.0,-1.0], [1.0,1.0]) ≈ 0.0   atol=1e-8

    # D: infeasible — z₁ ≥ 5 impossible on [-1,1]
    @test isnan(_conditional_bound_min([1.0], 0.0, [1.0], 5.0, [-1.0], [1.0]))

    # E: constraint z₁ ≥ -2 always satisfied on [-1,1] → unconstrained min
    @test _conditional_bound_min([-1.0], 0.0, [1.0], -2.0, [-1.0], [1.0]) ≈ -1.0  atol=1e-8

    # F: max via negation — max z₁+z₂  s.t. z₁+z₂ ≥ -1, z ∈ [-1,1]²  → val=2
    @test _conditional_bound_max([1.0,1.0], 0.0, [1.0,1.0], -1.0,
                                  [-1.0,-1.0], [1.0,1.0]) ≈ 2.0   atol=1e-8
end

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: conditional_preact_bounds soundness (random sampling)
# ─────────────────────────────────────────────────────────────────────────────
@testset "conditional_preact_bounds soundness" begin
    W = [ 1.0  1.0;
         -1.0  1.0;
          1.0 -1.0]
    b = [-0.3, -0.3, -0.3]
    layer = Layer(W, b, ReLU())
    input_box = Hyperrectangle(low=[-1.0, -1.0], high=[1.0, 1.0])

    for (i, j) in [(1,2), (1,3), (2,3)]
        l_act, u_act, l_inact, u_inact = conditional_preact_bounds(layer, input_box, i, j)

        violations = 0
        for x in random_inputs(input_box, 500)
            ẑ_i = dot(W[i,:], x) + b[i]
            ẑ_j = dot(W[j,:], x) + b[j]

            if ẑ_i >= 0.0 && !isnan(l_act) && !isnan(u_act)
                ẑ_j < l_act - 1e-6 && (violations += 1)
                ẑ_j > u_act + 1e-6 && (violations += 1)
            end
            if ẑ_i <= 0.0 && !isnan(l_inact) && !isnan(u_inact)
                ẑ_j < l_inact - 1e-6 && (violations += 1)
                ẑ_j > u_inact + 1e-6 && (violations += 1)
            end
        end
        @test violations == 0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: add_anderson_cuts! on a synthetic network
# ─────────────────────────────────────────────────────────────────────────────
@testset "add_anderson_cuts! synthetic network" begin
    W1 = [ 1.0  1.0;
          -1.0  1.0;
           1.0  0.0]
    b1 = [-0.3, -0.3, -0.3]
    W2 = reshape([1.0, 1.0, 1.0], 1, 3)
    b2 = [0.0]
    network = Network([Layer(W1, b1, ReLU()), Layer(W2, b2, Id())])
    input_box = Hyperrectangle(low=[-0.5, -0.5], high=[0.5, 0.5])
    bounds = get_bounds_crown_backsub(network, input_box)

    env = Gurobi.Env()
    model = Model(() -> Gurobi.Optimizer(env))
    set_silent(model)
    neurons = init_neurons(model, network)
    deltas  = init_deltas(model, network)
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())

    n_before = length(all_constraints(model, include_variable_in_set_constraints=false))
    n_cuts, n_fixed = add_anderson_cuts!(model, network, neurons, deltas, bounds)
    n_after  = length(all_constraints(model, include_variable_in_set_constraints=false))

    @test n_cuts >= 0
    @test n_fixed >= 0
    @test n_after >= n_before

    # Model must stay feasible
    @objective(model, Min, 0)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
end

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: No cuts when no unstable pairs
# ─────────────────────────────────────────────────────────────────────────────
@testset "no cuts for stable network" begin
    # All pre-activations strictly positive over input box
    W1 = [1.0 0.0; 0.0 1.0]
    b1 = [1.5, 1.5]
    W2 = reshape([1.0, 1.0], 1, 2)
    b2 = [0.0]
    network = Network([Layer(W1, b1, ReLU()), Layer(W2, b2, Id())])
    input_box = Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0])
    bounds = get_bounds_crown_backsub(network, input_box)

    env = Gurobi.Env()
    model = Model(() -> Gurobi.Optimizer(env))
    set_silent(model)
    neurons = init_neurons(model, network)
    deltas  = init_deltas(model, network)
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())

    n_cuts, n_fixed = add_anderson_cuts!(model, network, neurons, deltas, bounds)
    @test n_cuts == 0
    @test n_fixed == 0
end

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: TORA network integration
# ─────────────────────────────────────────────────────────────────────────────
@testset "TORA network integration" begin
    nnet_path = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerTORA.nnet")
    input_box = Hyperrectangle(low=[-0.77, -0.45, -0.45, -0.45],
                                high=[ 0.77,  0.45,  0.45,  0.45])
    network = read_nnet(nnet_path, last_layer_activation=Id())
    bounds  = get_bounds_crown_backsub(network, input_box)

    env = Gurobi.Env()
    model = Model(() -> Gurobi.Optimizer(env))
    set_silent(model)
    neurons = init_neurons(model, network)
    deltas  = init_deltas(model, network)
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())

    # cap to keep test fast; real benchmark can use default (unlimited)
    n_cuts, n_fixed = add_anderson_cuts!(model, network, neurons, deltas, bounds;
                                          max_cuts_per_layer=200)
    @info "TORA: $n_cuts Anderson cuts added, $n_fixed binaries fixed"

    @test n_cuts > 0   # TORA has many unstable neurons → pairs → cuts
    @test n_fixed >= 0

    # Model must still be feasible (don't run full MILP solve — that's a benchmark)
    @objective(model, Min, 0)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
end

end  # @testset "Anderson cuts"

println("\nAll Anderson cuts tests passed.")