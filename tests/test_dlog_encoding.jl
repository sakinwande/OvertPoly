"""
    test_dlog_encoding.jl

Tests for the DLOG (Disaggregated Logarithmic Convex Combination) encoding
implemented in src/overtPoly_to_mip.jl.

Run from repo root with:
    julia --project=. tests/test_dlog_encoding.jl

Tests cover:
  1. _dlog_assign_codes: n=1 edge case (K_total=0, no binary variables)
  2. _dlog_assign_codes: n=2 single cell (K_total=1, distinct codes)
  3. _dlog_assign_codes: 3x3 grid, 8 simplices — injectivity + Gray adjacency
  4. dlogEncoding! (GraphPolyQuery): binary count = K_total ≪ n
  5. dlogEncoding! vs ccEncoding!: same optimal value on identical problem
  6. dlogEncoding!: MILP solution lies within the triangulation domain
"""

using Test
using JuMP
using Gurobi
using LazySets
using MathOptInterface
const MOI = MathOptInterface

include(joinpath(@__DIR__, "..", "src", "overtPoly_to_mip.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""Count binary (ZeroOne) variables in a JuMP model."""
function count_binary_vars(model)
    return length(all_constraints(model, VariableRef, MOI.ZeroOne))
end

"""
Build a minimal GraphPolyQuery suitable for testing dlogEncoding! / ccEncoding!.
uCoef_val: scalar control coefficient (0.0 ⟹ no control influence on y).
"""
function make_graph_query(uCoef_val=0.0)
    problem = GraphPolyProblem(
        nothing, nothing,           # expr, dec_expr
        [[uCoef_val]],              # control_coef: control_coef[1][1] = uCoef_val
        Hyperrectangle(low=[0.0, 0.0], high=[2.0, 2.0]),
        [:f1],
        [nothing],
        x -> x,
        (a, b) -> nothing,
        x -> nothing,
        x -> nothing
    )
    return GraphPolyQuery(
        problem,
        nothing,              # network_file
        nothing,              # last_layer_activation
        "Gurobi",
        1, 1.0, 1,            # ntime, dt, N_overt
        Dict{Symbol,Any}(),   # var_dict
        Dict{Symbol,Any}(),   # mod_dict
        nothing               # case
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────────────────────

# 2D unit square: 4 vertices, 2 simplices in a single cell.
# Vertices: 1=(0,0), 2=(1,0), 3=(0,1), 4=(1,1)
const xS_2simplex = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
const Tri_2simplex = Vector{Int32}[[1,2,4], [1,3,4]]
const yVals_2simplex = [0.0, 1.0, 1.0, 2.0]   # f = x1 + x2 at each vertex

# 3x3 grid: 9 vertices, 8 simplices across 4 cells (2 per cell).
# Vertices: xS_3x3[k] = (x1, x2) for x1, x2 ∈ {0.0, 1.0, 2.0}
# Ordering: x1 varies in outer loop → (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
const xS_3x3 = Tuple{Float64,Float64}[(float(x), float(y)) for x in 0:2 for y in 0:2]
const Tri_3x3 = Vector{Int32}[
    [1,2,5], [1,4,5],   # cell (0,0): x1∈[0,1], x2∈[0,1]
    [2,3,6], [2,5,6],   # cell (0,1): x1∈[0,1], x2∈[1,2]
    [4,5,8], [4,7,8],   # cell (1,0): x1∈[1,2], x2∈[0,1]
    [5,6,9], [5,8,9],   # cell (1,1): x1∈[1,2], x2∈[1,2]
]
const yVals_3x3 = [x + y for (x, y) in xS_3x3]  # f = x1 + x2

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "DLOG encoding" begin

    # ─────────────────────────────────────────────────────────────────────────
    # 1. _dlog_assign_codes: n=1 → K_total=0
    # ─────────────────────────────────────────────────────────────────────────
    @testset "_dlog_assign_codes: n=1 edge case" begin
        xS  = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        Tri = [Int32[1, 2, 3]]
        codes, K = _dlog_assign_codes(xS, Tri)

        @test K == 0                   # no binary variables needed
        @test size(codes) == (1, 0)    # 1 simplex, 0 bits
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 2. _dlog_assign_codes: n=2, single cell → K_total=1
    # ─────────────────────────────────────────────────────────────────────────
    @testset "_dlog_assign_codes: n=2 single cell" begin
        codes, K = _dlog_assign_codes(xS_2simplex, Tri_2simplex)

        # 1 cell in each dimension (only one interval per axis) → K_dims=[0,0]
        # 2 sub-simplices per cell → K_sub = ceil(log2(2)) = 1
        @test K == 1
        @test size(codes) == (2, 1)

        # All entries must be binary
        @test all(c ∈ (0, 1) for c in codes)

        # Codes must be distinct (injectivity)
        @test codes[1, :] != codes[2, :]
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 3. _dlog_assign_codes: 3x3 grid — shape, injectivity, Gray adjacency
    # ─────────────────────────────────────────────────────────────────────────
    @testset "_dlog_assign_codes: 3x3 grid (8 simplices)" begin
        codes, K = _dlog_assign_codes(xS_3x3, Tri_3x3)
        n = length(Tri_3x3)  # 8

        # Grid has 2 intervals per axis → K_dims=[1,1]; 2 sub-simplices/cell → K_sub=1
        @test K == 3
        @test size(codes) == (8, 3)
        @test all(c ∈ (0, 1) for c in codes)

        # Injectivity: all codes distinct
        code_rows = [codes[i, :] for i in 1:n]
        @test length(unique(code_rows)) == n

        # Gray adjacency: simplices in the same cell share the same cell bits.
        # Simplices 1 and 2 are both in cell (0,0):
        @test codes[1, 1:2] == codes[2, 1:2]   # same dim-1 and dim-2 bits
        @test codes[1, 3]   != codes[2, 3]      # but different sub-simplex bit

        # Simplices 3 and 4 are both in cell (0,1):
        @test codes[3, 1:2] == codes[4, 1:2]

        # Gray adjacency across cells: cells (0,0) and (1,0) are adjacent in dim-1.
        # Their dim-1 bits (bit 1) must differ; dim-2 bits (bit 2) must agree.
        # Simplex 1 ∈ cell(0,0), simplex 5 ∈ cell(1,0):
        @test codes[1, 1] != codes[5, 1]   # dim-1 Gray bit differs
        @test codes[1, 2] == codes[5, 2]   # dim-2 Gray bit same

        # Cells (0,0) and (0,1) adjacent in dim-2:
        # Simplex 1 ∈ cell(0,0), simplex 3 ∈ cell(0,1):
        @test codes[1, 1] == codes[3, 1]   # dim-1 Gray bit same
        @test codes[1, 2] != codes[3, 2]   # dim-2 Gray bit differs
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 4. dlogEncoding!: binary variable count = K_total ≪ n
    # ─────────────────────────────────────────────────────────────────────────
    @testset "dlogEncoding! binary count = K_total" begin
        optimizer = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
        model = Model(optimizer)
        query = make_graph_query(0.0)

        dlogEncoding!(xS_3x3, yVals_3x3, yVals_3x3, Tri_3x3, query, :f1, 1, model)

        _, K_total = _dlog_assign_codes(xS_3x3, Tri_3x3)
        n_bin = count_binary_vars(model)

        @test n_bin == K_total               # exactly K_total binary vars
        @test K_total < length(Tri_3x3)      # logarithmic reduction (3 vs 8)
        @test K_total == 3
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 5. dlogEncoding! vs ccEncoding!: same optimal value, DLOG uses fewer binaries
    # ─────────────────────────────────────────────────────────────────────────
    @testset "dlogEncoding! and ccEncoding! agree on optimal value" begin
        optimizer = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

        # DLOG model: maximise y
        model_dlog  = Model(optimizer)
        query_dlog  = make_graph_query(0.0)
        dlogEncoding!(xS_3x3, yVals_3x3, yVals_3x3, Tri_3x3, query_dlog, :f1, 1, model_dlog)
        y_dlog = query_dlog.var_dict[:f1][2][1]
        @objective(model_dlog, Max, y_dlog)
        optimize!(model_dlog)

        # CC model: same problem
        model_cc   = Model(optimizer)
        query_cc   = make_graph_query(0.0)
        ccEncoding!(xS_3x3, yVals_3x3, yVals_3x3, Tri_3x3, query_cc, :f1, 1, model_cc)
        y_cc = query_cc.var_dict[:f1][2][1]
        @objective(model_cc, Max, y_cc)
        optimize!(model_cc)

        @test termination_status(model_dlog) == MOI.OPTIMAL
        @test termination_status(model_cc)   == MOI.OPTIMAL

        # Both must achieve the same optimal value (max of f = x1+x2 = 4.0 at (2,2))
        @test objective_value(model_dlog) ≈ objective_value(model_cc)  atol=1e-6
        @test objective_value(model_dlog) ≈ maximum(yVals_3x3)         atol=1e-6

        # DLOG must use strictly fewer binary variables
        @test count_binary_vars(model_dlog) < count_binary_vars(model_cc)
        @test count_binary_vars(model_cc)   == length(Tri_3x3)   # CC uses n=8 binaries
        @test count_binary_vars(model_dlog) == 3                  # DLOG uses K_total=3
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 6. dlogEncoding!: solution x lies within triangulation, y within bounds
    # ─────────────────────────────────────────────────────────────────────────
    @testset "dlogEncoding! solution is within triangulation domain" begin
        optimizer = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
        model = Model(optimizer)
        query = make_graph_query(0.0)

        dlogEncoding!(xS_3x3, yVals_3x3, yVals_3x3, Tri_3x3, query, :f1, 1, model)
        y_var = query.var_dict[:f1][2][1]
        @objective(model, Max, y_var)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL

        x_val = value.(query.var_dict[:f1][1])
        y_val = objective_value(model)

        tol = 1e-6
        @test all(x_val .>= minimum(xS_3x3[i][k] for i in eachindex(xS_3x3) for k in 1:2) - tol)
        @test all(x_val .<= maximum(xS_3x3[i][k] for i in eachindex(xS_3x3) for k in 1:2) + tol)
        @test y_val >= minimum(yVals_3x3) - tol
        @test y_val <= maximum(yVals_3x3) + tol
    end

end
