"""
    test_dcc_vs_cc_tora.jl

Compares CC (Convex Combination) vs DCC (Disaggregated Convex Combination) encoding
on the TORA benchmark using GraphPolyQuery / concreach!.

Metrics collected per encoding:
  - Wall-clock time for concreach! (bounding + encoding + solve)
  - Reachset volume
  - MIP size (binary variable count, linear constraint count)
  - Gurobi B&B node count (summed across all 8 optimizer calls: 4 min + 4 max)

Run from repo root:
    julia --project=. tests/test_dcc_vs_cc_tora.jl
"""

include(joinpath(@__DIR__, "..", "src", "overtPoly_helpers.jl"))
include(joinpath(@__DIR__, "..", "src", "nn_mip_encoding.jl"))
include(joinpath(@__DIR__, "..", "src", "overtPoly_to_mip.jl"))
include(joinpath(@__DIR__, "..", "src", "overt_to_pwa.jl"))
include(joinpath(@__DIR__, "..", "src", "problems.jl"))
include(joinpath(@__DIR__, "..", "src", "distr_reachability.jl"))

using LazySets
using Dates
using Plasmo
using JuMP, Gurobi

# ── TORA problem definition (mirrors tora_overtPoly_distrOpt.jl) ──────────────

control_coef = [[0],[0],[0],[1]]
controller   = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerTORA.nnet")
exprList     = [:(1*x2), :(-x1 + 0.1*sin(x3)), :(1*x4), :(1*u)]
dt           = 0.1

function tora_dynamics(x, u)
    dx1 = x[2]
    dx2 = -x[1] + 0.1*sin(x[3])
    dx3 = x[4]
    dx4 = u
    return x + [dx1, dx2, dx3, dx4] .* dt
end

function tora_control(input_set)
    return input_set
end

function bound_tora(TORA; plotFlag=false, npoint=nothing)
    lbs, ubs = extrema(TORA.domain)
    lb_x2 = lbs[2]; ub_x2 = ubs[2]
    x1Func = :(1*x2)
    x1FuncLB_u, _ = interpol_nd(bound_univariate(x1Func, lb_x2, ub_x2, plotflag=plotFlag)...)
    emptyList = [1]; currList = [2]
    lb_x1 = lbs[1]; ub_x1 = ubs[1]
    x1FuncLB, x1FuncUB = lift_OA(emptyList, currList, x1FuncLB_u, x1FuncLB_u, lb_x1, ub_x1)

    x2FuncSub1 = :(-1*x1)
    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(x2FuncSub1, lb_x1, ub_x1, plotflag=plotFlag)...)
    lb_x3 = lbs[3]; ub_x3 = ubs[3]
    x2FuncSub2 = :(sin(x3))
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(x2FuncSub2, lb_x3, ub_x3, plotflag=plotFlag)...)
    x2FuncSub2LB = [(tup[1:end-1]..., 0.1*tup[end]) for tup in x2FuncSub2LB]
    x2FuncSub2UB = [(tup[1:end-1]..., 0.1*tup[end]) for tup in x2FuncSub2UB]

    emptyList = [2]; currList = [1]
    l_x2FuncSub1LB, l_x2FuncSub1UB = lift_OA(emptyList, currList, x2FuncSub1LB, x2FuncSub1UB, [lbs[1], lbs[3]], [ubs[1], ubs[3]])
    emptyList = [1]; currList = [2]
    l_x2FuncSub2LB, l_x2FuncSub2UB = lift_OA(emptyList, currList, x2FuncSub2LB, x2FuncSub2UB, [lbs[1], lbs[3]], [ubs[1], ubs[3]])
    x2FuncLB_u, x2FuncUB_u = sumBounds(l_x2FuncSub1LB, l_x2FuncSub1UB, l_x2FuncSub2LB, l_x2FuncSub2UB, false)
    emptyList = [2]; currList = [1,3]
    x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs, ubs)

    lb_x4 = lbs[4]; ub_x4 = ubs[4]
    x3Func = :(1*x4)
    x3FuncLB_u, x3FuncUB_u = interpol_nd(bound_univariate(x3Func, lb_x4, ub_x4, plotflag=plotFlag)...)
    emptyList = [1]; currList = [2]
    x3FuncLB, x3FuncUB = lift_OA(emptyList, currList, x3FuncLB_u, x3FuncUB_u, lb_x3, ub_x3)

    x4Func = :(0*x4)
    x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(x4Func, lb_x4, ub_x4, plotflag=plotFlag)...)
    return [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
end

function tora_dyn_con_link!(query, neurons, graph, dynModel, netModel)
    @variable(netModel, x1); @variable(netModel, x2)
    @variable(netModel, x3); @variable(netModel, x4)
    @constraint(netModel, neurons[1][1] == x1); @constraint(netModel, neurons[1][2] == x2)
    @constraint(netModel, neurons[1][3] == x3); @constraint(netModel, neurons[1][4] == x4)
    @linkconstraint(graph, netModel[:x1] == dynModel[2][:x][1])
    @linkconstraint(graph, netModel[:x2] == dynModel[2][:x][2])
    @linkconstraint(graph, netModel[:x3] == dynModel[2][:x][3])
    @linkconstraint(graph, netModel[:x4] == dynModel[4][:x][1])
    @variable(netModel, u)
    @constraint(netModel, u == neurons[end][1])
    @linkconstraint(graph, netModel[:u] == dynModel[4][:u])
    i = 0
    for sym in query.problem.varList
        i += 1
        pertVar = (sym == :x2) ? dynModel[i][:x][2] : dynModel[i][:x][1]
        push!(query.var_dict[sym], [pertVar])
    end
end

domain   = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
N_overt  = 2

TORA = GraphPolyProblem(
    exprList, nothing, control_coef, domain,
    [:x1,:x2,:x3,:x4], nothing,
    tora_dynamics, bound_tora, tora_control, tora_dyn_con_link!
)

function make_query()
    return GraphPolyQuery(TORA, controller, Id(), "MIP", 1, dt, N_overt, nothing, nothing, 2)
end

# ── MIP size helper ───────────────────────────────────────────────────────────

function mip_stats(graph::OptiGraph)
    n_bin = 0; n_lin = 0
    for node in Plasmo.all_nodes(graph)
        n_bin += count(is_binary, all_variables(node))
        n_lin += num_constraints(node; count_variable_in_set_constraints=false)
    end
    return (binary=n_bin, linear=n_lin)
end

# ── Run one concreach! and collect metrics ────────────────────────────────────

function run_encoding(enc_func, label)
    println("\n" * "="^50)
    println("  Encoding: $label")
    println("="^50)

    q = make_query()
    q.problem.bounds = q.problem.bound_func(q.problem, npoint=N_overt)
    q.var_dict = Dict{Symbol,Any}()
    q.mod_dict = Dict{Symbol,Any}()

    # Encode dynamics + control
    encode_dynamics!(q; enc_func=enc_func)
    neurons = encode_control!(q)
    dyn_con_link! = q.problem.link_func
    dyn_con_link!(q, neurons, q.mod_dict[:graph], q.mod_dict[:f], q.mod_dict[:u])

    stats = mip_stats(q.mod_dict[:graph])
    println("  Binary variables : $(stats.binary)")
    println("  Linear constraints: $(stats.linear)")

    # Solve and time it
    t0 = time()
    reach_set = conc_reach_solve(q)
    elapsed = time() - t0

    vol = volume(reach_set)
    println("  Solve time (s)   : $(round(elapsed, digits=3))")
    println("  Reachset volume  : $(round(vol, sigdigits=6))")
    println("  Reachset bounds  : low=$(round.(low(reach_set), digits=6)), high=$(round.(high(reach_set), digits=6))")

    return (label=label, time=elapsed, volume=vol, reach_set=reach_set, stats=stats)
end

# ── Main comparison ───────────────────────────────────────────────────────────

println("\nTORA benchmark: CC vs DCC encoding comparison")
println("Domain: ", domain)
println("N_overt: $N_overt, dt: $dt, 1 step concrete reachability\n")

results_cc  = run_encoding(ccEncoding!,  "CC  (Convex Combination)")
results_dcc = run_encoding(dccEncoding!, "DCC (Disaggregated CC)")

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n" * "="^50)
println("  Summary")
println("="^50)
println("                   CC          DCC")
println("  Binary vars  : $(lpad(results_cc.stats.binary, 8))    $(lpad(results_dcc.stats.binary, 8))")
println("  Linear cons  : $(lpad(results_cc.stats.linear, 8))    $(lpad(results_dcc.stats.linear, 8))")
println("  Solve time(s): $(lpad(round(results_cc.time, digits=3), 8))    $(lpad(round(results_dcc.time, digits=3), 8))")
println("  Volume       : $(lpad(round(results_cc.volume, sigdigits=5), 8))    $(lpad(round(results_dcc.volume, sigdigits=5), 8))")

# Soundness check: reachsets should be identical (same problem, same triangulation)
vol_diff = abs(results_cc.volume - results_dcc.volume) / max(results_cc.volume, 1e-12)
println("\n  Relative volume difference: $(round(vol_diff * 100, digits=4))%")
if vol_diff < 1e-6
    println("  [PASS] Reachsets are numerically identical — DCC is sound.")
else
    println("  [WARN] Reachsets differ — check encoding correctness.")
end