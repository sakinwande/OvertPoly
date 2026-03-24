"""
    test_dlog_unicycle.jl

Compares CC, DCC, and DLOG encoding on the Unicycle benchmark using hybrid
symbolic reachability with concInt = [5, 5].

Run from repo root:
    julia --project=. tests/test_dlog_unicycle.jl
"""

include(joinpath(@__DIR__, "..", "src", "overtPoly_helpers.jl"))
include(joinpath(@__DIR__, "..", "src", "nn_mip_encoding.jl"))
include(joinpath(@__DIR__, "..", "src", "overtPoly_to_mip.jl"))
include(joinpath(@__DIR__, "..", "src", "overt_to_pwa.jl"))
include(joinpath(@__DIR__, "..", "src", "problems.jl"))
include(joinpath(@__DIR__, "..", "src", "distr_reachability.jl"))

using LazySets
using Plasmo
using JuMP, Gurobi
using Dates

# ── Unicycle problem definition ───────────────────────────────────────────────

control_coef = [[0],[0],[1],[1]]
controller   = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerUnicycle.nnet")
exprList     = [:(cos(x3)*x4), :(sin(x3)*x4), :(0*x3), :(0*x4)]
dt           = 0.2
w            = 1e-4
dig          = 15
domain       = Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high=[9.55,-4.45,2.11,1.51])
depMat       = [[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]
concInt      = [5, 5]
N_overt      = 2

function unicycle_dynamics(x, u)
    dx1 = x[4]*cos(x[3]); dx2 = x[4]*sin(x[3])
    dx3 = u[2];            dx4 = u[1]
    return x + [dx1, dx2, dx3, dx4] .* dt
end

function unicycle_control(input_set)
    return input_set
end

function bound_unicycle(Unicycle; npoint=2)
    lbs, ubs = extrema(Unicycle.domain)
    lbs = floor.(lbs, digits=dig)
    ubs = ceil.(ubs, digits=dig)

    lb_x4 = lbs[4]; ub_x4 = ubs[4]
    lb_x3 = lbs[3]; ub_x3 = ubs[3]

    x1FuncSub_1LB, x1FuncSub_1UB = interpol_nd(bound_univariate(:(1*x4), lb_x4, ub_x4)...)
    x1FuncSub_2LB, x1FuncSub_2UB = interpol_nd(bound_univariate(:(cos(x3)), lb_x3, ub_x3, npoint=npoint)...)
    l_x1Sub1LB, l_x1Sub1UB = lift_OA([1], [2], x1FuncSub_1LB, x1FuncSub_1UB, lbs[3:4], ubs[3:4])
    l_x1Sub2LB, l_x1Sub2UB = lift_OA([2], [1], x1FuncSub_2LB, x1FuncSub_2UB, lbs[3:4], ubs[3:4])
    x1FuncLB, x1FuncUB = prodBounds(l_x1Sub1LB, l_x1Sub1UB, l_x1Sub2LB, l_x1Sub2UB)

    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(:(1*x4), lb_x4, ub_x4)...)
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(:(sin(x3)), lb_x3, ub_x3, npoint=npoint)...)
    l_x2Sub1LB, l_x2Sub1UB = lift_OA([1], [2], x2FuncSub1LB, x2FuncSub1UB, lbs[3:4], ubs[3:4])
    l_x2Sub2LB, l_x2Sub2UB = lift_OA([2], [1], x2FuncSub2LB, x2FuncSub2UB, lbs[3:4], ubs[3:4])
    x2FuncLB, x2FuncUB = prodBounds(l_x2Sub1LB, l_x2Sub1UB, l_x2Sub2LB, l_x2Sub2UB)

    x1FuncLB_u = deepcopy(x1FuncLB); x1FuncUB_u = deepcopy(x1FuncUB)
    lbs_x1 = [lbs[1]]; append!(lbs_x1, lbs[3:4])
    ubs_x1 = [ubs[1]]; append!(ubs_x1, ubs[3:4])
    x1FuncLB, x1FuncUB = lift_OA([1], [2,3], x1FuncLB_u, x1FuncUB_u, lbs_x1, ubs_x1)

    x2FuncLB_u = deepcopy(x2FuncLB); x2FuncUB_u = deepcopy(x2FuncUB)
    lbs_x2 = lbs[2:4]; ubs_x2 = ubs[2:4]
    x2FuncLB, x2FuncUB = lift_OA([1], [2,3], x2FuncLB_u, x2FuncUB_u, lbs_x2, ubs_x2)

    x3FuncLB, x3FuncUB = interpol_nd(bound_univariate(:(0*x3), lb_x3, ub_x3)...)
    x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(:(0*x4), lb_x4, ub_x4, ϵ=w)...)

    return [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
end

function unicycle_dyn_con_link!(query, neurons, graph, dynModel, netModel, t_ind=nothing)
    @variable(netModel, x1); @variable(netModel, x2)
    @variable(netModel, x3); @variable(netModel, x4)
    @constraint(netModel, neurons[1][1] == x1); @constraint(netModel, neurons[1][2] == x2)
    @constraint(netModel, neurons[1][3] == x3); @constraint(netModel, neurons[1][4] == x4)
    @linkconstraint(graph, netModel[:x1] == dynModel[1][:x][1])
    @linkconstraint(graph, netModel[:x2] == dynModel[2][:x][1])
    @linkconstraint(graph, netModel[:x3] == dynModel[3][:x][1])
    @linkconstraint(graph, netModel[:x4] == dynModel[4][:x][1])
    @variable(netModel, u1); @variable(netModel, u2)
    @constraint(netModel, u1 == neurons[end][1]); @constraint(netModel, u2 == neurons[end][2])
    @linkconstraint(graph, netModel[:u1] == dynModel[4][:u])
    @linkconstraint(graph, netModel[:u2] == dynModel[3][:u])
    i = 0
    for sym in query.problem.varList
        sym_t = isnothing(t_ind) ? sym : Meta.parse("$(sym)_$(t_ind)")
        i += 1
        push!(query.var_dict[sym_t], [dynModel[i][:x][1]])
    end
end

Unicycle = GraphPolyProblem(
    exprList, nothing, control_coef, domain,
    [:x1,:x2,:x3,:x4], nothing,
    unicycle_dynamics, bound_unicycle, unicycle_control, unicycle_dyn_con_link!
)

function make_query()
    return GraphPolyQuery(Unicycle, controller, Id(), "MIP", 2, dt, N_overt, nothing, nothing, 2)
end

# ── Run one hybrid reach and collect metrics ──────────────────────────────────

function run_encoding(enc_func, label)
    println("\n" * "="^60)
    println("  Encoding: $label")
    println("="^60)
    flush(stdout)

    q = make_query()
    t0 = time()
    hyb_sets, _ = multi_step_hybreach(q, depMat, concInt; enc_func=enc_func)
    elapsed = time() - t0

    final_set = hyb_sets[end]
    vol = volume(final_set)
    println("  Total time (s)   : $(round(elapsed, digits=3))")
    println("  Final set volume : $(round(vol, sigdigits=6))")
    println("  Final set bounds :")
    println("    low  = $(round.(low(final_set),  digits=5))")
    println("    high = $(round.(high(final_set), digits=5))")
    flush(stdout)

    return (label=label, time=elapsed, volume=vol, final_set=final_set)
end

# ── Main ──────────────────────────────────────────────────────────────────────

println("\nUnicycle benchmark: CC vs DCC vs DLOG encoding")
println("Domain  : ", domain)
println("concInt : $concInt  ($(sum(concInt)) total steps)")
println("N_overt : $N_overt,  dt: $dt")
flush(stdout)

results_cc   = run_encoding(ccEncoding!,   "CC   (Convex Combination)")
results_dcc  = run_encoding(dccEncoding!,  "DCC  (Disaggregated CC)")
results_dlog = run_encoding(dlogEncoding!, "DLOG (Disaggregated Log CC)")

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  Summary — Unicycle, concInt=$concInt, N_overt=$N_overt, dt=$dt")
println("="^60)
println("  $(rpad("Encoding", 8)) | $(rpad("Time (s)", 10)) | Volume")
println("  " * "-"^40)
for r in (results_cc, results_dcc, results_dlog)
    println("  $(rpad(r.label, 8)) | $(rpad(string(round(r.time, digits=2)), 10)) | $(round(r.volume, sigdigits=6))")
end

# Volume consistency checks
vols = (results_cc.volume, results_dcc.volume, results_dlog.volume)
vmax = maximum(vols)
println("\n  Volume differences relative to CC:")
for r in (results_dcc, results_dlog)
    diff = abs(r.volume - results_cc.volume) / max(results_cc.volume, 1e-12)
    status = diff < 1e-3 ? "[OK]" : "[WARN]"
    println("  $status  $(r.label): $(round(diff*100, digits=4))% relative diff")
end
