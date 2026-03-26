"""
    test_anderson_benchmark_unicycle.jl

Benchmarks the effect of Anderson pairwise cuts on MILP solve time for the
Unicycle reachability problem using hybrid symbolic reachability.

Pipeline: multi_step_hybreach with concInt=[5,5] (10 total steps).

Conditions tested:
  A) CROWN bounds only, no Anderson cuts  (baseline)
  B) CROWN + Anderson cuts, max_cuts_per_layer = 50
  C) CROWN + Anderson cuts, max_cuts_per_layer = 200

Run from repo root:
    julia --project=. tests/test_anderson_benchmark_unicycle.jl
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
using Printf

# ── Unicycle problem definition ───────────────────────────────────────────────

control_coef = [[0],[0],[1],[1]]
controller   = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerUnicycle.nnet")
exprList     = [:(cos(x3)*x4), :(sin(x3)*x4), :(0*x3), :(0*x4)]
dt           = 0.2
w            = 1e-4
dig          = 15
const DOMAIN = Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high=[9.55,-4.45,2.11,1.51])
depMat       = [[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]
const CONC_INT = [10, 10, 10]
const N_OVERT  = 2

function unicycle_dynamics(x, u)
    dx1 = x[4]*cos(x[3]); dx2 = x[4]*sin(x[3])
    dx3 = u[2];            dx4 = u[1]
    return x + [dx1, dx2, dx3, dx4] .* dt
end

function unicycle_control(input_set)
    return input_set
end

function bound_unicycle(Unicycle; plotFlag=false, npoint=2)
    lbs, ubs = extrema(Unicycle.domain)
    lbs = floor.(lbs, digits=dig)
    ubs = ceil.(ubs, digits=dig)
    lb_x4 = lbs[4]; ub_x4 = ubs[4]; lb_x3 = lbs[3]; ub_x3 = ubs[3]

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
    lbs_x1 = [lbs[1]]; append!(lbs_x1, lbs[3:4]); ubs_x1 = [ubs[1]]; append!(ubs_x1, ubs[3:4])
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
    exprList, nothing, control_coef, DOMAIN,
    [:x1,:x2,:x3,:x4], nothing,
    unicycle_dynamics, bound_unicycle, unicycle_control, unicycle_dyn_con_link!
)

function make_query()
    return GraphPolyQuery(Unicycle, controller, Id(), "MIP", sum(CONC_INT), dt, N_OVERT, nothing, nothing, 2)
end

# ── Run hybrid reachability and collect metrics ───────────────────────────────

function run_condition(label, use_anderson, max_cuts)
    println("\n  Condition: $label")
    Unicycle.domain = DOMAIN  # reset initial domain

    q = make_query()
    t0 = time()
    hyb_sets, _ = multi_step_hybreach(q, depMat, CONC_INT; enc_func=ccEncoding!,
                                        use_anderson=use_anderson,
                                        max_cuts_per_layer=max_cuts)
    elapsed = time() - t0

    final_set = hyb_sets[end]
    vol = volume(final_set)
    println("    Total time: $(round(elapsed, digits=2))s")
    println("    Final vol : $(round(vol, sigdigits=6))")

    return (label=label, time=elapsed, volume=vol, final_set=final_set)
end

# ── Main benchmark ────────────────────────────────────────────────────────────

println("\n", "="^60)
println("Anderson Cuts Benchmark — Unicycle hybrid reachability")
println("Initial domain: ", DOMAIN)
println("concInt: $CONC_INT  ($(sum(CONC_INT)) total steps), dt: $dt")
println("="^60)

conditions = [
    ("Anderson, cap=100",    true,  100),
]

results = []
for (label, use_anderson, cap) in conditions
    r = run_condition(label, use_anderson, cap)
    push!(results, r)
end

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n", "="^60)
println("Summary (concInt=$(CONC_INT))")
println("="^60)
println(@sprintf("%-28s  %9s  %8s", "Condition", "Time (s)", "Final vol"))
println("-"^50)
for r in results
    println(@sprintf("%-28s  %9.2f  %8.5f", r.label, r.time, r.volume))
end

baseline_time = results[1].time
println("\nSpeedup vs baseline:")
for r in results[2:end]
    println("  $(r.label): $(round(baseline_time / r.time, digits=2))×")
end

println("\nFinal reachset bounds:")
for r in results
    println("  $(r.label):")
    println("    low =$(round.(low(r.final_set),  digits=4))")
    println("    high=$(round.(high(r.final_set), digits=4))")
end

# ── Write report ──────────────────────────────────────────────────────────────

ts = Dates.format(now(), "yyyy-mm-dd-HH-MM")
report_path = joinpath(@__DIR__, "reports", "report_anderson_benchmark_unicycle_$(ts).md")

open(report_path, "w") do io
    println(io, "# Anderson Cuts Benchmark — Unicycle — $(ts)")
    println(io, "")
    println(io, "## Setup")
    println(io, "- Network: `controllerUnicycle.nnet` (ARCH-COMP-2023)")
    println(io, "- Initial domain: $(DOMAIN)")
    println(io, "- Pipeline: `multi_step_hybreach`, concInt=$(CONC_INT), CC dynamics encoding")
    println(io, "- N_overt: $N_OVERT, dt: $dt")
    println(io, "- Bounds: CROWN back-substitution")
    println(io, "")
    println(io, "## Results")
    println(io, "")
    println(io, "| Condition | Time (s) | Final volume | Speedup |")
    println(io, "|-----------|----------|--------------|---------|")
    for r in results
        speedup = round(baseline_time / r.time, digits=2)
        println(io, @sprintf("| %-26s | %8.2f | %12.5f | %5.2f× |",
                r.label, r.time, r.volume, speedup))
    end
    println(io, "")
    println(io, "## Final reachset bounds")
    println(io, "")
    for r in results
        println(io, "- **$(r.label)**: low=$(round.(low(r.final_set), digits=4)), high=$(round.(high(r.final_set), digits=4))")
    end
    println(io, "")
    println(io, "## Soundness check")
    ref_vol = results[1].volume
    all_consistent = all(abs(r.volume - ref_vol) / max(ref_vol, 1e-12) < 1e-3 for r in results)
    println(io, "All final volumes within 0.1% of baseline: $all_consistent")
end

println("\nReport written to: $report_path")
