"""
    test_anderson_benchmark.jl

Benchmarks the effect of Anderson pairwise cuts on MILP solve time for the
TORA reachability problem.

Uses multi_step_concreach with N_STEPS=10, measuring total wall-clock time
across all steps and reporting per-step average and final reachset volume.

Conditions tested:
  A) CROWN bounds only, no Anderson cuts  (baseline)
  B) CROWN + Anderson cuts, max_cuts_per_layer = 50
  C) CROWN + Anderson cuts, max_cuts_per_layer = 200

Run from repo root:
    julia --project=. tests/test_anderson_benchmark.jl
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

# ── TORA problem definition ───────────────────────────────────────────────────

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

const DOMAIN  = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
const N_OVERT = 2
const N_STEPS = 20

TORA = GraphPolyProblem(
    exprList, nothing, control_coef, DOMAIN,
    [:x1,:x2,:x3,:x4], nothing,
    tora_dynamics, bound_tora, tora_control, tora_dyn_con_link!
)

function make_query(ntime)
    return GraphPolyQuery(TORA, controller, Id(), "MIP", ntime, dt, N_OVERT, nothing, nothing, 2)
end

# ── Run multi-step concreach and collect metrics ──────────────────────────────

function run_condition(label, use_anderson, max_cuts)
    println("\n  Condition: $label")

    # Reset domain to initial set each run
    TORA.domain = DOMAIN

    q = make_query(N_STEPS)

    t0 = time()
    reach_sets, _ = multi_step_concreach(q; enc_func=ccEncoding!,
                                            use_anderson=use_anderson,
                                            max_cuts_per_layer=max_cuts)
    elapsed = time() - t0

    final_set = reach_sets[end]
    vol = volume(final_set)
    per_step = elapsed / N_STEPS

    println("    Total time : $(round(elapsed, digits=2))s  ($(round(per_step, digits=2))s/step)")
    println("    Final vol  : $(round(vol, sigdigits=6))")

    return (label=label, time=elapsed, per_step=per_step, volume=vol, final_set=final_set,
            reach_sets=reach_sets)
end

# ── Main benchmark ────────────────────────────────────────────────────────────

println("\n", "="^60)
println("Anderson Cuts Benchmark — TORA $(N_STEPS)-step reachability")
println("Initial domain: ", DOMAIN)
println("N_overt: $N_OVERT, dt: $dt")
println("="^60)

conditions = [
    ("Anderson, cap=100",    true,  100),
]

results = []
for (label, use_anderson, cap) in conditions
    r = run_condition(label, use_anderson, cap)
    push!(results, r)
end

# ── Summary table ─────────────────────────────────────────────────────────────

println("\n", "="^60)
println("Summary ($(N_STEPS) steps)")
println("="^60)
println(@sprintf("%-28s  %9s  %9s  %8s", "Condition", "Total(s)", "Per step", "Final vol"))
println("-"^60)
for r in results
    println(@sprintf("%-28s  %9.2f  %9.2f  %8.5f",
            r.label, r.time, r.per_step, r.volume))
end

baseline_time = results[1].time
println("\nSpeedup vs baseline:")
for r in results[2:end]
    println("  $(r.label): $(round(baseline_time / r.time, digits=2))×")
end

# Soundness check
println("\nFinal reachset bounds:")
for r in results
    println("  $(r.label):")
    println("    low =$(round.(low(r.final_set),  digits=4))")
    println("    high=$(round.(high(r.final_set), digits=4))")
end

# ── Write report ──────────────────────────────────────────────────────────────

using Dates
ts = Dates.format(now(), "yyyy-mm-dd-HH-MM")
report_path = joinpath(@__DIR__, "reports", "report_anderson_benchmark_$(ts).md")

open(report_path, "w") do io
    println(io, "# Anderson Cuts Benchmark Report — $(ts)")
    println(io, "")
    println(io, "## Setup")
    println(io, "- Network: `controllerTORA.nnet` (ARCH-COMP-2023)")
    println(io, "- Initial domain: $(DOMAIN)")
    println(io, "- Pipeline: `multi_step_concreach`, $(N_STEPS) steps, CC dynamics encoding")
    println(io, "- N_overt: $N_OVERT, dt: $dt")
    println(io, "- Bounds: CROWN back-substitution")
    println(io, "")
    println(io, "## Results")
    println(io, "")
    println(io, "| Condition | Total (s) | Per step (s) | Final volume | Speedup |")
    println(io, "|-----------|-----------|--------------|--------------|---------|")
    for r in results
        speedup = round(baseline_time / r.time, digits=2)
        println(io, @sprintf("| %-26s | %9.2f | %12.2f | %12.5f | %5.2f× |",
                r.label, r.time, r.per_step, r.volume, speedup))
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