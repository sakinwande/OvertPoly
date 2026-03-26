"""
    test_dlog_ablation.jl

Compares CROWN+DCC vs CROWN+DLOG encoding on TORA (20 concrete steps)
and Unicycle (concInt=[10,10,10] hybrid steps). No Anderson cuts.

Run from repo root:
    julia --project=. tests/test_dlog_ablation.jl
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

tora_control_coef = [[0],[0],[0],[1]]
tora_controller   = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerTORA.nnet")
tora_exprList     = [:(1*x2), :(-x1 + 0.1*sin(x3)), :(1*x4), :(1*u)]
tora_dt           = 0.1
const TORA_DOMAIN = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
const TORA_STEPS  = 20

function tora_dynamics(x, u)
    dx1 = x[2]; dx2 = -x[1] + 0.1*sin(x[3]); dx3 = x[4]; dx4 = u
    return x + [dx1, dx2, dx3, dx4] .* tora_dt
end

function tora_control(input_set); return input_set; end

function bound_tora(TORA; npoint=2, kwargs...)
    lbs, ubs = extrema(TORA.domain)
    lb_x2 = lbs[2]; ub_x2 = ubs[2]
    x1FuncLB_u, _ = interpol_nd(bound_univariate(:(1*x2), lb_x2, ub_x2)...)
    emptyList = [1]; currList = [2]
    x1FuncLB, x1FuncUB = lift_OA(emptyList, currList, x1FuncLB_u, x1FuncLB_u, lbs[1], ubs[1])
    lb_x1 = lbs[1]; ub_x1 = ubs[1]; lb_x3 = lbs[3]; ub_x3 = ubs[3]; lb_x4 = lbs[4]; ub_x4 = ubs[4]
    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(:(-1*x1), lb_x1, ub_x1)...)
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(:(sin(x3)), lb_x3, ub_x3)...)
    x2FuncSub2LB = [(tup[1:end-1]..., 0.1*tup[end]) for tup in x2FuncSub2LB]
    x2FuncSub2UB = [(tup[1:end-1]..., 0.1*tup[end]) for tup in x2FuncSub2UB]
    emptyList = [2]; currList = [1]
    l_x2FuncSub1LB, l_x2FuncSub1UB = lift_OA(emptyList, currList, x2FuncSub1LB, x2FuncSub1UB, [lbs[1], lbs[3]], [ubs[1], ubs[3]])
    emptyList = [1]; currList = [2]
    l_x2FuncSub2LB, l_x2FuncSub2UB = lift_OA(emptyList, currList, x2FuncSub2LB, x2FuncSub2UB, [lbs[1], lbs[3]], [ubs[1], ubs[3]])
    x2FuncLB_u, x2FuncUB_u = sumBounds(l_x2FuncSub1LB, l_x2FuncSub1UB, l_x2FuncSub2LB, l_x2FuncSub2UB, false)
    emptyList = [2]; currList = [1,3]
    x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs, ubs)
    x3FuncLB_u, x3FuncUB_u = interpol_nd(bound_univariate(:(1*x4), lb_x4, ub_x4)...)
    emptyList = [1]; currList = [2]
    x3FuncLB, x3FuncUB = lift_OA(emptyList, currList, x3FuncLB_u, x3FuncUB_u, lb_x3, ub_x3)
    x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(:(0*x4), lb_x4, ub_x4)...)
    return [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
end

function tora_dyn_con_link!(query, neurons, graph, dynModel, netModel)
    @variable(netModel, x1); @variable(netModel, x2)
    @variable(netModel, x3); @variable(netModel, x4)
    @constraint(netModel, neurons[1][1] == x1); @constraint(netModel, neurons[1][2] == x2)
    @constraint(netModel, neurons[1][3] == x3); @constraint(netModel, neurons[1][4] == x4)
    @linkconstraint(graph, netModel[:x1] == dynModel[2][:x][1])
    @linkconstraint(graph, netModel[:x2] == dynModel[2][:x][2])
    @linkconstraint(graph, netModel[:x3] == dynModel[3][:x][1])
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

TORA = GraphPolyProblem(tora_exprList, nothing, tora_control_coef, TORA_DOMAIN,
    [:x1,:x2,:x3,:x4], nothing, tora_dynamics, bound_tora, tora_control, tora_dyn_con_link!)

# ── Unicycle problem definition ───────────────────────────────────────────────

uni_control_coef = [[0],[0],[1],[1]]
uni_controller   = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerUnicycle.nnet")
uni_exprList     = [:(cos(x3)*x4), :(sin(x3)*x4), :(0*x3), :(0*x4)]
uni_dt           = 0.2
uni_w            = 1e-4
uni_dig          = 15
const UNI_DOMAIN   = Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high=[9.55,-4.45,2.11,1.51])
uni_depMat         = [[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]
const UNI_CONC_INT = [10, 10, 10]

function unicycle_dynamics(x, u)
    dx1 = x[4]*cos(x[3]); dx2 = x[4]*sin(x[3])
    dx3 = u[2];            dx4 = u[1]
    return x + [dx1, dx2, dx3, dx4] .* uni_dt
end

function unicycle_control(input_set); return input_set; end

function bound_unicycle(Unicycle; npoint=2, kwargs...)
    lbs, ubs = extrema(Unicycle.domain)
    lbs = floor.(lbs, digits=uni_dig)
    ubs = ceil.(ubs, digits=uni_dig)
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
    x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(:(0*x4), lb_x4, ub_x4, ϵ=uni_w)...)
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

Unicycle = GraphPolyProblem(uni_exprList, nothing, uni_control_coef, UNI_DOMAIN,
    [:x1,:x2,:x3,:x4], nothing, unicycle_dynamics, bound_unicycle, unicycle_control, unicycle_dyn_con_link!)

# ── Ablation configs ──────────────────────────────────────────────────────────

ablations = [
    ("CROWN_DLOG", dlogEncoding!, true),
]

# ── Run helper ────────────────────────────────────────────────────────────────

function run_tora(label, enc_func, use_crown)
    println("\n--- TORA: $label (warmup) ---"); flush(stdout)
    TORA.domain = TORA_DOMAIN
    q_w = GraphPolyQuery(TORA, tora_controller, Id(), "MIP", 2, tora_dt, 2, nothing, nothing, 2)
    multi_step_concreach(q_w; enc_func=enc_func, use_crown=use_crown)

    println("--- TORA: $label (timed) ---"); flush(stdout)
    TORA.domain = TORA_DOMAIN
    q = GraphPolyQuery(TORA, tora_controller, Id(), "MIP", TORA_STEPS, tora_dt, 2, nothing, nothing, 2)
    t0 = time()
    sets, _ = multi_step_concreach(q; enc_func=enc_func, use_crown=use_crown)
    elapsed = time() - t0
    vol = volume(sets[end])
    println("  Time:   $(round(elapsed, digits=2)) s  ($(round(elapsed/TORA_STEPS, digits=2)) s/step)")
    println("  Volume: $vol")
    println("  Low:    $(round.(low(sets[end]),  digits=4))")
    println("  High:   $(round.(high(sets[end]), digits=4))")
    flush(stdout)
    return (label=label, time=elapsed, volume=vol, final_set=sets[end])
end

function run_unicycle(label, enc_func, use_crown)
    println("\n--- Unicycle: $label (warmup) ---"); flush(stdout)
    Unicycle.domain = UNI_DOMAIN
    q_w = GraphPolyQuery(Unicycle, uni_controller, Id(), "MIP", sum([2,2]), uni_dt, 2, nothing, nothing, 2)
    multi_step_hybreach(q_w, uni_depMat, [2, 2]; enc_func=enc_func, use_crown=use_crown)

    println("--- Unicycle: $label (timed) ---"); flush(stdout)
    Unicycle.domain = UNI_DOMAIN
    q = GraphPolyQuery(Unicycle, uni_controller, Id(), "MIP", sum(UNI_CONC_INT), uni_dt, 2, nothing, nothing, 2)
    t0 = time()
    sets, _ = multi_step_hybreach(q, uni_depMat, UNI_CONC_INT; enc_func=enc_func, use_crown=use_crown)
    elapsed = time() - t0
    vol = volume(sets[end])
    println("  Time:   $(round(elapsed, digits=2)) s")
    println("  Volume: $vol")
    println("  Low:    $(round.(low(sets[end]),  digits=4))")
    println("  High:   $(round.(high(sets[end]), digits=4))")
    flush(stdout)
    return (label=label, time=elapsed, volume=vol, final_set=sets[end])
end

# ── Main ──────────────────────────────────────────────────────────────────────

println("\n", "="^60)
println("CROWN × DCC/DLOG Ablation")
println("TORA: $TORA_STEPS steps, dt=$tora_dt")
println("Unicycle: concInt=$UNI_CONC_INT, dt=$uni_dt")
println("="^60)

tora_results = []
for (label, enc_func, use_crown) in ablations
    push!(tora_results, run_tora(label, enc_func, use_crown))
end

uni_results = []
for (label, enc_func, use_crown) in ablations
    push!(uni_results, run_unicycle(label, enc_func, use_crown))
end

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n", "="^60)
println("Summary")
println("="^60)
println(@sprintf("%-12s  %-12s  %9s  %8s", "Benchmark", "Config", "Time (s)", "Volume"))
println("-"^50)
for r in tora_results
    println(@sprintf("%-12s  %-12s  %9.2f  %.5g", "TORA (20 st)", r.label, r.time, r.volume))
end
for r in uni_results
    println(@sprintf("%-12s  %-12s  %9.2f  %.5g", "Unicycle(30)", r.label, r.time, r.volume))
end


# ── Write report ──────────────────────────────────────────────────────────────

ts = Dates.format(now(), "yyyy-mm-dd-HH-MM")
report_path = joinpath(@__DIR__, "reports", "report_dlog_ablation_$(ts).md")

open(report_path, "w") do io
    println(io, "# CROWN × DCC/DLOG Ablation — $(ts)")
    println(io, "")
    println(io, "## Setup")
    println(io, "- Bounds: CROWN back-substitution (both configs)")
    println(io, "- No Anderson cuts")
    println(io, "- DCC: standard disaggregated CC (one binary per simplex)")
    println(io, "- DLOG: logarithmic DCC with cell-wise Gray codes (⌈log₂(n)⌉ binaries)")
    println(io, "")
    println(io, "## Results")
    println(io, "")
    println(io, "| Benchmark | Config | Time (s) | Volume |")
    println(io, "|-----------|--------|----------|--------|")
    for r in tora_results
        println(io, @sprintf("| TORA (%d steps, dt=%.1f) | %s | %.2f | %.5g |",
                TORA_STEPS, tora_dt, r.label, r.time, r.volume))
    end
    for r in uni_results
        println(io, @sprintf("| Unicycle (concInt=%s, dt=%.1f) | %s | %.2f | %.5g |",
                string(UNI_CONC_INT), uni_dt, r.label, r.time, r.volume))
    end
    println(io, "")
    println(io, "## Final reachset bounds")
    println(io, "")
    for r in tora_results
        println(io, "- **TORA $(r.label)**: low=$(round.(low(r.final_set), digits=4)), high=$(round.(high(r.final_set), digits=4))")
    end
    for r in uni_results
        println(io, "- **Unicycle $(r.label)**: low=$(round.(low(r.final_set), digits=4)), high=$(round.(high(r.final_set), digits=4))")
    end
end

println("\nReport written to: $report_path")
