"""
Unicycle ablations only — run after TORA ablations are already complete.
TORA results (hardcoded from prior run):
  CROWN_only:  520.4s, volume 0.6656
  DCC_only:    793.6s, volume 0.6656
  CROWN_DCC:   563.7s, volume 0.6657
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

# ============================================================
# Unicycle problem definition (matching L4DC benchmark)
# ============================================================
control_coef = [[0],[0],[1],[1]]
controller   = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerUnicycle.nnet")
exprList     = [:(cos(x3)*x4), :(sin(x3)*x4), :(0*x3), :(0*x4)]
dt           = 0.2
w            = 1e-4
dig          = 15
domain       = Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high=[9.55,-4.45,2.11,1.51])
depMat       = [[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]
N_overt      = 2
numSteps     = 50

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
    exprList, nothing, control_coef, domain,
    [:x1,:x2,:x3,:x4], nothing,
    unicycle_dynamics, bound_unicycle, unicycle_control, unicycle_dyn_con_link!
)

uni_query = GraphPolyQuery(Unicycle, controller, Id(), "MIP", numSteps, dt, N_overt, nothing, nothing, 2)

# ============================================================
# Ablations
# ============================================================
ablations = [
    ("Baseline",   ccEncoding!,  false),
    ("CROWN_only", ccEncoding!,  true),
    ("DCC_only",   dccEncoding!, false),
    ("CROWN_DCC",  dccEncoding!, true),
]

concInt = [10, 10, 10]  # change to [10,10,10,10,10] for full benchmark

println("\n========== UNICYCLE ABLATIONS (concInt=$concInt) ==========")
uni_results = Dict()

for (label, enc_func, use_crown) in ablations
    println("\n--- Unicycle: $label ---")

    # Untimed warmup: 1-step hybreach to trigger Julia specialization for this
    # (enc_func, use_crown) combination before the clock starts.
    q_warm = deepcopy(uni_query)
    multi_step_hybreach(q_warm, depMat, [2, 2]; enc_func=enc_func, use_crown=use_crown)

    # Timed run
    q = deepcopy(uni_query)
    tstart = Dates.now()
    hyb_sets, _ = multi_step_hybreach(q, depMat, concInt; enc_func=enc_func, use_crown=use_crown)
    tend = Dates.now()

    elapsed_s = Dates.value(tend - tstart) / 1000.0
    final_sym = hyb_sets[end]
    vol = volume(final_sym)
    uni_results[label] = (time_s=elapsed_s, volume=vol, final_set=final_sym)

    println("  Total time: $(round(elapsed_s, digits=1)) s")
    println("  Volume:     $vol")
    println("  Low:        $(final_sym.center .- final_sym.radius)")
    println("  High:       $(final_sym.center .+ final_sym.radius)")
    flush(stdout)
end

# ============================================================
# Summary
# ============================================================
println("\n========== ABLATION SUMMARY ==========")

println("\nTORA (20 concrete steps, dt=0.1) — from prior run:")
println("  $(rpad("Configuration", 15)) | $(rpad("Time (s)", 10)) | Volume")
println("  " * "-"^48)
for (label, t, vol) in [("Baseline", 598.59, 0.6656), ("CROWN_only", 519.56, 0.6656), ("DCC_only", 785.54, 0.6655), ("CROWN_DCC", 557.77, 0.6657)]
    println("  $(rpad(label, 15)) | $(rpad(string(t), 10)) | $vol")
end

println("\nUnicycle (concInt=$concInt, dt=0.2):")
println("  $(rpad("Configuration", 15)) | $(rpad("Time (s)", 10)) | Volume")
println("  " * "-"^48)
for (label, _, _) in ablations
    r = uni_results[label]
    println("  $(rpad(label, 15)) | $(rpad(string(round(r.time_s, digits=1)), 10)) | $(r.volume)")
end
