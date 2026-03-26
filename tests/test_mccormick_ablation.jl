"""
Ablation: McCormick envelopes in prodBounds vs interval arithmetic.
Unicycle benchmark, concInt=[5], baseline CC encoding, no CROWN.
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
# Unicycle shared parameters
# ============================================================
uni_control_coef = [[0],[0],[1],[1]]
uni_controller   = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerUnicycle.nnet")
uni_exprList     = [:(cos(x3)*x4), :(sin(x3)*x4), :(0*x3), :(0*x4)]
uni_dt           = 0.2
uni_w            = 1e-4
uni_dig          = 15
uni_domain       = Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high=[9.55,-4.45,2.11,1.51])
uni_depMat       = [[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]
uni_numSteps     = 50

function unicycle_dynamics(x, u)
    dx1 = x[4]*cos(x[3]); dx2 = x[4]*sin(x[3]); dx3 = u[2]; dx4 = u[1]
    return x + [dx1, dx2, dx3, dx4] .* uni_dt
end
function unicycle_control(input_set); return input_set; end

# Bound function parameterised by use_mccormick
function make_bound_unicycle(use_mccormick::Bool)
    function bound_unicycle(Unicycle; npoint=2, kwargs...)
        lbs, ubs = extrema(Unicycle.domain)
        lbs = floor.(lbs, digits=uni_dig)
        ubs = ceil.(ubs, digits=uni_dig)
        lb_x4 = lbs[4]; ub_x4 = ubs[4]; lb_x3 = lbs[3]; ub_x3 = ubs[3]

        x1FuncSub_1LB, x1FuncSub_1UB = interpol_nd(bound_univariate(:(1*x4), lb_x4, ub_x4)...)
        x1FuncSub_2LB, x1FuncSub_2UB = interpol_nd(bound_univariate(:(cos(x3)), lb_x3, ub_x3, npoint=npoint)...)
        l_x1Sub1LB, l_x1Sub1UB = lift_OA([1], [2], x1FuncSub_1LB, x1FuncSub_1UB, lbs[3:4], ubs[3:4])
        l_x1Sub2LB, l_x1Sub2UB = lift_OA([2], [1], x1FuncSub_2LB, x1FuncSub_2UB, lbs[3:4], ubs[3:4])
        x1FuncLB, x1FuncUB = prodBounds(l_x1Sub1LB, l_x1Sub1UB, l_x1Sub2LB, l_x1Sub2UB; use_mccormick=use_mccormick)

        x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(:(1*x4), lb_x4, ub_x4)...)
        x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(:(sin(x3)), lb_x3, ub_x3, npoint=npoint)...)
        l_x2Sub1LB, l_x2Sub1UB = lift_OA([1], [2], x2FuncSub1LB, x2FuncSub1UB, lbs[3:4], ubs[3:4])
        l_x2Sub2LB, l_x2Sub2UB = lift_OA([2], [1], x2FuncSub2LB, x2FuncSub2UB, lbs[3:4], ubs[3:4])
        x2FuncLB, x2FuncUB = prodBounds(l_x2Sub1LB, l_x2Sub1UB, l_x2Sub2LB, l_x2Sub2UB; use_mccormick=use_mccormick)

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
    return bound_unicycle
end

function unicycle_dyn_con_link!(query, neurons, graph, dynModel, netModel, t_ind=nothing)
    @variable(netModel, x1); @variable(netModel, x2); @variable(netModel, x3); @variable(netModel, x4)
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

# ============================================================
# Ablation
# ============================================================
concInt = [5]

ablations = [
    ("McCormick",    true),
    ("IntervalArith", false),
]

results = Dict()

for (label, use_mccormick) in ablations
    println("\n--- $label ---")
    flush(stdout)

    bound_fn = make_bound_unicycle(use_mccormick)
    Unicycle = GraphPolyProblem(uni_exprList, nothing, uni_control_coef, uni_domain,
        [:x1,:x2,:x3,:x4], nothing, unicycle_dynamics, bound_fn, unicycle_control, unicycle_dyn_con_link!)
    q = GraphPolyQuery(Unicycle, uni_controller, Id(), "MIP", uni_numSteps, uni_dt, 2, nothing, nothing, 2)

    # Warmup
    q_warm = deepcopy(q)
    multi_step_hybreach(q_warm, uni_depMat, [2, 2]; enc_func=ccEncoding!, use_crown=false)

    # Timed run
    q2 = deepcopy(q)
    tstart = Dates.now()
    hyb_sets, _ = multi_step_hybreach(q2, uni_depMat, concInt; enc_func=ccEncoding!, use_crown=false)
    tend = Dates.now()

    elapsed_s = Dates.value(tend - tstart) / 1000.0
    final_set = hyb_sets[end]
    vol = volume(final_set)

    results[label] = (time_s=elapsed_s, volume=vol, final_set=final_set)
    println("  Time:   $(round(elapsed_s, digits=2)) s")
    println("  Volume: $vol")
    println("  Low:    $(final_set.center .- final_set.radius)")
    println("  High:   $(final_set.center .+ final_set.radius)")
    flush(stdout)
end

# ============================================================
# Summary
# ============================================================
println("\n========== MCCORMICK ABLATION SUMMARY ==========")
println("Unicycle, concInt=$concInt, CC encoding, no CROWN")
println("  $(rpad("Configuration", 15)) | $(rpad("Time (s)", 10)) | Volume")
println("  " * "-"^48)
for (label, _) in ablations
    r = results[label]
    println("  $(rpad(label, 15)) | $(rpad(string(round(r.time_s, digits=2)), 10)) | $(r.volume)")
end

vol_mc  = results["McCormick"].volume
vol_ia  = results["IntervalArith"].volume
println("\n  Volume reduction (McCormick vs IA): $(round((vol_ia - vol_mc)/vol_ia * 100, digits=2))%")
