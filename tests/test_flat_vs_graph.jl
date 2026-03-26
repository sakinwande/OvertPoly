"""
Compare FlatPolyQuery vs GraphPolyQuery on Single Pendulum and ACC benchmarks.
Both use concrete reachability (multi_step_concreach).
"""

# Include both reachability backends
include(joinpath(@__DIR__, "..", "src", "overtPoly_helpers.jl"))
include(joinpath(@__DIR__, "..", "src", "nn_mip_encoding.jl"))
include(joinpath(@__DIR__, "..", "src", "overtPoly_to_mip.jl"))
include(joinpath(@__DIR__, "..", "src", "overt_to_pwa.jl"))
include(joinpath(@__DIR__, "..", "src", "problems.jl"))
include(joinpath(@__DIR__, "..", "src", "reachability.jl"))        # FlatPolyQuery methods
include(joinpath(@__DIR__, "..", "src", "distr_reachability.jl"))  # GraphPolyQuery methods
using LazySets
using Dates
using Plasmo

# ============================================================
# SINGLE PENDULUM — shared parameters
# ============================================================
pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.0
pend_controller = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerSinglePendulum.nnet")
pend_expr = [:($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)]
pend_domain = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
pend_dt = 0.05
pend_steps = 20

function pend_dynamics(x, u, dt)
    dx1 = x[2]
    dx2 = (grav_const/pend_len) * sin(x[1]) + (1/(pend_mass*pend_len^2)) * u - (friction/(pend_mass*pend_len^2)) * x[2]
    return [x[1] + dx1*dt, x[2] + dx2*dt]
end

# --- Flat Pendulum bound function (uses interpol + addDim + MinkSum) ---
function flat_pend_bound(SinglePendulum; plotFlag=false)
    lbs, ubs = extrema(SinglePendulum.domain)
    lb1, ub1 = lbs[1], ubs[1]
    lb2, ub2 = lbs[2], ubs[2]

    bF1s1LB, bF1s1UB = bound_univariate(:($(grav_const/pend_len) * sin(x1)), lb1, ub1, plotflag=false)
    bF1s1LB, bF1s1UB = interpol(bF1s1LB, bF1s1UB)

    bF1s2LB, bF1s2UB = bound_univariate(:($((friction)/((pend_mass)*(pend_len)^2)) * -x2), lb2, ub2, plotflag=false)
    bF1s2LB, bF1s2UB = interpol(bF1s2LB, bF1s2UB)

    bF1s1LB_l = addDim(bF1s1LB, 2)
    bF1s1UB_l = addDim(bF1s1UB, 2)
    bF1s2LB_l = addDim(bF1s2LB, 1)
    bF1s2UB_l = addDim(bF1s2UB, 1)

    bF1LB = MinkSum(bF1s1LB_l, bF1s2LB_l)
    bF1UB = MinkSum(bF1s1UB_l, bF1s2UB_l)

    return [[bF1LB, bF1UB]]
end

function flat_pend_control(model, input_vars, control_vars, output_vars, input_set)
    return input_vars, input_set, control_vars
end

function flat_pend_update_rule(input_vars, overt_output_vars)
    ddth = overt_output_vars[1]
    return Dict(input_vars[1] => input_vars[2], input_vars[2] => ddth)
end

function flat_pend_link(p) end

FlatPendulum = FlatPolyProblem(
    pend_expr, nothing,
    [1/(pend_mass*pend_len^2)], 1,
    pend_domain, [:dθ], nothing,
    flat_pend_update_rule, pend_dynamics, flat_pend_bound, flat_pend_control, flat_pend_link
)
flat_pend_query = FlatPolyQuery(FlatPendulum, pend_controller, Id(), "MIP", pend_steps, pend_dt, 2, nothing, nothing, 2)

# --- Graph Pendulum bound function (uses interpol_nd + lift_OA + sumBounds) ---
function graph_pend_bound(SinglePendulum; plotFlag=false, npoint=2)
    lbs, ubs = extrema(SinglePendulum.domain)
    lb1, ub1 = lbs[1], ubs[1]
    lb2, ub2 = lbs[2], ubs[2]

    bF1s1LB, bF1s1UB = interpol_nd(bound_univariate(:($(grav_const/pend_len) * sin(x1)), lb1, ub1, plotflag=plotFlag)...)
    bF1s2LB, bF1s2UB = interpol_nd(bound_univariate(:($((friction)/((pend_mass)*(pend_len)^2)) *-x2), lb2, ub2, plotflag=plotFlag)...)

    l_bF1s1LB, l_bF1s1UB = lift_OA([2], [1], bF1s1LB, bF1s1UB, lbs, ubs)
    l_bF1s2LB, l_bF1s2UB = lift_OA([1], [2], bF1s2LB, bF1s2UB, lbs, ubs)
    bF1LB, bF1UB = sumBounds(l_bF1s1LB, l_bF1s1UB, l_bF1s2LB, l_bF1s2UB, false)

    # Bound angle dynamics (θ_dot = x2)
    θ_dot_lb_u, θ_dot_ub_u = bound_univariate(:(1*x2), lb2, ub2, plotflag=plotFlag)
    θ_dot_lb, θ_dot_ub = lift_OA([1], [2], θ_dot_lb_u, θ_dot_ub_u, lbs[1], ubs[1])

    return [[θ_dot_lb, θ_dot_ub], [bF1LB, bF1UB]]
end

function graph_pend_control(input_set)
    return input_set
end

function graph_pend_link!(query, neurons, graph, dynModel, netModel, t_ind=nothing)
    @linkconstraint(graph, dynModel[2][:x][1] == neurons[1][1])
    @linkconstraint(graph, dynModel[2][:x][2] == neurons[1][2])
    @variable(netModel, u)
    @constraint(netModel, u == neurons[end][1])
    @linkconstraint(graph, dynModel[2][:u] == neurons[end][1])
    i = 0
    for sym in query.problem.varList
        sym_t = isnothing(t_ind) ? sym : Meta.parse("$(sym)_$(t_ind)")
        i += 1
        pertVar = (sym == :dθ) ? dynModel[i][:x][end] : dynModel[i][:x][1]
        push!(query.var_dict[sym_t], [pertVar])
    end
end

GraphPendulum = GraphPolyProblem(
    pend_expr, nothing,
    [[0], [1/(pend_mass*pend_len^2)]],
    pend_domain, [:θ, :dθ], nothing,
    pend_dynamics, graph_pend_bound, graph_pend_control, graph_pend_link!
)
graph_pend_query = GraphPolyQuery(GraphPendulum, pend_controller, Id(), "MIP", pend_steps, pend_dt, 1, nothing, nothing, 2)

# ============================================================
# ACC — shared parameters
# ============================================================
ac_lead = -2.0
mu = 0.0001
acc_controller = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerACC.nnet")
acc_exprList = [:(- $mu*x2^2 + -2*x3 + 2*$ac_lead), :(- $mu*x5^2 - 2*x6)]
acc_ϵ = 1e-8
acc_domain = Hyperrectangle(low=[90,32,-acc_ϵ,10,30,-acc_ϵ], high=[110,32.2,acc_ϵ,11,30.2,acc_ϵ])
acc_dt = 0.1
acc_steps = 50
acc_vSet = 30.0
acc_tGap = 1.40

function acc_dynamics(x, u)
    dx1 = x[2]; dx2 = x[3]
    dx3 = -2*x[3] + 2*ac_lead - mu*x[2]^2
    dx4 = x[5]; dx5 = x[6]
    dx6 = -2*x[6] + 2*u[1] - mu*x[5]^2
    return x + [dx1, dx2, dx3, dx4, dx5, dx6] .* acc_dt
end

# --- Flat ACC ---
function flat_acc_control(model, input_vars, control_vars, output_vars, input_set, ϵ=1e-12)
    vSet_v = @variable(model, [1:1], base_name="v_set")
    tGap_v = @variable(model, [1:1], base_name="tGap")
    dRel_v = @variable(model, [1:1], base_name="dRel")
    vRel_v = @variable(model, [1:1], base_name="vRel")
    @constraint(model, vSet_v .== acc_vSet)
    @constraint(model, tGap_v .== acc_tGap)
    @constraint(model, dRel_v .== input_vars[1] - input_vars[4])
    @constraint(model, vRel_v .== input_vars[2] - input_vars[5])
    con_inp_vars = [vSet_v[1], tGap_v[1], input_vars[5], dRel_v[1], vRel_v[1]]
    con_net_vars = control_vars[end]
    LBs, UBs = extrema(input_set)
    dRel_LB = LBs[1] - UBs[4]; dRel_UB = UBs[1] - LBs[4]
    vRel_LB = LBs[2] - UBs[5]; vRel_UB = UBs[2] - LBs[5]
    con_inp_set = Hyperrectangle(low=[30.0-ϵ, 1.40-ϵ, LBs[5], dRel_LB, vRel_LB], high=[30.0+ϵ, 1.40+ϵ, UBs[5], dRel_UB, vRel_UB])
    return con_inp_vars, con_inp_set, con_net_vars
end

function flat_acc_update_rule(input_vars, overt_output_vars)
    return Dict(
        input_vars[1] => input_vars[2], input_vars[2] => input_vars[3], input_vars[3] => overt_output_vars[1],
        input_vars[4] => input_vars[5], input_vars[5] => input_vars[6], input_vars[6] => overt_output_vars[2]
    )
end

function flat_acc_bound(ACC; plotFlag=false)
    lbs, ubs = extrema(ACC.domain)
    leadSub1LB, leadSub1UB = interpol(bound_univariate(:(2*$ac_lead - $mu*x2^2), lbs[2], ubs[2], plotflag=plotFlag)...)
    leadSub2LB, leadSub2UB = interpol(bound_univariate(:(-2*x3), lbs[3], ubs[3], plotflag=plotFlag)..., 9)
    leadSub1LB_l = addDim(leadSub1LB, 2); leadSub1UB_l = addDim(leadSub1UB, 2)
    leadSub2LB_l = addDim(leadSub2LB, 1); leadSub2UB_l = addDim(leadSub2UB, 1)
    leadLB = unique(MinkSum(leadSub1LB_l, leadSub2LB_l))
    leadUB = unique(MinkSum(leadSub1UB_l, leadSub2UB_l))
    leadLB_l, leadUB_l = lift_OA([1], [2,3], leadLB, leadUB, lbs, ubs)
    egoSub1LB, egoSub1UB = interpol(bound_univariate(:(-$mu*x5^2), lbs[5], ubs[5], plotflag=plotFlag)...)
    egoSub2LB, egoSub2UB = interpol(bound_univariate(:(-2*x6), lbs[6], ubs[6], plotflag=plotFlag)..., 9)
    egoSub1LB_l = addDim(egoSub1LB, 2); egoSub1UB_l = addDim(egoSub1UB, 2)
    egoSub2LB_l = addDim(egoSub2LB, 1); egoSub2UB_l = addDim(egoSub2UB, 1)
    egoLB = unique(MinkSum(egoSub1LB_l, egoSub2LB_l))
    egoUB = unique(MinkSum(egoSub1UB_l, egoSub2UB_l))
    egoLB_l, egoUB_l = lift_OA([1], [2,3], egoLB, egoUB, lbs, ubs)
    return [[leadLB, leadUB], [egoLB, egoUB]]
end

function flat_acc_link!(query)
    model = query.mod_dict[:x3]
    @constraint(model, query.var_dict[:X][1][2] == query.var_dict[:x3][1][1])
    @constraint(model, query.var_dict[:X][1][3] == query.var_dict[:x3][1][2])
    @constraint(model, query.var_dict[:X][1][5] == query.var_dict[:x6][1][1])
    @constraint(model, query.var_dict[:X][1][6] == query.var_dict[:x6][1][2])
end

FlatACC = FlatPolyProblem(
    acc_exprList, nothing,
    [[0],[2]], 1,
    acc_domain, [:x3, :x6], nothing,
    flat_acc_update_rule, acc_dynamics, flat_acc_bound, flat_acc_control, flat_acc_link!
)
flat_acc_query = FlatPolyQuery(FlatACC, acc_controller, Id(), "MIP", acc_steps, acc_dt, 2, nothing, nothing, 3)

# --- Graph ACC ---
function graph_acc_control(input_set, ϵ=1e-12)
    LBs, UBs = extrema(input_set)
    dRel_LB = LBs[1] - UBs[4]; dRel_UB = UBs[1] - LBs[4]
    vRel_LB = LBs[2] - UBs[5]; vRel_UB = UBs[2] - LBs[5]
    return Hyperrectangle(low=[30.0-ϵ, 1.40-ϵ, LBs[5], dRel_LB, vRel_LB], high=[30.0+ϵ, 1.40+ϵ, UBs[5], dRel_UB, vRel_UB])
end

function graph_acc_bound(ACC; plotFlag=false, npoint=2)
    lbs, ubs = extrema(ACC.domain)
    # Lead acceleration
    aleadSub1LB, aleadSub1UB = interpol_nd(bound_univariate(:(2*$ac_lead - $mu*x2^2), lbs[2], ubs[2], plotflag=plotFlag)...)
    aleadSub2LB, aleadSub2UB = interpol_nd(bound_univariate(:(-2*x3), lbs[3], ubs[3], plotflag=plotFlag)...)
    l_aleadSub1LB, l_aleadSub1UB = lift_OA([2], [1], aleadSub1LB, aleadSub1UB, lbs[2:3], ubs[2:3])
    l_aleadSub2LB, l_aleadSub2UB = lift_OA([1], [2], aleadSub2LB, aleadSub2UB, lbs[2:3], ubs[2:3])
    aleadLB, aleadUB = sumBounds(l_aleadSub1LB, l_aleadSub1UB, l_aleadSub2LB, l_aleadSub2UB, false)
    # Lead velocity
    vLeadLB_u, vLeadUB_u = bound_univariate(:(1*x3), lbs[3], ubs[3], plotflag=plotFlag)
    vLeadLB, vLeadUB = lift_OA([1], [2], vLeadLB_u, vLeadUB_u, lbs[2], ubs[2])
    # Lead position
    pLeadLB_u, pLeadUB_u = bound_univariate(:(1*x2), lbs[2], ubs[2], plotflag=plotFlag)
    pLeadLB, pLeadUB = lift_OA([1], [2], pLeadLB_u, pLeadUB_u, lbs[1], ubs[1])
    # Ego acceleration
    aegoSub1LB, aegoSub1UB = interpol_nd(bound_univariate(:(-$mu*x5^2), lbs[5], ubs[5], plotflag=plotFlag)...)
    aegoSub2LB, aegoSub2UB = interpol_nd(bound_univariate(:(-2*x6), lbs[6], ubs[6], plotflag=plotFlag)...)
    l_aegoSub1LB, l_aegoSub1UB = lift_OA([2], [1], aegoSub1LB, aegoSub1UB, lbs[5:6], ubs[5:6])
    l_aegoSub2LB, l_aegoSub2UB = lift_OA([1], [2], aegoSub2LB, aegoSub2UB, lbs[5:6], ubs[5:6])
    aegoLB, aegoUB = sumBounds(l_aegoSub1LB, l_aegoSub1UB, l_aegoSub2LB, l_aegoSub2UB, false)
    # Ego velocity
    vEgoLB_u, vEgoUB_u = bound_univariate(:(1*x6), lbs[6], ubs[6], plotflag=plotFlag)
    vEgoLB, vEgoUB = lift_OA([1], [2], vEgoLB_u, vEgoUB_u, lbs[5], ubs[5])
    # Ego position
    pEgoLB_u, pEgoUB_u = bound_univariate(:(1*x5), lbs[5], ubs[5], plotflag=plotFlag)
    pEgoLB, pEgoUB = lift_OA([1], [2], pEgoLB_u, pEgoUB_u, lbs[4], ubs[4])
    return [[pLeadLB, pLeadUB], [vLeadLB, vLeadUB], [aleadLB, aleadUB], [pEgoLB, pEgoUB], [vEgoLB, vEgoUB], [aegoLB, aegoUB]]
end

function graph_acc_link!(query, neurons, graph, dynModel, netModel)
    @constraint(netModel, neurons[1][1] == acc_vSet)
    @constraint(netModel, neurons[1][2] == acc_tGap)
    @variable(netModel, vEgo)
    @linkconstraint(graph, netModel[:vEgo] == dynModel[6][:x][1])
    @constraint(netModel, neurons[1][3] == vEgo)
    @variable(netModel, dRel)
    @linkconstraint(graph, netModel[:dRel] == dynModel[1][:x][1] - dynModel[4][:x][1])
    @constraint(netModel, neurons[1][4] == dRel)
    @variable(netModel, vRel)
    @linkconstraint(graph, netModel[:vRel] == dynModel[2][:x][1] - dynModel[5][:x][1])
    @constraint(netModel, neurons[1][5] == vRel)
    @variable(netModel, u)
    @constraint(netModel, neurons[end][1] == u)
    @linkconstraint(graph, netModel[:u] == dynModel[6][:u])
    i = 0
    for sym in query.problem.varList
        i += 1
        pertVar = (sym == :x3 || sym == :x6) ? dynModel[i][:x][end] : dynModel[i][:x][1]
        push!(query.var_dict[sym], [pertVar])
    end
end

GraphACC = GraphPolyProblem(
    acc_exprList, nothing,
    [[0],[0],[0],[0],[0],[2]],
    acc_domain, [:x1,:x2,:x3,:x4,:x5,:x6], nothing,
    acc_dynamics, graph_acc_bound, graph_acc_control, graph_acc_link!
)
graph_acc_query = GraphPolyQuery(GraphACC, acc_controller, Id(), "MIP", acc_steps, acc_dt, 2, nothing, nothing, 3)

# ============================================================
# Run benchmarks
# ============================================================
results = Dict{String, NamedTuple}()

benchmarks = [
    ("Pend_Flat",  flat_pend_query,  pend_steps),
    ("Pend_Graph", graph_pend_query, pend_steps),
    ("ACC_Flat",   flat_acc_query,   acc_steps),
    ("ACC_Graph",  graph_acc_query,  acc_steps),
]

for (label, query, nsteps) in benchmarks
    println("\n--- $label ---")
    flush(stdout)

    # Dispatch helper: Graph queries use baseline (CC + MaxSens), Flat uses defaults
    run_concreach(q) = q isa GraphPolyQuery ?
        multi_step_concreach(q; enc_func=ccEncoding!, use_crown=false) :
        multi_step_concreach(q)

    # Warmup (2 steps)
    q_warm = deepcopy(query)
    q_warm.ntime = 2
    try
        run_concreach(q_warm)
    catch e
        println("  Warmup failed: $e")
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println()
        end
        results[label] = (time_s=NaN, volume=NaN, status="warmup_failed")
        continue
    end

    # Timed run
    q = deepcopy(query)
    q.ntime = nsteps
    tstart = Dates.now()
    local reachSets, boundSets
    try
        reachSets, boundSets = run_concreach(q)
    catch e
        println("  Timed run failed: $e")
        results[label] = (time_s=NaN, volume=NaN, status="failed")
        continue
    end
    tend = Dates.now()

    elapsed_s = Dates.value(tend - tstart) / 1000.0
    vol = volume(reachSets[end])
    lo = reachSets[end].center .- reachSets[end].radius
    hi = reachSets[end].center .+ reachSets[end].radius

    results[label] = (time_s=elapsed_s, volume=vol, status="ok", low=lo, high=hi)
    println("  Time:   $(round(elapsed_s, digits=2)) s")
    println("  Volume: $vol")
    println("  Low:    $lo")
    println("  High:   $hi")
    flush(stdout)
end

# ============================================================
# Summary
# ============================================================
println("\n========== FLAT vs GRAPH SUMMARY ==========")
println("  $(rpad("Benchmark", 15)) | $(rpad("Time (s)", 12)) | $(rpad("Volume", 20)) | Status")
println("  " * "-"^65)
for (label, _, _) in benchmarks
    r = results[label]
    t_str = isnan(r.time_s) ? "N/A" : string(round(r.time_s, digits=2))
    v_str = isnan(r.volume) ? "N/A" : string(r.volume)
    println("  $(rpad(label, 15)) | $(rpad(t_str, 12)) | $(rpad(v_str, 20)) | $(r.status)")
end
