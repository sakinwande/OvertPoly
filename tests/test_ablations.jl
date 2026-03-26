include("../src/overtPoly_helpers.jl")
include("../src/nn_mip_encoding.jl")
include("../src/overtPoly_to_mip.jl")
include("../src/overt_to_pwa.jl")
include("../src/problems.jl")
include("../src/distr_reachability.jl")
using LazySets
using Dates
using Plasmo

# ============================================================
# TORA problem definition (from L4DC benchmark)
# ============================================================
tora_control_coef = [[0],[0],[0],[1]]
tora_controller = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerTORA.nnet")
tora_exprList = [:(1*x2), :(-x1 + 0.1*sin(x3)), :(1*x4), :(1*u)]

function tora_dynamics(x, u)
    dx1 = x[2]; dx2 = -x[1] + 0.1*sin(x[3]); dx3 = x[4]; dx4 = u
    return x + [dx1, dx2, dx3, dx4] .* 0.1
end

function tora_control(input_set); return input_set; end

tora_domain = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
tora_dt = 0.1
tora_numSteps = 20

function bound_tora(TORA; plotFlag=false, npoint=2)
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
    @variable(netModel, x1); @variable(netModel, x2); @variable(netModel, x3); @variable(netModel, x4)
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

TORA = GraphPolyProblem(tora_exprList, nothing, tora_control_coef, tora_domain,
    [:x1,:x2,:x3,:x4], nothing, tora_dynamics, bound_tora, tora_control, tora_dyn_con_link!)

tora_query = GraphPolyQuery(TORA, tora_controller, Id(), "MIP", tora_numSteps, tora_dt, 2, nothing, nothing, 2)

# ============================================================
# Unicycle problem definition (from L4DC benchmark)
# ============================================================
uni_control_coef = [[0],[0],[1],[1]]
uni_controller = joinpath(@__DIR__, "..", "Networks", "ARCH-COMP-2023", "nnet", "controllerUnicycle.nnet")
uni_exprList = [:(cos(x3)*x4), :(sin(x3)*x4), :(0*x3), :(0*x4)]

function unicycle_dynamics(x, u)
    dx1 = x[4]*cos(x[3]); dx2 = x[4]*sin(x[3]); dx3 = u[2]; dx4 = u[1]
    return x + [dx1, dx2, dx3, dx4] .* 0.2
end
function unicycle_control(input_set); return input_set; end

uni_dt = 0.2
uni_numSteps = 50
uni_w = 1e-4
uni_dig = 15
uni_domain = Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high=[9.55,-4.45,2.11,1.51])
uni_depMat = [[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]

function bound_unicycle(Unicycle; plotFlag=false, npoint=2)
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

Unicycle = GraphPolyProblem(uni_exprList, nothing, uni_control_coef, uni_domain,
    [:x1,:x2,:x3,:x4], nothing, unicycle_dynamics, bound_unicycle, unicycle_control, unicycle_dyn_con_link!)

uni_query = GraphPolyQuery(Unicycle, uni_controller, Id(), "MIP", uni_numSteps, uni_dt, 2, nothing, nothing, 2)

# ============================================================
# Ablation runner
# ============================================================

ablations = [
    ("Baseline",   ccEncoding!,  false),
    ("CROWN_only", ccEncoding!,  true),
    ("DCC_only",   dccEncoding!, false),
    ("CROWN_DCC",  dccEncoding!, true),
]

# ============================================================
# TORA ablations
# ============================================================
println("\n========== TORA ABLATIONS ==========")
tora_results = Dict()

for (label, enc_func, use_crown) in ablations
    println("\n--- TORA: $label ---")

    # Untimed warmup (2 steps) — triggers Julia specialization for this
    # (enc_func, use_crown) combination before the timed run.
    q_warm = deepcopy(tora_query)
    q_warm.ntime = 2
    multi_step_concreach(q_warm; enc_func=enc_func, use_crown=use_crown)

    # Timed run (20 steps)
    q = deepcopy(tora_query)
    q.ntime = 20
    tstart = Dates.now()
    reachSets, _ = multi_step_concreach(q; enc_func=enc_func, use_crown=use_crown)
    tend = Dates.now()

    elapsed_s = Dates.value(tend - tstart) / 1000.0
    vol = volume(reachSets[end])
    final_set = reachSets[end]

    tora_results[label] = (time_s=elapsed_s, volume=vol, final_set=final_set)
    println("  Time:   $(elapsed_s) s")
    println("  Volume: $vol")
    println("  Low:    $(final_set.center .- final_set.radius)")
    println("  High:   $(final_set.center .+ final_set.radius)")
end

# ============================================================
# Unicycle ablations (batched hybrid: 5 warmup + 5×10 timed)
# ============================================================
println("\n========== UNICYCLE ABLATIONS ==========")
uni_results = Dict()

for (label, enc_func, use_crown) in ablations
    println("\n--- Unicycle: $label ---")

    # Untimed warmup (5 conc + 5 sym) — triggers Julia specialization for this

    # (enc_func, use_crown) combination before the timed run.
    cq_w = deepcopy(uni_query); sq_w = deepcopy(uni_query)
    cq_w.ntime = 5; sq_w.ntime = 5
    concRS_w, BoundS_w = multi_step_concreach(cq_w; enc_func=enc_func, use_crown=use_crown)
    sq_w.problem.bounds = BoundS_w
    symreach(sq_w, concRS_w, uni_depMat, 5; enc_func=enc_func, use_crown=use_crown)

    # Timed run: 5 batches of 10 conc + 10 sym
    cq = deepcopy(uni_query); sq = deepcopy(uni_query)
    cq.ntime = 10; sq.ntime = 10
    symReachList = []

    tstart = Dates.now()
    for batch = 1:5
        concReachSets, BoundSets = multi_step_concreach(cq; enc_func=enc_func, use_crown=use_crown)
        sq.problem.bounds = BoundSets
        sym_set = symreach(sq, concReachSets, uni_depMat, 10; enc_func=enc_func, use_crown=use_crown)
        push!(symReachList, sym_set)
        t_batch = Dates.now()
        println("  Batch $batch ($(batch*10) hybrid steps): $(Dates.value(t_batch-tstart)/1000.0) s")
        cq.problem.domain = sym_set
    end
    tend = Dates.now()

    elapsed_s = Dates.value(tend - tstart) / 1000.0
    final_sym = symReachList[end]
    vol = volume(final_sym)

    uni_results[label] = (time_s=elapsed_s, volume=vol, final_set=final_sym)
    println("  Total time: $(elapsed_s) s")
    println("  Volume:     $vol")
    println("  Low:        $(final_sym.center .- final_sym.radius)")
    println("  High:       $(final_sym.center .+ final_sym.radius)")
end

# ============================================================
# Summary
# ============================================================
println("\n========== TORA ABLATION SUMMARY ==========")
println("\nTORA (20 concrete steps, dt=0.1):")
println("  $(rpad("Configuration", 15)) | $(rpad("Time (s)", 12)) | Volume")
println("  " * "-"^50)
for (label, _, _) in ablations
    r = tora_results[label]
    println("  $(rpad(label, 15)) | $(rpad(string(round(r.time_s, digits=2)), 12)) | $(r.volume)")
end

println("\n========== UNICYCLE ABLATION SUMMARY ==========")
println("\nUnicycle (50 hybrid steps, dt=0.2, 5×[10 conc + 10 sym]):")
println("  $(rpad("Configuration", 15)) | $(rpad("Time (s)", 12)) | Volume")
println("  " * "-"^50)
for (label, _, _) in ablations
    r = uni_results[label]
    println("  $(rpad(label, 15)) | $(rpad(string(round(r.time_s, digits=2)), 12)) | $(r.volume)")
end
