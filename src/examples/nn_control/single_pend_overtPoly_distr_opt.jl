include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

#Define problem parameters
pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.0
controller = "Networks/ARCH-COMP-2023/nnet/controllerSinglePendulum.nnet"
expr = [:($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)]
control_coef = [[0], [1/(pend_mass*pend_len^2)]]
# control_coef = [[0], [0]]

numSteps = 20
domain = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
# domain = Hyperrectangle(low=[0., -0.1], high=[1, 0.1])
dt = 0.05


function single_pend_dynamics(x, u, dt)
    """
    Dynamics of the single pendulum for a single time step
    """
    dx1 = x[2]
    dx2 = (grav_const/pend_len) * sin(x[1]) + (1/(pend_mass*pend_len^2)) * u - (friction/(pend_mass*pend_len^2)) * x[2]
    
    xNew = [x[1] + dx1*dt, x[2] + dx2*dt]
    return xNew
end

#NOTE Debugging the bounds for the single pendulum
function bound_pend(SinglePendulum; plotFlag=false)
    #Define the true dynamics
    # single_pend_θ_doubledot = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)

    #get input bounds 
    lbs, ubs = extrema(SinglePendulum.domain)

    #Bound f(x1)
    lb1 = lbs[1]
    ub1 = ubs[1]
    bF1sub1 = :($(grav_const/pend_len) * sin(x1))
    bF1s1LB, bF1s1UB = interpol(bound_univariate(bF1sub1, lb1, ub1, plotflag = plotFlag)...) 

    #Bound f(x2)
    lb2 = lbs[2]
    ub2 = ubs[2]
    bF1sub2 = :($((friction)/((pend_mass)*(pend_len)^2)) *-x2)
    bF1s2LB, bF1s2UB = interpol(bound_univariate(bF1sub2, lb2, ub2, plotflag = plotFlag)...)

    #Add a dimension to prepare for Minkowski sum
    bF1s1LB_l = addDim(bF1s1LB, 2)
    bF1s1UB_l = addDim(bF1s1UB, 2)

    bF1s2LB_l = addDim(bF1s2LB, 1)
    bF1s2UB_l = addDim(bF1s2UB, 1)

    #Combine to get f(x1) + f(x2)
    bF1LB = MinkSum(bF1s1LB_l, bF1s2LB_l)
    bF1UB = MinkSum(bF1s1UB_l, bF1s2UB_l)

    #Bound angle dynamics 
    lb_θ = lbs[2]
    ub_θ = ubs[2]
    θ_dot = :(1*x2)
    θ_dot_lb_u, θ_dot_ub_u = interpol(bound_univariate(θ_dot, lb_θ, ub_θ, plotflag = plotFlag)...)
    #NOTE: Angle needs to be a function of angle as well as angular velocity
    emptyList = [1]
    currList = [2]
    lb_θ_empty = lbs[1]
    ub_θ_empty = ubs[1]
    θ_dot_lb, θ_dot_ub = lift_OA(emptyList, currList, θ_dot_lb_u, θ_dot_ub_u, lb_θ_empty, ub_θ_empty)
    
    if plotFlag
        xS = Any[tup[1] for tup in bF1s1LB]
        yS = Any[tup[1] for tup in bF1s2LB]
        surfDim = (size(yS)[1],size(xS)[1])
        exp2Plot = :($(grav_const/pend_len) * sin(x1) - $(friction/(pend_mass*pend_len^2)) * x2)
        plotSurf(exp2Plot, bF1LB, bF1UB, surfDim, xS, yS, true)
    end
    bounds = [[θ_dot_lb, θ_dot_ub],[bF1LB, bF1UB]]
    return bounds
end

function single_pend_control(input_set)
    con_inp_set = input_set
    return con_inp_set
end

function single_pend_dyn_con_link!(query, neurons, graph, dynModel, netModel,t_ind=nothing)
    #Define variables that are inputs to the network model 
    # @variable(netModel, θ)
    # @variable(netModel, dθ)
    # #Specify inputs to the network 
    # @constraint(netModel, θ == neurons[1][1])
    # @constraint(netModel, dθ == neurons[1][2])
    # #Link these variables to the appropriate models
    # @linkconstraint(graph, netModel[:θ] == dynModel[1][:x][1])
    # @linkconstraint(graph, netModel[:dθ] == dynModel[2][:x][2])

    @linkconstraint(graph, dynModel[1][:x][1] == neurons[1][1])
    @linkconstraint(graph, dynModel[2][:x][2] == neurons[1][2])

    #Link network output to pendulum torque 
    @variable(netModel, u)
    @constraint(netModel, u == neurons[end][1])
    # @linkconstraint(graph, netModel[:u] == dynModel[2][:u])
    # @linkconstraint(graph, dynModel[2][:u] == netModel[:u])
    @linkconstraint(graph, dynModel[2][:u] == neurons[end][1])

    #Iterate through dynModel and identify pertinent input variable for each 
    i = 0
    for sym in query.problem.varList
        if !isnothing(t_ind)
            sym_t = Meta.parse("$(sym)_$(t_ind)")
        else
            sym_t = sym
        end
        i += 1
        #For acceleration, the pertinent variable is the last element of the state vector (which is acceleration). For others it's the first
        if sym == :dθ 
            pertVar = dynModel[i][:x][end]
        else 
            pertVar = dynModel[i][:x][1]
        end
        #Add pertinent variable to var dict 
        push!(query.var_dict[sym_t], [pertVar])
    end 
end

function single_pend_dyn_con_linkalt!(query, neurons, graph, dynModel,t_ind=nothing)
    #Define variables that are inputs to the network model 
    # @variable(netModel, θ)
    # @variable(netModel, dθ)
    # #Specify inputs to the network 
    # @constraint(netModel, θ == neurons[1][1])
    # @constraint(netModel, dθ == neurons[1][2])
    # #Link these variables to the appropriate models
    # @linkconstraint(graph, netModel[:θ] == dynModel[1][:x][1])
    # @linkconstraint(graph, netModel[:dθ] == dynModel[2][:x][2])

    @constraint(graph, dynModel[2][:x][1] == neurons[1][1])
    @constraint(graph, dynModel[2][:x][2] == neurons[1][2])

    #Link network output to pendulum torque 
    # @variable(netModel, u)
    # @constraint(netModel, u == neurons[end][1])
    # @linkconstraint(graph, netModel[:u] == dynModel[2][:u])
    # @linkconstraint(graph, dynModel[2][:u] == netModel[:u])
    @constraint(graph, dynModel[2][:u] == neurons[end][1])

    #Iterate through dynModel and identify pertinent input variable for each 
    i = 0
    for sym in query.problem.varList
        if !isnothing(t_ind)
            sym_t = Meta.parse("$(sym)_$(t_ind)")
        else
            sym_t = sym
        end
        i += 1
        #For acceleration, the pertinent variable is the last element of the state vector (which is acceleration). For others it's the first
        if sym == :dθ 
            pertVar = dynModel[i][:x][end]
        else 
            pertVar = dynModel[i][:x][1]
        end
        #Add pertinent variable to var dict 
        push!(query.var_dict[sym_t], [pertVar])
    end 
end

SinglePendulum = DistrOvertProblem(
    expr, # dynamics
    nothing, #decomposed form of the dynamics. Done manually
    control_coef, # control coefficient
    domain, # domain
    [:θ,:dθ], #List of variables that have OVERT bounds
	nothing, #undefined bounds to start
    single_pend_dynamics,
    bound_pend,
    single_pend_control,
    single_pend_dyn_con_link!
)

query = DistrOvertQuery(
	SinglePendulum,    # problem
	controller,        # network file
	Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",             # query solver, "MIP" or "ReluPlex"
	numSteps,                # ntime
	dt,               # dt
	1,                # N_overt
    nothing,         # var_dict
    nothing,         # mod_dict
    2                # case
)


#Use concrete reachability to trace out the trajectory
query1 = deepcopy(query)
query1.ntime = 10
#@time reachSets, boundSets = multi_step_concreach(query1);


#########TEST: Debugging Distributed Reachability######
#########TEST: Preamble ###########
query1.problem.bounds = query1.problem.bound_func(query1.problem)
query1.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
query1.mod_dict = Dict{Symbol,Any}()

encode_dynamics!(query1)

# #Encode the network and link the control to the dynamics
# if !isnothing(query1.network_file)
#     neurons = encode_control!(query1)
# end

#####################################
#TEST: Try something new... encode control in dynamics directly 
input_set = query1.problem.control_func(query1.problem.domain)
network_file = query1.network_file
neurons = add_controller_constraints!(query1.mod_dict[:f][2], network_file, input_set, Id())
####################################


dyn_con_link! = query1.problem.link_func
graph = query1.mod_dict[:graph]
dynModel = query1.mod_dict[:f]
# netModel = query1.mod_dict[:u]
# dyn_con_link!(query1, neurons, graph, dynModel, netModel)
single_pend_dyn_con_linkalt!(query1, neurons, graph, dynModel)
#####TEST: Debugging conc reach solve#######
# function conc_reach_solve(query)
    max_query = deepcopy(query1)
    min_query = deepcopy(query1)
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)

    #####Prepping Models###############
    min_dynModel = min_query.mod_dict[:f]
    #min_netModel = min_query.mod_dict[:u]
    minGraph = min_query.mod_dict[:graph]

    max_dynModel = max_query.mod_dict[:f]
    maxGraph = max_query.mod_dict[:graph]
    #max_netModel = max_query.mod_dict[:u]

    #Define Minimization Objective
    i = 0
    #Compute lower bounds
    for sym in min_query.problem.varList 
        i += 1
        model = min_dynModel[i]
        v = min_query.var_dict[sym][end][1]
        dv = min_query.var_dict[sym][2][1]
        next_v_l = v + min_query.dt*dv
        # @variable(model, next_v)
        # @constraint(model, next_v == next_v_l)
        @objective(model, Min, next_v_l)
    end
    #TEST: Do we need to optimize the netmodel as well? 
    #@objective(min_netModel, Min, min_netModel[:u])

    #Define Maximization Objective
    i = 0
    for sym in max_query.problem.varList 
        i += 1
        model = max_dynModel[i]
        v = max_query.var_dict[sym][end][1]
        dv = max_query.var_dict[sym][2][1]
        next_v_u = v + max_query.dt*dv
        # @variable(model, next_v)
        # @constraint(model, next_v == next_v_u)
        @objective(model, Max, next_v_u)
    end
    #TEST: Do we need to optimize the netmodel as well?
    # @objective(max_netModel, Max, max_netModel[:u])

    set_optimizer(minGraph, Gurobi.Optimizer)
    set_optimizer(maxGraph, Gurobi.Optimizer)
    
    optimize!(minGraph)
    optimize!(maxGraph)

    @assert termination_status(minGraph) == MOI.OPTIMAL
    @assert termination_status(maxGraph) == MOI.OPTIMAL
    
    i = 0
    for _ in min_query.problem.varList
        i += 1
        push!(lows, value(objective_function(min_dynModel[i])))
    end
    
    i = 0
    #NOTE: We use negative value for the Maximization because Plasmo converts everything into a minimization problem, and that yields negative values 
    #WARN: This may cause errors, watch carefully
    for _ in max_query.problem.varList
        i += 1
        push!(highs, -value(objective_function(max_dynModel[i])))
    end
    
    # value(objective_function(max_netModel))
    reach_set = Hyperrectangle(low=lows, high=highs)
    extrema(reach_set)
#     return reach_set 
# end
#################################################
symQuery = deepcopy(query)
symQuery.problem.bounds = boundsets
reachSets = reachsets


#######Sketching out sym reach###################
symQuery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
symQuery.mod_dict = Dict{Symbol,Any}()

############Sketching out encode_sym_dynamics
#####Inputs to encode sym dynamics 
x_dim = length(symQuery.problem.varList) #state dimension

function encode_sym_dynamics!(symQuery, x_dim)
    """
    Method to encode symbolic dynamics. Takes symQuery as input
    """
    symGraph = OptiGraph()
    #####Enter time loop######
    for t_ind = 1:symQuery.ntime
        x_ind = 0
        #Create a new set of nodes for each time step
        dynNodes = @optinode(symGraph, nodes[1:x_dim])
        #####Enter Symbol loop####
        for sym in symQuery.problem.varList
            sym_t = Meta.parse("$(sym)_$(t_ind)")
            x_ind += 1
            #Get lower and upper bounds for first variable in first time step
            LB, UB = symQuery.problem.bounds[t_ind][x_ind]
            Tri = OA2PWA(LB)
            xS = [(tup[1:end-1]) for tup in LB]
            yUB = [tup[end] for tup in UB]
            yLB = [tup[end] for tup in LB]

            ccEncoding!(xS, yLB, yUB, Tri, symQuery, sym_t, x_ind, dynNodes[x_ind])
        end

        f_t = Meta.parse("f_$(t_ind)")
        symQuery.mod_dict[f_t] = dynNodes
    end
    symQuery.mod_dict[:graph] = symGraph
end

encode_sym_dynamics!(symQuery, x_dim)

######Sketching out Encode Sym Control####
function encode_sym_control!(symQuery)
    """
    Method to encode symbolic control. Takes symQuery as input
    """
    network_file = symQuery.network_file
    neurList = []
    for t_ind = 1:symQuery.ntime
        input_set = reachSets[t_ind]
        network_file = symQuery.network_file
        netModel = @optinode(symQuery.mod_dict[:graph])
        neurons = add_controller_constraints!(netModel, network_file, input_set, Id())
        u_ind = Meta.parse("u_$(t_ind)")
        symQuery.mod_dict[u_ind] = netModel
        push!(neurList, neurons)
    end
    return neurList
end
########################################
neurList = encode_sym_control!(symQuery)


#########Sketching out time encoding with dynamics/control link######
function encode_time(symQuery, neurList)
    for t_ind = 1:symQuery.ntime
        symGraph = symQuery.mod_dict[:graph]
        dynModel = symQuery.mod_dict[Meta.parse("f_$(t_ind)")]
        netModel = symQuery.mod_dict[Meta.parse("u_$(t_ind)")]

        #Link the dynamics and control first 
        symQuery.problem.link_func(symQuery, neurList[t_ind], symGraph, dynModel, netModel, t_ind)
    end

    #Next link time steps
    symGraph = symQuery.mod_dict[:graph]
    #######Enter time loop######
    #TEST: Entering time loop manually
    # t_ind = 1
    for t_ind = 1:symQuery.ntime-1
        currDyn = symQuery.mod_dict[Meta.parse("f_$(t_ind)")]
        nextDyn = symQuery.mod_dict[Meta.parse("f_$(t_ind+1)")]
        #Iterate through models and link pertinent variables 
        x_ind = 1
        #TEST: Entering the loop manually
        # sym = symQuery.problem.varList[x_ind]
        for sym in symQuery.problem.varList
            currModel = currDyn[x_ind]
            currSym = Meta.parse("$(sym)_$(t_ind)")
            nextModel = nextDyn[x_ind]
            nextSym = Meta.parse("$(sym)_$(t_ind+1)")

            xNow = symQuery.var_dict[currSym][end][1] 
            yNow = symQuery.var_dict[currSym][2][1]
            xNext = symQuery.var_dict[nextSym][end][1]

            @linkconstraint(symGraph, xNext == xNow + symQuery.dt*yNow)
            x_ind += 1
        end
    end
end

###########################
encode_time(symQuery, neurList)

##############################
#inputs to sym reach solve 
t_sym = symQuery.ntime
#######Next define Sym Reach Solve###########
function sym_reach_solve(symQuery, t_sym)
    #Ensure that the time step is within bounds
    @assert t_sym <= symQuery.ntime
    #Akin to conc_reach_solve
    max_query = deepcopy(symQuery)
    min_query = deepcopy(symQuery)
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)
    f_sym = Meta.parse("f_$(t_sym)")
    min_dynModel = min_query.mod_dict[f_sym]
    minGraph = min_query.mod_dict[:graph]
    i = 0

    #Compute lower bounds
    for sym in min_query.problem.varList
        sym_t = Meta.parse("$(sym)_$(t_sym)") 
        i += 1
        model = min_dynModel[i]
        v = min_query.var_dict[sym_t][end][1]
        dv = min_query.var_dict[sym_t][2][1]
        @variable(model, next_v)
        @constraint(model, next_v == v + query.dt*dv)
        @objective(model, Min, next_v)
    end

    set_optimizer(minGraph, Gurobi.Optimizer)
    optimize!(minGraph)
    @assert termination_status(minGraph) == MOI.OPTIMAL
    i = 0
    for _ in symQuery.problem.varList
        i += 1
        push!(lows, value(min_dynModel[i][:next_v]))
    end


    #Compute upper bounds
    max_dynModel = max_query.mod_dict[f_sym]
    maxGraph = max_query.mod_dict[:graph]
    i = 0
    for sym in query.problem.varList 
        sym_t = Meta.parse("$(sym)_$(t_sym)") 
        i += 1
        model = max_dynModel[i]
        v = max_query.var_dict[sym_t][end][1]
        dv = max_query.var_dict[sym_t][2][1]
        @variable(model, next_v)
        @constraint(model, next_v == v + max_query.dt*dv)
        @objective(model, Max, next_v)
    end

    set_optimizer(maxGraph, Gurobi.Optimizer)
    optimize!(maxGraph)
    i = 0
    for _ in query.problem.varList
        i += 1
        push!(highs, value(max_dynModel[i][:next_v]))
    end
    reach_set = Hyperrectangle(low=lows, high=highs)
    return reach_set
end


@time sym_hyp = sym_reach_solve(symQuery, t_sym)

plot(reachsets[end])
plot!(sym_hyp)