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
domain = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
# domain = Hyperrectangle(low=[0., -0.1], high=[1, 0.1])
dt = 0.05
numSteps = 1/dt


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
function bound_pend_old(SinglePendulum; plotFlag=false)
    #Define the true dynamics
    # single_pend_θ_doubledot = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)

    #get input bounds 
    lbs, ubs = extrema(SinglePendulum.domain)

    #Bound f(x1)
    lb1 = lbs[1]
    ub1 = ubs[1]
    bF1sub1 = :($(grav_const/pend_len) * sin(x1))
    bF1s1LB, bF1s1UB = interpol_nd(bound_univariate(bF1sub1, lb1, ub1, plotflag = plotFlag)...) 

    #Bound f(x2)
    lb2 = lbs[2]
    ub2 = ubs[2]
    bF1sub2 = :($((friction)/((pend_mass)*(pend_len)^2)) *-x2)
    bF1s2LB, bF1s2UB = interpol_nd(bound_univariate(bF1sub2, lb2, ub2, plotflag = plotFlag)...)

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
    θ_dot_lb_u, θ_dot_ub_u = interpol_nd(bound_univariate(θ_dot, lb_θ, ub_θ, plotflag = plotFlag)...)
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

function bound_pend(SinglePendulum; plotFlag=false)
    #Define the true dynamics
    # single_pend_θ_doubledot = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)

    #get input bounds 
    lbs, ubs = extrema(SinglePendulum.domain)

    #Bound f(x1)
    lb1 = lbs[1]
    ub1 = ubs[1]
    bF1sub1 = :($(grav_const/pend_len) * sin(x1))
    bF1s1LB, bF1s1UB = interpol_nd(bound_univariate(bF1sub1, lb1, ub1, plotflag = plotFlag)...) 

    #Bound f(x2)
    lb2 = lbs[2]
    ub2 = ubs[2]
    bF1sub2 = :($((friction)/((pend_mass)*(pend_len)^2)) *-x2)
    bF1s2LB, bF1s2UB = interpol_nd(bound_univariate(bF1sub2, lb2, ub2, plotflag = plotFlag)...)

    #Lift bounds to same space
    emptyList = [2] #f(x1) missing theta_dot
    currList = [1]
    l_bF1s1LB, l_bF1s1UB = lift_OA(emptyList, currList, bF1s1LB, bF1s1UB, lbs, ubs)

    #Lift f2
    emptyList = [1] #f(x2) missing theta
    currList = [2]
    l_bF1s2LB, l_bF1s2UB = lift_OA(emptyList, currList, bF1s2LB, bF1s2UB, lbs, ubs)

    #Combine to get f(x1) + f(x2)
    bF1LB, bF1UB = sumBounds(l_bF1s1LB, l_bF1s1UB, l_bF1s2LB, l_bF1s2UB, false)

    #Bound angle dynamics 
    lb_θ = lbs[2]
    ub_θ = ubs[2]
    θ_dot = :(1*x2)
    θ_dot_lb_u, θ_dot_ub_u = interpol_nd(bound_univariate(θ_dot, lb_θ, ub_θ, plotflag = plotFlag)...)
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
    #TEST: Only link node 2 to the controller node 
    @linkconstraint(graph, dynModel[2][:x][1] == neurons[1][1])
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

SinglePendulum = GraphPolyProblem(
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

query = GraphPolyQuery(
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
query1.ntime = 20

tstart = Dates.now()
reachSets, boundSets = multi_step_concreach(query1)
tend = Dates.now()
println("Time taken to compute concrete reach: ", tend-tstart)

extrema(reachSets[11])
volume(reachSets[11])
######Testing the sym reach################
symQuery = deepcopy(query)
symQuery.problem.bounds = boundSets

#Define variable dependency matrix. A vector of vectors 
depMat = [[1,1], [1,1]]
t_sym = 10

tstart = Dates.now()
sym_set = symreach(symQuery, reachSets, depMat, t_sym)
tend = Dates.now()
println("Time taken to compute symbolic reach at time step $(t_sym): ", tend-tstart)

sym_set ⊆ reachSets[t_sym + 1]
######Testing hyb reach################
hybQuery = deepcopy(query)
@time reach_set = hybreach(hybQuery, depMat, t_sym, reachSets, boundSets)

#####Testing multi step hyb reach################
hybQuery2 = deepcopy(query)
conc_int = [10,5,5]
@time reach_set2 = multi_step_hybreach(hybQuery2, depMat, conc_int)
################################################
