include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../reachability.jl")
using LazySets
using Dates

#TODO: Evaluates to infeasible 
ac_lead = -2.0
mu = 0.0001
control_coef = [[0],[0]]
exprList = [:(- $mu*x2^2 + -2*x3 + 2*$ac_lead ), :(- $mu*x5^2 -2*x6)]
controller = "Networks/ARCH-COMP-2023/nnet/controllerACC.nnet"

######Define ACC Dynamics#######
function acc_dynamics(x, u)
    """
    Dynamics of the ACC benchmark. 

    Args:
        x: state with form [x1, x2, x3, x4, x5, x6]
            x1 - x3 are the state of the lead vehicle
            x4 - x6 are the state of the ego vehicle
        u: control input with form [u]
            u - acceleration of the ego vehicle
    """
    dx1 = x[2] # lead vehicle position
    dx2 = x[3] # lead vehicle velocity
    dx3 = -2*x[3] + 2*ac_lead - mu*x[2]^2  # lead vehicle acceleration
    dx4 = x[5] # ego vehicle position
    dx5 = x[6] # ego vehicle velocity
    dx6 = -2*x[6] + 2*u[1] - mu*x[5]^2 # ego vehicle acceleration

    xNew = x + [dx1, dx2, dx3, dx4, dx5, dx6].*dt # Euler integration
    return xNew
end

function acc_update_rule(input_vars, overt_output_vars)
    """
    Update rule for the ACC benchmark. 

    Args:
        input_vars: dictionary containing the input variables
        overt_output_vars: dictionary containing the output variables from the overt model
    """
    integration_map = Dict(
        input_vars[1] => input_vars[2],
        input_vars[2] => input_vars[3],
        input_vars[3] => overt_output_vars[1],
        input_vars[4] => input_vars[5],
        input_vars[5] => input_vars[6],
        input_vars[6] => overt_output_vars[2]
    )

    return integration_map
end

function acc_control(model, input_vars, control_vars, output_vars, input_set, ϵ=1e-12)
    """
    Control function for the ACC benchmark.
    
    Inputs:
        model: MIP model, either concrete or symbolic
        input_vars: Input variables to the nonlinear dynamics (and hence input variables to the MIP model). Note that this could be different from the input variables needed by the network 
        control_vars: Control variables provided in variable order.  
        output_vars: Output variables of the network
        input_set: Input set for the states at the current time step

    Outputs:
        con_inp_vars: Input variables to the network in the order expected by the network
        con_inp_set: Input set for the network
        net_con_vars: Control variables expected by the network
    Takes as input the MIP model, input variables to the nonlinear dynamics (and hence input variables to the MIP model), and the control variables in 
    """

    #First define the control inputs expected by the network 
    vSet = @variable(model, [1:1], base_name = "v_set")
    tGap = @variable(model, [1:1], base_name = "tGap")
    # dRel = @variable(model, [1:1], base_name = "dRel")
    # vRel = @variable(model, [1:1], base_name = "vRel")
    @variable(model, 89 <=dRel <= 100)
    @variable(model, 1.8 <=vRel <= 2.2)
    @variable(model, 30 <= vEgo <= 30.2)

    #Constraint these to be constants/params. Should these be set as constants? 
    #TODO: Review if these are best posed as constants 
    @constraint(model, vSet .== 30.0)
    @constraint(model, tGap .== 1.40)
    @constraint(model, dRel .== input_vars[1] - input_vars[4])
    @constraint(model, vRel .== input_vars[2] - input_vars[5])

    #con_inp_vars = [vSet[1], tGap[1], input_vars[5], dRel[1], vRel[1]]
    con_inp_vars = [30.0, 1.40, vEgo, dRel, vRel]
    
    #Next, provide network order control variables to the network
    con_net_vars = control_vars[end]

    #Finally, provide input range for the network
    #TODO: Review if this difference should be interval subtraction or set difference. Big difference here 
    LBs, UBs = extrema(input_set)
    
    dRel_LB = LBs[1] - UBs[4]
    dRel_UB = UBs[1] - LBs[4]

    vRel_LB = LBs[2] - UBs[5]
    vRel_UB = UBs[2] - LBs[5]

    con_inp_set = Hyperrectangle(low=[30.0-ϵ, 1.40-ϵ, LBs[5], dRel_LB, vRel_LB], high=[30.0+ϵ, 1.40+ϵ, UBs[5], dRel_UB, vRel_UB])
    return con_inp_vars, con_inp_set, con_net_vars
end
 
#Some issues with starting at zero, start with eps
ϵ = 1e-8
domain = Hyperrectangle(low=[90,32,-ϵ,10,30,-ϵ], high=[110,32.2,ϵ,11,30.2,ϵ])
numSteps = 50
dt = 0.1

########Define Bound ACC Dynamics#######
function bound_acc(ACC; plotFlag = false)
    lbs, ubs = extrema(ACC.domain)

    #Bound first component of lead dynamics
    lbLead_sub1 = lbs[2]
    ubLead_sub1 = ubs[2]
    leadSub1 = :(2*$ac_lead - $mu*x2^2)
    leadSub1LB, leadSub1UB = interpol(bound_univariate(leadSub1, lbLead_sub1, ubLead_sub1, plotflag = plotFlag)...)

    #Bound second component of lead dynamics
    lbLead_sub2 = lbs[3]
    ubLead_sub2 = ubs[3]
    leadSub2 = :(-2*x3)
    leadSub2LB, leadSub2UB = interpol(bound_univariate(leadSub2, lbLead_sub2, ubLead_sub2, plotflag = plotFlag)..., 9)
    
    #Add a dimension to prepare for Minkowski sum
    leadSub1LB_l = addDim(leadSub1LB, 2)
    leadSub1UB_l = addDim(leadSub1UB, 2)

    leadSub2LB_l = addDim(leadSub2LB, 1)
    leadSub2UB_l = addDim(leadSub2UB, 1)

    #Minkowski sum of the bounds
    #NOTE: Unique flag added to eliminate duplicate points
    leadLB = unique(MinkSum(leadSub1LB_l, leadSub2LB_l))
    leadUB = unique(MinkSum(leadSub1UB_l, leadSub2UB_l))
    
    #Experiment with adding additional dimensions for unused variables
    # emptyList = [1,4,5,6]
    emptyList = [1]
    currList = [2,3]
    leadLB_l, leadUB_l = lift_OA(emptyList, currList, leadLB, leadUB, lbs, ubs)

    #Bound ego function
    lbs_ego = [lbs[5], lbs[6]]
    ubs_ego = [ubs[5], ubs[6]]

    #Bound first component of ego dynamics
    lbEgo_sub1 = lbs_ego[1]
    ubEgo_sub1 = ubs_ego[1]
    egoSub1 = :(-$mu*x5^2)
    egoSub1LB, egoSub1UB = interpol(bound_univariate(egoSub1, lbEgo_sub1, ubEgo_sub1, plotflag = plotFlag)...)

    #Bound second component of ego dynamics
    lbEgo_sub2 = lbs_ego[2]
    ubEgo_sub2 = ubs_ego[2]
    egoSub2 = :(-2*x6)
    egoSub2LB, egoSub2UB = interpol(bound_univariate(egoSub2, lbEgo_sub2, ubEgo_sub2, plotflag = plotFlag)..., 9)

    #Add a dimension to prepare for Minkowski sum
    egoSub1LB_l = addDim(egoSub1LB, 2)
    egoSub1UB_l = addDim(egoSub1UB, 2)

    egoSub2LB_l = addDim(egoSub2LB, 1)
    egoSub2UB_l = addDim(egoSub2UB, 1)

    #Minkowski sum of the bounds
    #NOTE: Unique flag added to eliminate duplicate points
    egoLB = unique(MinkSum(egoSub1LB_l, egoSub2LB_l))
    egoUB = unique(MinkSum(egoSub1UB_l, egoSub2UB_l))

    #Experiment with adding additional dimensions for unused variables
    # emptyList = [1,2,3,4]
    # currList = [5,6]
    emptyList = [1]
    currList = [2,3]
    egoLB_l, egoUB_l = lift_OA(emptyList, currList, egoLB, egoUB, lbs, ubs)
    if plotFlag
        xS_L = unique(Any[tup[1] for tup in leadSub1LB])
        yS_L = unique(Any[tup[1] for tup in leadSub2LB])
        xS_E = unique(Any[tup[1] for tup in egoSub1LB])
        yS_E = unique(Any[tup[1] for tup in egoSub2LB])

        surfDim_L = (size(yS_L)[1], size(xS_L)[1])
        surfDim_E = (size(yS_E)[1], size(xS_E)[1])

        leadExpr = exprList[1]
        egoExpr = exprList[2]
        plotSurf(leadExpr, leadLB, leadUB, surfDim_L, xS_L, yS_L, plotFlag)
        plotSurf(egoExpr, egoLB, egoUB, surfDim_E, xS_E, yS_E, plotFlag)
    end

    bounds = [[leadLB_l, leadUB_l], [egoLB_l, egoUB_l]]
    return bounds
end

ACC = FlatPolyProblem(
    exprList, # list of dynamics expressions
    nothing,  # decomposed form of dynamics. Done manually
    control_coef, # control coefficients
    1, # control dimension
    domain, # domain of the problem
    [:x3, :x6], # List of variables that have OVERT bounds 
    nothing, #Problem bounds. Undefined to start
    acc_update_rule, #Update rule for ACC dynamics
    acc_dynamics, #Dynamics of the ACC benchmark
    bound_acc, #Function to bound ACC dynamics
    acc_control #Control function for ACC benchmark
)

query = FlatPolyQuery(
    ACC, #Problem definition
    controller, #Path to controller network file
    Id(), #Last layer activation
    "MIP", #Solver type
    numSteps, #Number of steps to compute
    dt, #Euler integration step size
    2, #Number of OVERT points
    nothing, #var_dict, empty to start
    nothing, #mod_dict, empty to start
    3, #case (x, dx, ddx)
)

###########TEST: Getting to the Backend###############
query2 = deepcopy(query)
query2.ntime = 1

query2.problem.bounds = query2.problem.bound_func(query2.problem, plotFlag=false)
query2.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
query2.mod_dict = Dict{Symbol,JuMP.Model}()

encode_dynamics!(query2)

# #Encode the controller if it exists
# if !isnothing(query2.network_file)
#     encode_control!(query2)
# end

################################################
#Debug encode control 
    input_set = query2.problem.domain
    network_file = query2.network_file
    input_vars = []
    control_vars = []
    output_vars = []
    for sym in query2.problem.varList
        #Get dictionary of MIP variables 
        push!(input_vars, query2.var_dict[sym][1]...)
        push!(control_vars, query2.var_dict[sym][3][1]...)
        push!(output_vars, query2.var_dict[sym][2]...)
    end
    input_vars
    control_vars
    output_vars
    
    mipModel = query2.mod_dict[query2.problem.varList[1]]
    con_inp_vars, con_inp_set, con_vars = query2.problem.control_func(mipModel, input_vars, control_vars, output_vars, input_set)

    con_inp_vars
    con_inp_set
    con_vars
    #cb = add_controller_constraints!(mipModel, network_file, con_inp_set, con_inp_vars, con_vars) 

    #Read network file 
    network = read_nnet(network_file, last_layer_activation=Id())
    #Initialize neurons (adds variables)
    neurons = init_neurons(mipModel, network)
    #Initialize deltas (adds binary variables)
    deltas = init_deltas(mipModel, network)
    #Use Taylor Johnson paper (https://arxiv.org/abs/1708.03322) to get bounds  
    bounds = get_bounds(network, con_inp_set)
    #Add NN MIP model to the given model
    #This is defined in the constraints.jl file. Appears to be the Tjeng paper encoding
    encode_network!(mipModel, network, neurons, deltas, bounds, BoundedMixedIntegerLP())
    #Relate the NN variables to the dynamics variables
    neurons[1]
    neurons[end]
    #@constraint(mipModel, neurons[1] == con_inp_vars)  # set inputvars
    @constraint(mipModel, neurons[1][1:3] .== con_inp_vars[1:3])  # set inputvars
    @constraint(mipModel, con_vars .== neurons[end])  # set outputvars
    ####################################################

reach_solve(query2)
###################################
t_idx = nothing
########################################
    stateVar = query2.problem.varList
    trueInp = []
    trueOut = []
    stateVarTimed = Any[]
    
    #Compute true input and output variables 
    for sym in stateVar
        if !isnothing(t_idx)
            #Account for symbolic case where dynamics are timed
            sym_timed = Meta.parse("$(sym)_$t_idx")
            input_vars = query2.var_dict[sym_timed][1]
            output_vars = query2.var_dict[sym_timed][2]
            push!(stateVarTimed, sym_timed)
        else   
            input_vars = query2.var_dict[sym][1]
            output_vars = query2.var_dict[sym][2]
        end

        #TODO: To this more cleverly
        #NOTE: Done, avoids the need for different cases
        push!(trueInp, input_vars...)
        push!(trueOut, output_vars...)
    end
    
    integration_map = query2.problem.update_rule(trueInp, trueOut)
    
    timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)

    #Loop over symbols with OVERT approximations to compute reach steps
    sym = stateVar[2]
    for sym in stateVar
        #Account for symbolic case with timed dynamics
        if !isnothing(t_idx)
            symTimed = Meta.parse("$(sym)_$t_idx")
            input_vars = query2.var_dict[symTimed][1]
        else
            input_vars = query2.var_dict[sym][1]
        end
        mipModel = query2.mod_dict[sym]
        #TEST: remove
        v = input_vars[3]
        for v in input_vars 
            if v in trueInp
                dv = integration_map[v]
                next_v = v + query2.dt*dv
                push!(timestep_nplus1_vars, next_v)
                @objective(mipModel, Min, next_v)
                # @objective(mipModel, Max, dv)
                JuMP.optimize!(mipModel)
                termination_status(mipModel)
                @assert termination_status(mipModel) == MOI.OPTIMAL
                objective_bound(mipModel)
                push!(lows, objective_value(mipModel))
                @objective(mipModel, Max, next_v)
                JuMP.optimize!(mipModel)
                @assert termination_status(mipModel) == MOI.OPTIMAL
                objective_bound(mipModel)
                push!(highs, objective_value(mipModel))
            end
        end
    end
    #NOTE: Hyperrectangle can plot in higher dimensions as well
    reacheable_set = Hyperrectangle(low=lows, high=highs)
