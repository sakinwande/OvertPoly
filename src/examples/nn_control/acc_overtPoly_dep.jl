include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../reachability.jl")
using LazySets
using Dates

ac_lead = -2.0
mu = 0.0001
control_coef = [[0],[2]]
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
        input_vars[3] => overt_output_vars[1][1],
        input_vars[4] => input_vars[5],
        input_vars[5] => input_vars[6],
        input_vars[6] => overt_output_vars[2][1]
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
    dRel = @variable(model, [1:1], base_name = "dRel")
    vRel = @variable(model, [1:1], base_name = "vRel")

    #Constraint these to be constants/params. Should these be set as constants? 
    #TODO: Review if these are best posed as constants 
    @constraint(model, vSet .== 30.0)
    @constraint(model, tGap .== 1.40)
    @constraint(model, dRel .== input_vars[1] - input_vars[4])
    @constraint(model, vRel .== input_vars[2] - input_vars[5])

    con_inp_vars = [vSet[1], tGap[1], input_vars[5], dRel[1], vRel[1]]
    
    #Next, provide network order control variables to the network
    con_net_vars = [control_vars]

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
numSteps = 1
dt = 0.1
plotFlag = false
lbs, ubs = extrema(domain)
########Define Bound ACC Dynamics#######
function bound_acc(ACC; plotFlag = false)
    lbs, ubs = extrema(ACC.domain)

    #Bound lead function 
    lbs_lead = [lbs[2], lbs[3]]
    ubs_lead = [ubs[2], ubs[3]]

    #Bound first component of lead dynamics
    lbLead_sub1 = lbs[2]
    ubLead_sub1 = ubs[2]
    leadSub1 = :(2*$ac_lead - $mu*x2^2)
    leadSub1LB, leadSub1UB = bound_univariate(leadSub1, lbLead_sub1, ubLead_sub1, plotflag = plotFlag)

    #Bound second component of lead dynamics
    #TODO: remove plotflag
    lbLead_sub2 = lbs[3]
    ubLead_sub2 = ubs[3]
    leadSub2 = :(-2*x3)
    leadSub2LB, leadSub2UB = bound_univariate(leadSub2, lbLead_sub2, ubLead_sub2, plotflag = plotFlag)

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
    egoSub1 = :($mu*x5^2)
    egoSub1LB, egoSub1UB = bound_univariate(egoSub1, lbEgo_sub1, ubEgo_sub1, plotflag = plotFlag)

    #Bound second component of ego dynamics
    lbEgo_sub2 = lbs_ego[2]
    ubEgo_sub2 = ubs_ego[2]
    egoSub2 = :(-2*x6)
    egoSub2LB, egoSub2UB = bound_univariate(egoSub2, lbEgo_sub2, ubEgo_sub2, plotflag = plotFlag)

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

ACC = OvertPProblem(
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

query = OvertPQuery(
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

#####Debugging Concreach again########
query.problem.bounds = query.problem.bound_func(query.problem)
query.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
query.mod_dict = Dict{Symbol,JuMP.Model}()
encode_dynamics!(query)

#Encode the controller if it exists
if !isnothing(query.network_file)
    encode_control!(query)
end

reachSet =  reach_solve(query)
return reachSet, query.problem.bounds