include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

ac_lead = -2.0
mu = 0.0001
control_coef = [[0],[0], [0], [0], [0],[2]]
exprList = [:(- $mu*x2^2 + -2*x3 + 2*$ac_lead ), :(- $mu*x5^2 -2*x6)]
controller = "Networks/ARCH-COMP-2023/nnet/controllerACC.nnet"


vSet = 30.0
tGap = 1.40
dDefault = 10
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

function acc_control(input_set, ϵ=1e-12)
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

    #Finally, provide input range for the network
    #TODO: Review if this difference should be interval subtraction or set difference. Big difference here 
    LBs, UBs = extrema(input_set)
    
    dRel_LB = LBs[1] - UBs[4]
    dRel_UB = UBs[1] - LBs[4]

    vRel_LB = LBs[2] - UBs[5]
    vRel_UB = UBs[2] - LBs[5]

    con_inp_set = Hyperrectangle(low=[30.0-ϵ, 1.40-ϵ, LBs[5], dRel_LB, vRel_LB], high=[30.0+ϵ, 1.40+ϵ, UBs[5], dRel_UB, vRel_UB])
    return con_inp_set
end

#Some issues with starting at zero, start with eps
ϵ = 1e-8
domain = Hyperrectangle(low=[90,32,-ϵ,10,30,-ϵ], high=[110,32.2,ϵ,11,30.2,ϵ])
numSteps = 50
dt = 0.1

########Define Bound ACC Dynamics#######
function bound_acc_old(ACC; plotFlag = false, npoint=2)
    lbs, ubs = extrema(ACC.domain)

    ##Bound Lead Car Dynamics#####
    #Bound first component of lead acceleration dynamics
    lb_a_Lead_sub1 = lbs[2]
    ub_a_Lead_sub1 = ubs[2]
    aleadSub1 = :(2*$ac_lead - $mu*x2^2)
    #The interpolation is to ensure upper and lower bounds are over the same set of points 
    aleadSub1LB, aleadSub1UB = interpol_nd(bound_univariate(aleadSub1, lb_a_Lead_sub1, ub_a_Lead_sub1, plotflag = plotFlag)...)

    # #######################
    #Bound second component of lead acceleration dynamics
    lb_a_Lead_sub2 = lbs[3]
    ub_a_Lead_sub2 = ubs[3]
    aleadSub2 = :(-2*x3)
    #Set number of digits for this one to avoid rounding out the points 
    aleadSub2LB, aleadSub2UB = interpol_nd(bound_univariate(aleadSub2, lb_a_Lead_sub2, ub_a_Lead_sub2, plotflag = plotFlag)...)

    #Add a dimension to prepare for Minkowski sum
    aleadSub1LB_l = addDim(aleadSub1LB, 2)
    aleadSub1UB_l = addDim(aleadSub1UB, 2)

    aleadSub2LB_l = addDim(aleadSub2LB, 1)
    aleadSub2UB_l = addDim(aleadSub2UB, 1)

    #Minkowski sum of the bounds
    #NOTE: Unique flag added to eliminate duplicate points
    aleadLB = unique(MinkSum(aleadSub1LB_l, aleadSub2LB_l))
    aleadUB = unique(MinkSum(aleadSub1UB_l, aleadSub2UB_l))
    
    #Bound lead veclocity dynamics 
    #Note that velocity dynamics are just acceleration
    lb_v_lead = lbs[3]
    ub_v_lead = ubs[3]
    vLead = :(1*x3)
    vLeadLB_u, vLeadUB_u = bound_univariate(vLead, lb_v_lead, ub_v_lead, plotflag = plotFlag)
    ##NOTE: Velocity needs to be a function of velocity as well. Use lifting to achieve this 
    emptyList = [1]
    currList = [2]
    #Bounds of velocity value 
    lbs_v_lead_empty = lbs[2]
    ubs_v_lead_empty = ubs[2]
    vLeadLB, vLeadUB = lift_OA(emptyList, currList, vLeadLB_u, vLeadUB_u, lbs_v_lead_empty, ubs_v_lead_empty)
    
    #Bound lead position dynamics
    lb_p_lead = lbs[2]
    ub_p_lead = ubs[2]
    pLead = :(1*x2)
    pLeadLB_u, pLeadUB_u = bound_univariate(pLead, lb_p_lead, ub_p_lead, plotflag = plotFlag)
    ##NOTE: Position needs to be a function of position as well. Use lifting to achieve this
    emptyList = [1]
    currList = [2]
    #Bounds of position value
    lbs_p_lead_empty = lbs[1]
    ubs_p_lead_empty = ubs[1]
    pLeadLB, pLeadUB = lift_OA(emptyList, currList, pLeadLB_u, pLeadUB_u, lbs_p_lead_empty, ubs_p_lead_empty)
    

    ###Bound Ego Car Dynamics#####
    #Bound first component of ego dynamics
    lb_a_Ego_sub1 = lbs[5]
    ub_a_Ego_sub1 = ubs[5]
    aegoSub1 = :(-$mu*x5^2)
    #NOTE: Interpol is to ensure that the bounds are defined over same set of points
    aegoSub1LB, aegoSub1UB = interpol_nd(bound_univariate(aegoSub1, lb_a_Ego_sub1, ub_a_Ego_sub1, plotflag = plotFlag)...)

    #Bound second component of ego dynamics
    lb_a_Ego_sub2 = lbs[6]
    ub_a_Ego_sub2 = ubs[6]
    aegoSub2 = :(-2*x6)
    #Set number of digits to be 9 bc we're dealing with very small numbers here
    aegoSub2LB, aegoSub2UB = interpol_nd(bound_univariate(aegoSub2, lb_a_Ego_sub2, ub_a_Ego_sub2, plotflag = plotFlag)...)

    #Add a dimension to prepare for Minkowski sum
    aegoSub1LB_l = addDim(aegoSub1LB, 2)
    aegoSub1UB_l = addDim(aegoSub1UB, 2)

    aegoSub2LB_l = addDim(aegoSub2LB, 1)
    aegoSub2UB_l = addDim(aegoSub2UB, 1)

    #Minkowski sum of the bounds
    #NOTE: Unique flag added to eliminate duplicate points
    aegoLB = unique(MinkSum(aegoSub1LB_l, aegoSub2LB_l))
    aegoUB = unique(MinkSum(aegoSub1UB_l, aegoSub2UB_l))

    #Bound ego velocity dynamics
    lb_v_ego = lbs[6]
    ub_v_ego = ubs[6]
    vEgo = :(1*x6)
    vEgoLB_u, vEgoUB_u = bound_univariate(vEgo, lb_v_ego, ub_v_ego, plotflag = plotFlag)
    ##NOTE: Velocity needs to be a function of velocity as well. Use lifting to achieve this
    emptyList = [1]
    currList = [2]
    #Bounds of velocity value
    lbs_v_ego_empty = lbs[5]
    ubs_v_ego_empty = ubs[5]
    vEgoLB, vEgoUB = lift_OA(emptyList, currList, vEgoLB_u, vEgoUB_u, lbs_v_ego_empty, ubs_v_ego_empty)

    #Bound ego position dynamics
    lb_p_ego = lbs[5]
    ub_p_ego = ubs[5]
    pEgo = :(1*x5)
    pEgoLB_u, pEgoUB_u = bound_univariate(pEgo, lb_p_ego, ub_p_ego, plotflag = plotFlag)
    ##NOTE: Position needs to be a function of position as well. Use lifting to achieve this
    emptyList = [1]
    currList = [2]
    #Bounds of position value
    lbs_p_ego_empty = lbs[4]
    ubs_p_ego_empty = ubs[4]
    pEgoLB, pEgoUB = lift_OA(emptyList, currList, pEgoLB_u, pEgoUB_u, lbs_p_ego_empty, ubs_p_ego_empty)
    
    if plotFlag
        xS_L = unique(Any[tup[1] for tup in aleadSub1LB])
        yS_L = unique(Any[tup[1] for tup in aleadSub2LB])
        xS_E = unique(Any[tup[1] for tup in aegoSub1LB])
        yS_E = unique(Any[tup[1] for tup in aegoSub2LB])

        surfDim_L = (size(yS_L)[1], size(xS_L)[1])
        surfDim_E = (size(yS_E)[1], size(xS_E)[1])

        leadExpr = exprList[1]
        egoExpr = exprList[2]
        plotSurf(leadExpr, aleadLB, aleadUB, surfDim_L, xS_L, yS_L, plotFlag)
        plotSurf(egoExpr, aegoLB, aegoUB, surfDim_E, xS_E, yS_E, plotFlag)
    end

    #Return bounds in variable order 
    bounds = [[pLeadLB, pLeadUB], [vLeadLB, vLeadUB], [aleadLB, aleadUB], [pEgoLB, pEgoUB], [vEgoLB, vEgoUB], [aegoLB, aegoUB]]
    return bounds
end

function bound_acc(ACC; plotFlag = false, npoint=2)
    lbs, ubs = extrema(ACC.domain)

    ##Bound Lead Car Dynamics#####
    #Bound first component of lead acceleration dynamics
    lb_a_Lead_sub1 = lbs[2]
    ub_a_Lead_sub1 = ubs[2]
    aleadSub1 = :(2*$ac_lead - $mu*x2^2)
    #The interpolation is to ensure upper and lower bounds are over the same set of points 
    aleadSub1LB, aleadSub1UB = interpol_nd(bound_univariate(aleadSub1, lb_a_Lead_sub1, ub_a_Lead_sub1, plotflag = plotFlag)...)

    # #######################
    #Bound second component of lead acceleration dynamics
    lb_a_Lead_sub2 = lbs[3]
    ub_a_Lead_sub2 = ubs[3]
    aleadSub2 = :(-2*x3)
    #Set number of digits for this one to avoid rounding out the points 
    aleadSub2LB, aleadSub2UB = interpol_nd(bound_univariate(aleadSub2, lb_a_Lead_sub2, ub_a_Lead_sub2, plotflag = plotFlag)...)

    #Lift bounds to same space
    emptyList = [2] #f(x1) missing v
    currList = [1]
    l_aleadSub1LB, l_aleadSub1UB = lift_OA(emptyList, currList, aleadSub1LB, aleadSub1UB, lbs[2:3], ubs[2:3])

    #Lift f2
    emptyList = [1] #f(x2) missing x
    currList = [2]
    l_aleadSub2LB, l_aleadSub2UB = lift_OA(emptyList, currList, aleadSub2LB, aleadSub2UB, lbs[2:3], ubs[2:3])

    #Combine to get f(x1) + f(x2)
    aleadLB, aleadUB = sumBounds(l_aleadSub1LB, l_aleadSub1UB, l_aleadSub2LB, l_aleadSub2UB, false)
    
    #Bound lead veclocity dynamics 
    #Note that velocity dynamics are just acceleration
    lb_v_lead = lbs[3]
    ub_v_lead = ubs[3]
    vLead = :(1*x3)
    vLeadLB_u, vLeadUB_u = bound_univariate(vLead, lb_v_lead, ub_v_lead, plotflag = plotFlag)
    ##NOTE: Velocity needs to be a function of velocity as well. Use lifting to achieve this 
    emptyList = [1]
    currList = [2]
    #Bounds of velocity value 
    lbs_v_lead_empty = lbs[2]
    ubs_v_lead_empty = ubs[2]
    vLeadLB, vLeadUB = lift_OA(emptyList, currList, vLeadLB_u, vLeadUB_u, lbs_v_lead_empty, ubs_v_lead_empty)
    
    #Bound lead position dynamics
    lb_p_lead = lbs[2]
    ub_p_lead = ubs[2]
    pLead = :(1*x2)
    pLeadLB_u, pLeadUB_u = bound_univariate(pLead, lb_p_lead, ub_p_lead, plotflag = plotFlag)
    ##NOTE: Position needs to be a function of position as well. Use lifting to achieve this
    emptyList = [1]
    currList = [2]
    #Bounds of position value
    lbs_p_lead_empty = lbs[1]
    ubs_p_lead_empty = ubs[1]
    pLeadLB, pLeadUB = lift_OA(emptyList, currList, pLeadLB_u, pLeadUB_u, lbs_p_lead_empty, ubs_p_lead_empty)
    

    ###Bound Ego Car Dynamics#####
    #Bound first component of ego dynamics
    lb_a_Ego_sub1 = lbs[5]
    ub_a_Ego_sub1 = ubs[5]
    aegoSub1 = :(-$mu*x5^2)
    #NOTE: Interpol is to ensure that the bounds are defined over same set of points
    aegoSub1LB, aegoSub1UB = interpol_nd(bound_univariate(aegoSub1, lb_a_Ego_sub1, ub_a_Ego_sub1, plotflag = plotFlag)...)

    #Bound second component of ego dynamics
    lb_a_Ego_sub2 = lbs[6]
    ub_a_Ego_sub2 = ubs[6]
    aegoSub2 = :(-2*x6)
    #Set number of digits to be 9 bc we're dealing with very small numbers here
    aegoSub2LB, aegoSub2UB = interpol_nd(bound_univariate(aegoSub2, lb_a_Ego_sub2, ub_a_Ego_sub2, plotflag = plotFlag)...)

    #Lift bounds to same space
    emptyList = [2] #f(x1) missing v
    currList = [1]
    l_aegoSub1LB, l_aegoSub1UB = lift_OA(emptyList, currList, aegoSub1LB, aegoSub1UB, lbs[5:6], ubs[5:6])

    #Lift f2
    emptyList = [1] #f(x2) missing x
    currList = [2]
    l_aegoSub2LB, l_aegoSub2UB = lift_OA(emptyList, currList, aegoSub2LB, aegoSub2UB, lbs[5:6], ubs[5:6])

    #Combine to get f(x1) + f(x2)
    aegoLB, aegoUB = sumBounds(l_aegoSub1LB, l_aegoSub1UB, l_aegoSub2LB, l_aegoSub2UB, false)

    #Bound ego velocity dynamics
    lb_v_ego = lbs[6]
    ub_v_ego = ubs[6]
    vEgo = :(1*x6)
    vEgoLB_u, vEgoUB_u = bound_univariate(vEgo, lb_v_ego, ub_v_ego, plotflag = plotFlag)
    ##NOTE: Velocity needs to be a function of velocity as well. Use lifting to achieve this
    emptyList = [1]
    currList = [2]
    #Bounds of velocity value
    lbs_v_ego_empty = lbs[5]
    ubs_v_ego_empty = ubs[5]
    vEgoLB, vEgoUB = lift_OA(emptyList, currList, vEgoLB_u, vEgoUB_u, lbs_v_ego_empty, ubs_v_ego_empty)

    #Bound ego position dynamics
    lb_p_ego = lbs[5]
    ub_p_ego = ubs[5]
    pEgo = :(1*x5)
    pEgoLB_u, pEgoUB_u = bound_univariate(pEgo, lb_p_ego, ub_p_ego, plotflag = plotFlag)
    ##NOTE: Position needs to be a function of position as well. Use lifting to achieve this
    emptyList = [1]
    currList = [2]
    #Bounds of position value
    lbs_p_ego_empty = lbs[4]
    ubs_p_ego_empty = ubs[4]
    pEgoLB, pEgoUB = lift_OA(emptyList, currList, pEgoLB_u, pEgoUB_u, lbs_p_ego_empty, ubs_p_ego_empty)
    
    if plotFlag
        xS_L = unique(Any[tup[1] for tup in aleadSub1LB])
        yS_L = unique(Any[tup[1] for tup in aleadSub2LB])
        xS_E = unique(Any[tup[1] for tup in aegoSub1LB])
        yS_E = unique(Any[tup[1] for tup in aegoSub2LB])

        surfDim_L = (size(yS_L)[1], size(xS_L)[1])
        surfDim_E = (size(yS_E)[1], size(xS_E)[1])

        leadExpr = exprList[1]
        egoExpr = exprList[2]
        plotSurf(leadExpr, aleadLB, aleadUB, surfDim_L, xS_L, yS_L, plotFlag)
        plotSurf(egoExpr, aegoLB, aegoUB, surfDim_E, xS_E, yS_E, plotFlag)
    end

    #Return bounds in variable order 
    bounds = [[pLeadLB, pLeadUB], [vLeadLB, vLeadUB], [aleadLB, aleadUB], [pEgoLB, pEgoUB], [vEgoLB, vEgoUB], [aegoLB, aegoUB]]
    return bounds
end

#####Problem Specific Function#######
###Function to link control and relevant dynamics
function acc_dyn_con_link!(query, neurons, graph, dynModel, netModel)

    #Defining ACC network required inputs. First two inputs are constants 
    @constraint(netModel, neurons[1][1] == vSet)
    @constraint(netModel, neurons[1][2] == tGap) 

    #Define vEgo link using vEgo from the ego velocity model  
    #First, define vEgo variable in the neural network model
    @variable(netModel, vEgo)
    #Next, link vEgo to the velocity of the ego vehicle in the dynamics model
    #NOTE: We are now using the ego acceleration vEgo
    @linkconstraint(graph, netModel[:vEgo] == dynModel[6][:x][1])
    #Finally, link the vEgo variable to the neural network input
    @constraint(netModel, neurons[1][3] == vEgo)

    #Define dRel for ACC
    @variable(netModel, dRel)
    #In words, the dRel variable in netnode is the difference between xLead (used as input for the first dynamics model) and xEgo (used as input for the fourth dynamics model)
    @linkconstraint(graph, netModel[:dRel] == dynModel[1][:x][1] - dynModel[4][:x][1])
    @constraint(netModel, neurons[1][4] == dRel)


    #Define vRel for ACC 
    @variable(netModel, vRel)
    #In words, the vRel variable in netnode is the difference between vLead  and vEgo
    @linkconstraint(graph, netModel[:vRel] == dynModel[2][:x][1] - dynModel[5][:x][1])
    @constraint(netModel, neurons[1][5] == vRel)

    ##Also link network output to ego acceleration
    @variable(netModel, u)
    @constraint(netModel, neurons[end][1] == u)
    @linkconstraint(graph, netModel[:u] == dynModel[6][:u])

    #Iterate through dynModel and identify pertinent input variable for each
    i = 0 
    for sym in query.problem.varList
        i += 1
        #For acceleration, the pertinent variable is the last element of the state vector (which is acceleration). For others it's the first
        if sym == :x3 || sym == :x6
            pertVar = dynModel[i][:x][end]
        else 
            pertVar = dynModel[i][:x][1]
        end
        #Add pertinent variable to var dict 
        push!(query.var_dict[sym], [pertVar])
    end

end
##############################################
ACC = GraphPolyProblem(
    exprList, # list of dynamics expressions
    nothing,  # decomposed form of dynamics. Done manually
    control_coef, # control coefficients
    domain, # domain of the problem
    [:x1,:x2,:x3,:x4,:x5,:x6], # List of variables that have OVERT bounds 
    nothing, #Problem bounds. Undefined to start
    acc_dynamics, #Dynamics of the ACC benchmark
    bound_acc, #Function to bound ACC dynamics
    acc_control, #Control function for ACC benchmark
    acc_dyn_con_link! #link function for ACC benchmark
)
query = GraphPolyQuery(
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

###################
query0 = deepcopy(query);
@time reachSets, boundSets = concreach!(query0);

query00 = deepcopy(query);
query00.problem.bound_func = bound_acc_old;
@time reachSets_old, boundSets_old = concreach!(query00);

reachSets_old ⊆ reachSets
reachSets ⊆ reachSets_old

#########################
query1 = deepcopy(query);
query1.ntime = 50;
@time reachSets, boundSets = multi_step_concreach(query1);

volume(reachSets[end])

query11 = deepcopy(query);
query11.problem.bound_func = bound_acc_old;
query11.ntime = 50;
@time reachSets_old, boundSets_old = multi_step_concreach(query11);

trueFlag = true
for i = 1:50
    if !(reachSets_old[i] ⊆ reachSets[i])
        println("Unexpected behavior at time $i")
        trueFlag = false
        break
    end
end

trueFlag
volume(reachSets[end])
t = 50
#Lead
plot(project(reachSets[t], [1,2]), label="GraphReach")
plot!(project(reachsets[t], [1,2]), label="FlatReach")
#plot!(project(overtSet, [1,2]), label="OVERT")

plot(project(reachSets[t], [2,3]), label="GraphReach")
plot!(project(reachsets[t], [2,3]), label="FlatReach")
#plot!(project(overtSet, [2,3]), label="OVERT")

#Ego
plot(project(reachSets[t], [4,5]), label="GraphReach")
plot!(project(reachsets[t], [4,5]), label="FlatReach")
#plot!(project(overtSet, [4,5]), label="OVERT")

plot(project(reachSets[t], [5,6]), label="GraphReach")
plot!(project(reachsets[t], [5,6]), label="FlatReach")
#plot!(project(overtSet, [5,6]), label="OVERT")
#Verifying the property


dRel = Any[]
dSafe = Any[]
for reachset in reachSets
    reachInts = extrema(reachset)
    dRel_min = minimum([reachInts[1][1] - reachInts[1][4], reachInts[1][1] - reachInts[2][4], reachInts[2][1] - reachInts[1][4], reachInts[2][1] - reachInts[2][4]])
    dRel_max = maximum([reachInts[1][1] - reachInts[1][4], reachInts[1][1] - reachInts[2][4], reachInts[2][1] - reachInts[1][4], reachInts[2][1] - reachInts[2][4]])
    dSafe_min = dDefault + tGap*reachInts[1][5]
    dSafe_max = dDefault + tGap*reachInts[2][5]
    # vRel_min = minimum([reachInts[1][2] - reachInts[1][5], reachInts[1][2] - reachInts[2][5], reachInts[2][2] - reachInts[1][5], reachInts[2][2] - reachInts[2][5]])
    # vRel_max = maximum([reachInts[1][2] - reachInts[1][5], reachInts[1][2] - reachInts[2][5], reachInts[2][2] - reachInts[1][5], reachInts[2][2] - reachInts[2][5]])
    # dRel_hyp = Hyperrectangle(low=[dRel_min, vRel_min], high=[dRel_max, vRel_max])
    dRel_hyp = Hyperrectangle(low=[dRel_min], high=[dRel_max])
    dSafe_hyp = Hyperrectangle(low=[dSafe_min], high=[dSafe_max])
    push!(dRel, dRel_hyp)
    push!(dSafe, dSafe_hyp)
end

(extrema(dRel[end])[2] - extrema(dRel[end])[1])[1]

tstart = Dates.now()
veriFlag = true

for i = 1:51
    if !isdisjoint(dRel[i], dSafe[i])
        veriFlag = false
        println("Property violated at time $i")
        break
    end
    if i == 51
        println("Property verified")
    end
end
tend = Dates.now()
println("Verification time: $(tend-tstart)")
