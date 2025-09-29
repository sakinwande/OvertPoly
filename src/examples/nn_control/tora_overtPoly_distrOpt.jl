include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

#NOTE: Controller used is unclear :(. Start w. small controller to be safe 
#NOTE: Now we know, it's the large network :|
#NOTE: Focus on spec. 1 since it seems to be the one relevant to ReLU net
control_coef = [[0],[0],[0],[1]]
controller = "Networks/ARCH-COMP-2023/nnet/controllerTORA.nnet"
exprList = [:(1*x2), :(-x1 + 0.1*sin(x3)), :(1*x4), :(1*u)]

##Define TORA Dynamics#####
#TODO: Explain what each dimension means 
function tora_dynamics(x, u)
    """
    Dynamics of the TORA benchmark. NGL I don't know why I use this
    """
    dx1 = x[2]
    dx2 = -x[1] + 0.1*sin(x[3])
    dx3 = x[4]
    dx4 = u

    xNew = x + [dx1, dx2, dx3, dx4].*dt

    return xNew
end

function tora_control(input_set)
    con_inp_set = input_set 
    return con_inp_set
end
domain = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high = [0.7, -0.6, -0.3, 0.6])
#TODO: Decide on step size needed to make discrete time reachability reasonable
numSteps = 20
dt = 0.1


####Define Bound TORA########
function bound_tora_old(TORA; plotFlag=false, npoint=2)
    lbs, ubs = extrema(TORA.domain)

    ##Bound initial state variable (dx1 = x2)
    lb_x2 = lbs[2]
    ub_x2 = ubs[2]
    x1Func = :(1*x2)
    x1FuncLB_u, x1FuncUB_u = interpol_nd(bound_univariate(x1Func, lb_x2, ub_x2, plotflag = plotFlag)...)
    #NOTE: dx1 needs to be a function of x1 as well, use lifting to achieve
    emptyList = [1]
    currList = [2]
    #Bounds of x1
    lb_x1 = lbs[1]
    ub_x1 = ubs[1]
    x1FuncLB, x1FuncUB = lift_OA(emptyList, currList, x1FuncLB_u, x1FuncLB_u, lb_x1, ub_x1)

    #Next, bound dx2 (dx2 = -x1 + 0.1*sin(x3))
    #Bound first component of dx2 (-x1)
    x2FuncSub1 = :(-1*x1)
    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(x2FuncSub1, lb_x1, ub_x1, plotflag = plotFlag)...)

    #Bound second component of dx2 (0.1*sin(x3))
    lb_x3 = lbs[3]
    ub_x3 = ubs[3]
    x2FuncSub2 = :(sin(x3))
    #TEST: Debugging 
    println("lb_x3: $lb_x3, ub_x3: $ub_x3")
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(x2FuncSub2, lb_x3, ub_x3, plotflag=plotFlag)...)

    x2FuncSub2LB = [(tup[1:end-1]..., 0.1*tup[end]) for tup in x2FuncSub2LB]
    x2FuncSub2UB = [(tup[1:end-1]..., 0.1*tup[end]) for tup in x2FuncSub2UB]

    #NOTE: I checked, bounds appear valid :)
    #Add a dimension to prepare for Minkowski sum 
    x2FuncSub1LB_l = addDim(x2FuncSub1LB, 2)
    x2FuncSub1UB_l = addDim(x2FuncSub1UB, 2)

    x2FuncSub2LB_l = addDim(x2FuncSub2LB, 1)
    x2FuncSub2UB_l = addDim(x2FuncSub2UB, 1)

    #Take the Minkowski sum of the bounds 
    x2FuncLB_u = unique(MinkSum(x2FuncSub1LB_l, x2FuncSub2LB_l))
    x2FuncUB_u = unique(MinkSum(x2FuncSub1UB_l, x2FuncSub2UB_l))

    #Finally, dx3 must be a function of x3
    emptyList = [2]
    currList = [1,3]
    x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs, ubs)

    #Next, bound dx3 (dx3 = x4)
    lb_x4 = lbs[4]
    ub_x4 = ubs[4]
    x3Func = :(1*x4)
    x3FuncLB_u, x3FuncUB_u = interpol_nd(bound_univariate(x3Func, lb_x4, ub_x4, plotflag = plotFlag)...)
    #NOTE: dx3 needs to be a function of x3 as well, use lifting to achieve
    emptyList = [1]
    currList = [2]
    x3FuncLB, x3FuncUB = lift_OA(emptyList, currList, x3FuncLB_u, x3FuncUB_u, lb_x3, ub_x3)

    #Finally, bound dx4 (dx4 = u)
    #Here, since dx4 is directly a function of u, just use a constant 
    κ = 1e-8
    # x4Func = :($κ*x4)
    x4Func = :(0*x4)
    x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(x4Func, lb_x4, ub_x4, plotflag = plotFlag)...)
    bounds = [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
    return bounds 
end

function bound_tora(TORA; plotFlag=false,npoint=2)
    lbs, ubs = extrema(TORA.domain)

    ##Bound initial state variable (dx1 = x2)
    lb_x2 = lbs[2]
    ub_x2 = ubs[2]
    x1Func = :(1*x2)
    x1FuncLB_u, x1FuncUB_u = interpol_nd(bound_univariate(x1Func, lb_x2, ub_x2, plotflag = plotFlag)...)
    #NOTE: dx1 needs to be a function of x1 as well, use lifting to achieve
    emptyList = [1]
    currList = [2]
    #Bounds of x1
    lb_x1 = lbs[1]
    ub_x1 = ubs[1]
    x1FuncLB, x1FuncUB = lift_OA(emptyList, currList, x1FuncLB_u, x1FuncLB_u, lb_x1, ub_x1)

    #Next, bound dx2 (dx2 = -x1 + 0.1*sin(x3))
    #Bound first component of dx2 (-x1)
    x2FuncSub1 = :(-1*x1)
    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(x2FuncSub1, lb_x1, ub_x1, plotflag = plotFlag)...)

    #Bound second component of dx2 (0.1*sin(x3))
    lb_x3 = lbs[3]
    ub_x3 = ubs[3]
    x2FuncSub2 = :(sin(x3))
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(x2FuncSub2, lb_x3, ub_x3, plotflag=plotFlag)...)

    #Add the 0.1 scaling to the second component of dx2
    x2FuncSub2LB = [(tup[1:end-1]..., 0.1*tup[end]) for tup in x2FuncSub2LB]
    x2FuncSub2UB = [(tup[1:end-1]..., 0.1*tup[end]) for tup in x2FuncSub2UB]

    #Lift the bounds to the same space 
    #First lift the first component of dx2
    emptyList = [2]
    currList = [1]
    l_x2FuncSub1LB, l_x2FuncSub1UB = lift_OA(emptyList, currList, x2FuncSub1LB, x2FuncSub1UB, [lbs[1], lbs[3]], [ubs[1], ubs[3]])

    #Next lift the second component of dx2
    emptyList = [1]
    currList = [2]
    l_x2FuncSub2LB, l_x2FuncSub2UB = lift_OA(emptyList, currList, x2FuncSub2LB, x2FuncSub2UB, [lbs[1], lbs[3]], [ubs[1], ubs[3]])

    #Combine the two components to get dx2
    x2FuncLB_u, x2FuncUB_u = sumBounds(l_x2FuncSub1LB, l_x2FuncSub1UB, l_x2FuncSub2LB, l_x2FuncSub2UB, false)

    #Finally, dx2 must be a function of x2
    emptyList = [2]
    currList = [1,3]
    x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs, ubs)

    #Next, bound dx3 (dx3 = x4)
    lb_x4 = lbs[4]
    ub_x4 = ubs[4]
    x3Func = :(1*x4)
    x3FuncLB_u, x3FuncUB_u = interpol_nd(bound_univariate(x3Func, lb_x4, ub_x4, plotflag = plotFlag)...)
    #NOTE: dx3 needs to be a function of x3 as well, use lifting to achieve
    emptyList = [1]
    currList = [2]
    x3FuncLB, x3FuncUB = lift_OA(emptyList, currList, x3FuncLB_u, x3FuncUB_u, lb_x3, ub_x3)

    #Finally, bound dx4 (dx4 = u)
    #Here, since dx4 is directly a function of u, just use a constant 
    κ = 1e-8
    # x4Func = :($κ*x4)
    x4Func = :(0*x4)
    x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(x4Func, lb_x4, ub_x4, plotflag = plotFlag)...)
    bounds = [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
    return bounds 
end


####Next Define function to link control and relevant dynamics###
function tora_dyn_con_link!(query, neurons, graph, dynModel, netModel)
    #Define variables that are inputs to the network model 
    @variable(netModel, x1)
    @variable(netModel, x2)
    @variable(netModel, x3)
    @variable(netModel, x4)

    #Specify inputs to the network 
    @constraint(netModel, neurons[1][1] == x1)
    @constraint(netModel, neurons[1][2] == x2)
    @constraint(netModel, neurons[1][3] == x3)
    @constraint(netModel, neurons[1][4] == x4)

    #Link network inputs to appropriate dynamics models 
    @linkconstraint(graph, netModel[:x1] == dynModel[2][:x][1])
    @linkconstraint(graph, netModel[:x2] == dynModel[2][:x][2])
    @linkconstraint(graph, netModel[:x3] == dynModel[2][:x][3])
    @linkconstraint(graph, netModel[:x4] == dynModel[4][:x][1])

    #Link network output to x4
    @variable(netModel, u)
    #Normalizing output of the network as well
    #NOTE: This was the source of our error. Chelsea already normalized the output in the NNET file.
    @constraint(netModel, u == neurons[end][1])
    @linkconstraint(graph, netModel[:u] == dynModel[4][:u])

    #Finally, identify pertinent input variable for each model
    i = 0
    for sym in query.problem.varList 
        i += 1
        if sym == :x2
            pertVar = dynModel[i][:x][2]
        else
            pertVar = dynModel[i][:x][1]
        end
        #Add pertinent variable to var dict 
        push!(query.var_dict[sym], [pertVar])
    end
end

TORA = GraphPolyProblem(
    exprList, #dynamics
    nothing, #no decomposed dynamics 
    control_coef, #control coefficients
    domain, #input domain 
    [:x1,:x2,:x3,:x4], #Variables that have bounds 
    nothing, #undefined bounds to start 
    tora_dynamics, 
    bound_tora,
    tora_control,
    tora_dyn_con_link!
)

query = GraphPolyQuery(
    TORA, #problem 
    controller, #path to network file 
    Id(), #last layer activation 
    "MIP", #query solver, can be MIP or Marabou 
    numSteps, #number of discrete intervals 
    dt, #time step size 
    2, #N_overt
    nothing, #var_dict
    nothing, #mod_dict
    2 #case. Delete this param
)



#########Debug Bound TORA#############
#####################
#Test single step concrete reachability
query1 = deepcopy(query)
query1.ntime = 1
@time reachset, boundset = concreach!(query1);

query11 = deepcopy(query)
query11.ntime = 1
query11.problem.bound_func = bound_tora_old
@time reachset1, boundset1 = concreach!(query11);

hypContained(reachset1, reachset)

boundLen1 = []
for bound in boundset 
    boundVec = []
    push!(boundVec, length(bound[1]))
    boundTup = tuple(boundVec...)
    push!(boundLen1, boundTup)
end

boundLen2 = []
for bound in boundset1 
    boundVec = []
    push!(boundVec, length(bound[1]))
    boundTup = tuple(boundVec...)
    push!(boundLen2, boundTup)
end

boundLen1
boundLen2
#Test multi-step concrete reachability
query2 = deepcopy(query)
query2.ntime = 20
@time reachsets, boundsets = multi_step_concreach(query2);

query22 = deepcopy(query)
query22.ntime = 20
query22.problem.bound_func = bound_tora_old
@time reachsets1, boundsets1 = multi_step_concreach(query22);

boundsets[1]
boundsets1[1]

boundLen1 = []
for bound in boundsets 
    boundVec = []
    for i = 1:length(bound)
        push!(boundVec, length(bound[i][1]))
    end
    boundTup = tuple(boundVec...)
    push!(boundLen1, boundTup)
end

boundLen2 = []
for bound in boundsets1 
    boundVec = []
    for i = 1:length(bound)
        push!(boundVec, length(bound[i][1]))
    end
    boundTup = tuple(boundVec...)
    push!(boundLen2, boundTup)
end

boundLen1
boundLen2

extrema(reachsets[end])
extrema(reachsets1[end])
overtSet = Hyperrectangle(low=[
    -0.34415609789501067
 -1.070147842066413
 -0.020800937713150447
  0.13458397572173186
],
high=[
    -0.0723092674737037
 -0.7876645636084808
  0.2220092844249411
  0.4163216175335922
])


t = 20
#Comparing to OVERT
plot(project(reachsets[t+1], [1, 2]), lab="Reach Set", color="lightpink", lw=0.5)
plot!(project(overtSet, [1, 2]), lab="OVERT Set", color="lightblue", lw=0.5)

plot(project(reachsets[t+1], [3, 4]), lab="Reach Set", color="lightpink", lw=0.5)
plot!(project(overtSet, [3, 4]), lab="OVERT Set", color="lightblue", lw=0.5)
#checking the property 
safeSet = Hyperrectangle(low=[-2, -2, -2, -2], high=[2, 2, 2, 2])


overtSet
reachsets[t+1]


for (i, reachset) in enumerate(reachsets)
    if !(LazySets.issubset(reachset, safeSet))
        println("Property is violated at time $i")
    end
end

volume(reachsets[end])

p = plot(project(safeSet, [1, 2]), lab="Safe Set", color="lightblue", lw=0.5)
for reachset in reachsets
    plot!(project(reachset, [1, 2]), color="lightpink", lw=0.5)
end
display(p)

q = plot(project(safeSet, [3, 4]), lab="Safe Set", color="lightblue", lw=0.5)
for reachset in reachsets
    plot!(project(reachset, [3, 4]), color="lightpink", lw=0.5)
end
display(q)