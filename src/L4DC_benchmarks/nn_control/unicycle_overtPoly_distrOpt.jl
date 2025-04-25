include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo
control_coef = [[0],[0],[1],[1]]

println("Running Unicycle Benchmark")
controller = "../../../Networks/ARCH-COMP-2023/nnet/controllerUnicycle.nnet"
exprList = [:(cos(x3)*x4), :(sin(x3)*x4), :(0*x3), :(0*x4)]

##Define Unicycle Dynamics#####
function unicycle_dynamics(x, u)
    """
    Dynamics of the unicycle benchmark. NGL I don't know why I use this
    """
    dx1 = x[4]*cos(x[3])
    dx2 = x[4]*sin(x[3])
    dx3 = u[2]
    dx4 = u[1]

    xNew = x + [dx1, dx2, dx3, dx4].*dt

    return xNew
end


function unicycle_control(input_set)
    con_inp_set = input_set 
    return con_inp_set
end

dt = 0.2
#This should really be 50 steps to recover 10 seconds in discrete time 
numSteps = 50
w = 1e-4
domain = Hyperrectangle(low=[9.5,-4.5,2.1,1.5], high = [9.55,-4.45,2.11,1.51])
depMat = [[1,0,1,1],[0,1,1,1],[0,0,1,0],[0,0,0,1]]
########TEST: Debugging Bound Unicycle#########
# lbs, ubs = extrema(domain)
# plotFlag = true
######################################

###Define Bound Unicycle########
function bound_unicycle(Unicycle; plotFlag=false)
    lbs, ubs = extrema(Unicycle.domain)

    ##Bound initial state variable (dx1 = x4*cos(x3))#####
    #Weird behavior with Hyperrectangle
    lb_x4 = lbs[4]
    ub_x4 = ubs[4]

    #First bound x4
    x1FuncSub_1 = :(1*x4)
    x1FuncSub_1LB, x1FuncSub_1UB = interpol_nd(bound_univariate(x1FuncSub_1, lb_x4, ub_x4)...)
    
    #Also bound cos(x3)
    lb_x3 = lbs[3]
    ub_x3 = ubs[3]
    x1FuncSub_2 = :(cos(x3))
    x1FuncSub_2LB, x1FuncSub_2UB = interpol_nd(bound_univariate(x1FuncSub_2, lb_x3, ub_x3)...)

    #Lift the bounds to the same space
    #First lift the first component of dx1
    emptyList = [1]
    currList = [2]
    l_x1FuncSub_1LB, l_x1FuncSub_1UB = lift_OA(emptyList, currList, x1FuncSub_1LB, x1FuncSub_1UB, lbs[3:4], ubs[3:4])

    #Next lift the second component of dx1
    emptyList = [2]
    currList = [1]
    l_x1FuncSub_2LB, l_x1FuncSub_2UB = lift_OA(emptyList, currList, x1FuncSub_2LB, x1FuncSub_2UB, lbs[3:4], ubs[3:4])

    #Combine to get x4*cos(x3)
    x1FuncLB, x1FuncUB = prodBounds(l_x1FuncSub_1LB, l_x1FuncSub_1UB, l_x1FuncSub_2LB, l_x1FuncSub_2UB)

    #Check if bounds are valid by plotting the surface
    if plotFlag
        xS = unique!(Any[tup[1] for tup in x1FuncLB])
        yS = unique!(Any[tup[2] for tup in x1FuncLB])

        surfDim = (size(yS)[1], size(xS)[1])
        baseFunc = exprList[1]

        #Plot the surface
        plotSurf(baseFunc, x1FuncLB, x1FuncUB, surfDim, xS, yS, true)
    end
    #############Next, bound dx2 (dx2 = x4*sin(x3))#####
    #Bound first component of dx2 (x4)
    x2FuncSub1 = :(1*x4)
    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(x2FuncSub1, lb_x4, ub_x4)...)

    #Bound second component of dx2 (sin(x3))
    x2FuncSub2 = :(sin(x3))
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(x2FuncSub2, lb_x3, ub_x3)...)

    #Lift the bounds to the same space
    #First lift the first component of dx2\
    emptyList = [1]
    currList = [2]
    l_x2FuncSub1LB, l_x2FuncSub1UB = lift_OA(emptyList, currList, x2FuncSub1LB, x2FuncSub1UB, lbs[3:4], ubs[3:4])

    #Next lift the second component of dx2
    emptyList = [2]
    currList = [1]
    l_x2FuncSub2LB, l_x2FuncSub2UB = lift_OA(emptyList, currList, x2FuncSub2LB, x2FuncSub2UB, lbs[3:4], ubs[3:4])

    #Combine to get x4*sin(x3)
    x2FuncLB, x2FuncUB = prodBounds(l_x2FuncSub1LB, l_x2FuncSub1UB, l_x2FuncSub2LB, l_x2FuncSub2UB)
    #Check if bounds are valid by plotting the surface
    if plotFlag
        xS = unique!(Any[tup[1] for tup in x2FuncLB])
        yS = unique!(Any[tup[2] for tup in x2FuncLB])

        surfDim = (size(yS)[1], size(xS)[1])
        baseFunc = exprList[2]

        #Plot the surface
        plotSurf(baseFunc, x2FuncLB, x2FuncUB, surfDim, xS, yS, true)

    end

   #dx1 and dx2 must be functions of x1 and x2 respectively
   emptyList = [1]
   currList = [2,3]
   
   #Retcon x1FuncLB and x1FuncUB to be unlifted 
   x1FuncUB_u = deepcopy(x1FuncUB)
   x1FuncLB_u = deepcopy(x1FuncLB)
   
   lbs_x1 = [lbs[1]]
   append!(lbs_x1, lbs[3:4])
   ubs_x1 = [ubs[1]]
   append!(ubs_x1, ubs[3:4])
   x1FuncLB, x1FuncUB = lift_OA(emptyList, currList, x1FuncLB_u, x1FuncUB_u, lbs_x1, ubs_x1)
   
   emptyList = [1]
   currList = [2,3]
   
   #Retcon x2FuncLB and x2FuncUB to be unlifted
   x2FuncLB_u = deepcopy(x2FuncLB)
   x2FuncUB_u = deepcopy(x2FuncUB)
   
   lbs_x2 = lbs[2:4]
   ubs_x2 = ubs[2:4]
   x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs_x2, ubs_x2)
   
   #############Next, bound dx3 (dx3 = u[2])#####
   #Since dx3 is solely a function of u[2], just use a constant
   x3Func = :(0*x3)
   x3FuncLB, x3FuncUB = interpol_nd(bound_univariate(x3Func, lb_x3, ub_x3)...)
   
   #############Finally, bound dx4 (dx4 = u[1] + w)#####
   #Here, dx4 is a function of u[1] and a disturbance term. Treat disturbance as a zero mean constant 
   x4Func = :(0*x4)
   x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(x4Func, lb_x4, ub_x4, ϵ = w)...)
   
   bounds = [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
   
   return bounds
end

###Next Define function to link control and relevant dynamics###
function unicycle_dyn_con_link!(query, neurons, graph, dynModel, netModel, t_ind=nothing)

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
    @linkconstraint(graph, netModel[:x1] == dynModel[1][:x][1])
    @linkconstraint(graph, netModel[:x2] == dynModel[2][:x][1])
    @linkconstraint(graph, netModel[:x3] == dynModel[3][:x][1])
    @linkconstraint(graph, netModel[:x4] == dynModel[4][:x][1])

    #Link network output to x3 and x4
    @variable(netModel, u1)
    @variable(netModel, u2)

    #Normalizing output of the network as well
    #NOTE: This was the source of our error. Chelsea already normalized the output in the NNET file.
    @constraint(netModel, u1 == neurons[end][1])
    @constraint(netModel, u2 == neurons[end][2])

    @linkconstraint(graph, netModel[:u1] == dynModel[4][:u])
    @linkconstraint(graph, netModel[:u2] == dynModel[3][:u])

    #Finally, identify pertinent input variable for each model
    i = 0
    for sym in query.problem.varList
        if !isnothing(t_ind)
            sym_t = Meta.parse("$(sym)_$(t_ind)")
        else
            sym_t = sym
        end
        i += 1
        pertVar = dynModel[i][:x][1]
        push!(query.var_dict[sym_t],[pertVar])
    end

end

Unicycle = GraphPolyProblem(
    exprList, #dynamics
    nothing, #no decomposed dynamics
    control_coef, #control coefficients
    domain, #input domain
    [:x1,:x2,:x3,:x4], #Variables that have bounds
    nothing, #undefined bounds to start
    unicycle_dynamics,
    bound_unicycle,
    unicycle_control,
    unicycle_dyn_con_link!
)

query = GraphPolyQuery(
    Unicycle, #problem
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

currSplit = 5;
println("Trying to verify with [5,5,5,5] split")
#Trying to run Unicycle Benchmark
reachList = []
symReachList = []
boundsList = []
#Untimed run
cquery = deepcopy(query)
squery = deepcopy(query)
cquery.ntime = 5
squery.ntime = 5
t_sym = 5
concReachSets, BoundSets = multi_step_concreach(cquery);
squery.problem.bounds = BoundSets
sym_set = symreach(squery, concReachSets, depMat, t_sym);
cquery.problem.domain = sym_set
print(sym_set)
#Timed run
cquery = deepcopy(query)
squery = deepcopy(query)
cquery.ntime = 5
squery.ntime = 5
t_sym = 5
tStart = Dates.now()
concReachSets, BoundSets = multi_step_concreach(cquery);
squery.problem.bounds = BoundSets;
push!(boundsList,BoundSets...);
push!(reachList,concReachSets...);
sym_set = symreach(squery, concReachSets, depMat, t_sym);
push!(symReachList, sym_set)
t1 = Dates.now()
cquery.problem.domain = sym_set;
print("Time to compute 5 hybrid reach sets: ", t1-tStart)
concReachSets, BoundSets = multi_step_concreach(cquery);
squery.problem.bounds = BoundSets;
push!(boundsList,BoundSets...);
push!(reachList,concReachSets...);
sym_set = symreach(squery, concReachSets, depMat, t_sym);
push!(symReachList, sym_set)
t2 = Dates.now()
cquery.problem.domain = sym_set;
print("Time to compute 10 hybrid reach sets: ", t2-tStart)
concReachSets, BoundSets = multi_step_concreach(cquery);
squery.problem.bounds = BoundSets;
push!(boundsList,BoundSets...);
push!(reachList,concReachSets...);
sym_set = symreach(squery, concReachSets, depMat, t_sym);
push!(symReachList, sym_set)
t3 = Dates.now()
cquery.problem.domain = sym_set;
print("Time to compute 15 hybrid reach sets: ", t3-tStart)
concReachSets, BoundSets = multi_step_concreach(cquery);
squery.problem.bounds = BoundSets;
push!(boundsList,BoundSets...);
push!(reachList,concReachSets...);
sym_set = symreach(squery, concReachSets, depMat, t_sym);
push!(symReachList, sym_set)
t4 = Dates.now()
cquery.problem.domain = sym_set;
print("Time to compute 20 hybrid reach sets: ", t4-tStart)




#####Scaling/Ablation Run
#Timed run
# cquery = deepcopy(query)
# squery = deepcopy(query)
# cquery.ntime = 10
# tstart = Dates.now()
# concReachSets, BoundSets = multi_step_concreach(cquery);
# push!(boundsList,BoundSets...)
# push!(reachList,concReachSets...)
# squery.problem.bounds = boundsList
# t_sym = 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# tend1 = Dates.now()
# println("Time to compute 10 hybrid reach sets: ", tend1-tstart)
# cquery.problem.domain = sym_set
# concReachSets, BoundSets = multi_step_concreach(cquery);
# push!(boundsList,BoundSets...)
# push!(reachList,concReachSets[2:end]...)
# squery.problem.bounds = boundsList
# #Use reachlist!!!
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# tend2 = Dates.now()
# println("Time to compute 20 hybrid reach sets: ", tend2-tstart)
# cquery.problem.domain = sym_set
# concReachSets, BoundSets = multi_step_concreach(cquery);
# push!(boundsList,BoundSets...)
# push!(reachList,concReachSets[2:end]...)
# squery.problem.bounds = boundsList
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# tend3 = Dates.now()
# println("Time to compute 30 hybrid reach sets: ", tend3-tstart)
# cquery.problem.domain = sym_set
# concReachSets, BoundSets = multi_step_concreach(cquery);
# push!(boundsList,BoundSets...)
# push!(reachList,concReachSets[2:end]...)
# squery.problem.bounds = boundsList
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# tend4 = Dates.now()
# println("Time to compute 40 hybrid reach sets: ", tend4-tstart)
# cquery.problem.domain = sym_set
# concReachSets, BoundSets = multi_step_concreach(cquery);
# push!(boundsList,BoundSets...)
# push!(reachList,concReachSets[2:end]...)
# squery.problem.bounds = boundsList
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# t_sym += 1
# println(t_sym)
# squery.ntime = t_sym
# @time sym_set = symreach(squery, reachList, depMat, t_sym);
# push!(symReachList, sym_set)
# println(volume(sym_set))
# println(extrema(sym_set))
# tend = Dates.now()
# # push!(reachList,concReachSets...)
# # push!(boundsList,BoundSets...)
# println("##################################################################")
# println("Time taken to compute 50 hybrid reach: ", tend-tstart)
# println("##################################################################")
