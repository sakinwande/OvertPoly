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

controller = "Networks/ARCH-COMP-2023/nnet/controllerUnicycle.nnet"
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
#Should be 50?
numSteps = 10
w = 1e-4
domain = Hyperrectangle(low=[9.5,-4.5,2.1,1.5], high = [9.55,-4.45,2.11,1.51])
depMat = [[1,0,1,1],[0,1,1,1], [0,0,1,0], [0,0,0,1]]
########TEST: Debugging Bound Unicycle#########
# lbs, ubs = extrema(domain)
# plotFlag = true
#######################################

###Define Bound Unicycle########
function bound_unicycle(Unicycle; plotFlag=false)
    lbs, ubs = extrema(Unicycle.domain)

    ##Bound initial state variable (dx1 = x4*cos(x3))#####
    #K-A Decomposition exp(ln(x4) + ln(cos(x3)))
    #Bound ln(x4)
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

    #Find how much to shift log x4 by 
    sx4 = inpShiftLog(lb_x4, ub_x4, bounds=x1FuncSub_1LB)
    sx3 = inpShiftLog(lb_x3, ub_x3, bounds=x1FuncSub_2LB)

    #Apply log 
    x1FuncSub_1LB_l = [(tup[1:end-1]..., log(tup[end] + sx4)) for tup in x1FuncSub_1LB]
    x1FuncSub_1UB_l = [(tup[1:end-1]..., log(tup[end] + sx4)) for tup in x1FuncSub_1UB]

    x1FuncSub_2LB_l = [(tup[1:end-1]..., log(tup[end] + sx3)) for tup in x1FuncSub_2LB]
    x1FuncSub_2UB_l = [(tup[1:end-1]..., log(tup[end] + sx3)) for tup in x1FuncSub_2UB]

    #Add a dimension to prepare for Minkowski sum. Put x3 before x4 :)
    x1FuncSub_1LB_ll = addDim(x1FuncSub_1LB_l, 1)
    x1FuncSub_1UB_ll = addDim(x1FuncSub_1UB_l, 1)
    
    x1FuncSub_2LB_ll = addDim(x1FuncSub_2LB_l, 2)
    x1FuncSub_2UB_ll = addDim(x1FuncSub_2UB_l, 2)

    #Combine to get log(x4*cos(x3))
    x1FuncLB_l = MinkSum(x1FuncSub_1LB_ll, x1FuncSub_2LB_ll)
    x1FuncUB_l = MinkSum(x1FuncSub_1UB_ll, x1FuncSub_2UB_ll)
    
    #Combine to get x4*cos(x3)
    x1FuncLB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1FuncLB_l]
    x1FuncUB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1FuncUB_l]
    
    #Account for the shift
    x1FuncLB = Any[]
    x1FuncUB = Any[]
    
    for tup in x1FuncLB_s
        #First find the corresponding f(x) and f(y) values
        #NOTE: Round to avoid floating point errors
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x1FuncSub_2LB)[1]
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x1FuncSub_1LB)[1]

        xInd = findall(x->x[1] == tup[1], x1FuncSub_2LB)[1]
        yInd = findall(y->y[1] == tup[2], x1FuncSub_1LB)[1]

        
        

        #Quadratic shift down
        #NOTE: You care about function value, not index value ;)
        #NOTE: Interval subtraction 
        newXY = tup[end] - sx3 * x1FuncSub_1UB[yInd][end] - sx4 * x1FuncSub_2UB[xInd][end] - sx3*sx4
        
        push!(x1FuncLB, (tup[1:end-1]..., newXY))
    end


    for tup in x1FuncUB_s
        #First find the corresponding f(x) and f(y) values
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x1FuncSub_2LB)[1]
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x1FuncSub_1LB)[1]

        xInd = findall(x->x[1] == tup[1], x1FuncSub_2LB)[1]
        yInd = findall(y->y[1] == tup[2], x1FuncSub_1LB)[1]
        
        #Quadratic shift down
        #NOTE: Interval subtraction
        newXY = tup[end] - sx3 * x1FuncSub_1LB[yInd][end] - sx4 * x1FuncSub_2LB[xInd][end] - sx3*sx4
        
        push!(x1FuncUB, (tup[1:end-1]..., newXY))
    end

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

    #Find how much to shift log x4 by
    sx4 = inpShiftLog(lb_x4, ub_x4, bounds=x2FuncSub1LB)
    sx3 = inpShiftLog(lb_x3, ub_x3, bounds=x2FuncSub2LB)

    #Apply log
    x2FuncSub1LB_l = [(tup[1:end-1]..., log(tup[end] + sx4)) for tup in x2FuncSub1LB]
    x2FuncSub1UB_l = [(tup[1:end-1]..., log(tup[end] + sx4)) for tup in x2FuncSub1UB]

    x2FuncSub2LB_l = [(tup[1:end-1]..., log(tup[end] + sx3)) for tup in x2FuncSub2LB]
    x2FuncSub2UB_l = [(tup[1:end-1]..., log(tup[end] + sx3)) for tup in x2FuncSub2UB]

    #Add a dimension to prepare for Minkowski sum. Put x3 before x4 :)
    x2FuncSub1LB_ll = addDim(x2FuncSub1LB_l, 1)
    x2FuncSub1UB_ll = addDim(x2FuncSub1UB_l, 1)

    x2FuncSub2LB_ll = addDim(x2FuncSub2LB_l, 2)
    x2FuncSub2UB_ll = addDim(x2FuncSub2UB_l, 2)

    #Combine to get log(x4*sin(x3))
    x2FuncLB_u = unique(MinkSum(x2FuncSub1LB_ll, x2FuncSub2LB_ll))
    x2FuncUB_u = unique(MinkSum(x2FuncSub1UB_ll, x2FuncSub2UB_ll))

    #Combine to get x4*sin(x3)
    x2FuncLB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x2FuncLB_u]
    x2FuncUB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x2FuncUB_u]

    #Account for the shift
    x2FuncLB = Any[]
    x2FuncUB = Any[]

    for tup in x2FuncLB_s
        #First find the corresponding f(x) and f(y) values
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x2FuncSub2LB)
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x2FuncSub1LB)
        xInd = findall(x->x[1] == tup[1], x2FuncSub2LB)
        yInd = findall(y->y[1] == tup[2], x2FuncSub1LB)

        #Quadratic shift down
        newXY = tup[end] - sx3 * x2FuncSub1UB[yInd][1][end] - sx4 * x2FuncSub2UB[xInd][1][end] - sx3*sx4

        push!(x2FuncLB, (tup[1:end-1]..., newXY))
    end

    for tup in x2FuncUB_s
        #First find the corresponding f(x) and f(y) values
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x2FuncSub2LB)
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x2FuncSub1LB)

        xInd = findall(x->x[1] == tup[1], x2FuncSub2LB)
        yInd = findall(y->y[1] == tup[2], x2FuncSub1LB)

        #Quadratic shift down
        newXY = tup[end] - sx3 * x2FuncSub1LB[yInd][1][end] - sx4 * x2FuncSub2LB[xInd][1][end] - sx3*sx4

        push!(x2FuncUB, (tup[1:end-1]..., newXY))
    end

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
    x1FuncLB_u = deepcopy(x1FuncLB)
    x1FuncUB_u = deepcopy(x1FuncUB)

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
        push!(query.var_dict[sym_t], [pertVar])
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


#Next, test multi-step concrete reachability
query1 = deepcopy(query)
query1.ntime = 1
@time reachSet, boundSet = concreach!(query1);

#Next, test multi-step concrete reachability
query2 = deepcopy(query)
query2.ntime = 2
@time reachSets, boundSets = multi_step_concreach(query2);

#Next, test direct symreach 
query3 = deepcopy(query)
query3.problem.bounds = boundSets
query3.ntime = 2
@time symReach = symreach(query3, depMat, 2)

query.problem.varList
boundSets[2]

#Test hybrid reachability

t_sym = 2
concInt = [2,2]
query4 = deepcopy(query)
query4.problem.domain = reach_set
query4.ntime = t_sym
# @time reachSets = multi_step_hybreach(query4, depMat, concInt)
@time reach_set = hybreach(query4, depMat, t_sym)

query.problem.domain
#############################

goalSet = Hyperrectangle(low = [-0.6, -0.2, -0.06, -0.3], high=[0.6, 0.2, 0.06, 0.3])

plot(project(reachSets[end], [1,2]), lab="Reachable Set", color="lightblue", lw=0.5)
plot!(project(goalSet, [1,2]), lab="Goal Set", color="red", lw=0.5)


plot(project(reachSets[end], [3,4]))
plot!(project(goalSet, [3,4]))

symQuery = deepcopy(query)
symQuery.problem.bounds = boundSets
reachSets[end]

# # reachSets[1]
# query.problem.bound_func(query.problem; plotFlag=true)

##################################################
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

sym_hyp
reachSets[end]


plot(reachSets[end][3,4])
plot!(sym_hyp)

plot(project(reachSets[end], [3,4]))
plot!(project(sym_hyp, [3,4]))