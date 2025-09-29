include("overtPoly_helpers.jl")
include("nn_mip_encoding.jl")
include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")
include("problems.jl")
using LazySets
using Random
using Plasmo

function encode_dynamics!(query::GraphPolyQuery)
    #create an optigraph to store the model
    graph = OptiGraph()
    #create a vector of models for the dynamics 
    dynNodes = @optinode(graph, nodes[1:length(query.problem.varList)])
    ind = 0
    #Iterate through elements of varList and add appropriate variables to the appropriate model 
    for sym in query.problem.varList
        #print(sym)
        ind += 1
        LB, UB = query.problem.bounds[ind]
        Tri = OA2PWA(LB)
        xS = [(tup[1:end-1]) for tup in LB]
        yUB = [tup[end] for tup in UB]
        yLB = [tup[end] for tup in LB]

        dynNodes[ind] = ccEncoding!(xS, yLB, yUB, Tri, query, query.problem.varList[ind], ind, dynNodes[ind])
    end

    #Reuse mod dict to store graph, dynNodes, and neural network 
    query.mod_dict[:graph] = graph
    query.mod_dict[:f] = dynNodes
end
###Create new Encode Control Function###
function encode_control!(query::GraphPolyQuery)
    input_set = query.problem.control_func(query.problem.domain)
    network_file = query.network_file
    #NOTE: Changed NetModel to be an graph instead of a node
    netNode = @optinode(query.mod_dict[:graph])
    neurons = add_controller_constraints!(netNode, network_file, input_set, Id())
    query.mod_dict[:u] = netNode
    return neurons
end

function JuMP.objective_bound(graph::OptiGraph)
    return MOI.get(graph, MOI.ObjectiveBound())
end

function conc_reach_solve(query;threads=0, digits=15)
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)

    ###Minimization Step########
    min_dynModel = query.mod_dict[:f]
    minGraph = query.mod_dict[:graph]
    min_netModel = query.mod_dict[:u]
    i = 0
    #Compute lower bounds
    # set_optimizer(minGraph, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "Threads" => threads))
    set_optimizer(minGraph, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))
    #NOTE: Optimize each variable separately
    for sym in query.problem.varList 
        v = query.var_dict[sym][end][1]
        dv = query.var_dict[sym][2][1]
        next_v_l = v + query.dt*dv
        #NOTE: Set graph level objective directly
        @objective(minGraph, Min, next_v_l)
        optimize!(minGraph)
        @assert termination_status(minGraph) == MOI.OPTIMAL
        #push!(lows, objective_value(minGraph))
        push!(lows, JuMP.objective_bound(minGraph))
    end
    
    #Compute upper bounds
    max_dynModel = query.mod_dict[:f]
    maxGraph = query.mod_dict[:graph]
    max_netModel = query.mod_dict[:u]
    # set_optimizer(maxGraph, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "Threads" => threads))
    set_optimizer(maxGraph, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))
    for sym in query.problem.varList 
        v = query.var_dict[sym][end][1]
        dv = query.var_dict[sym][2][1]
        next_v_u = v + query.dt*dv
        #NOTE: Set graph level objective directly
        @objective(maxGraph, Min, -next_v_u)
        optimize!(maxGraph)
        #push!(highs, -objective_value(maxGraph))
        @assert termination_status(maxGraph) == MOI.OPTIMAL
        push!(highs, -JuMP.objective_bound(maxGraph))
    end

    lows = floor.(lows, digits=digits)
    highs = ceil.(highs, digits=digits)

    #Since the default 
    reach_set = Hyperrectangle(low=lows, high=highs)
    return reach_set 
end

function concreach!(query::GraphPolyQuery; digits=15)
    query.problem.bounds = query.problem.bound_func(query.problem, npoint=query.N_overt)
    query.var_dict = Dict{Symbol,Any}()
    query.mod_dict = Dict{Symbol,Any}()

    encode_dynamics!(query)

    #Encode the network and link the control to the dynamics
    if !isnothing(query.network_file)
        neurons = encode_control!(query)
    end

    dyn_con_link! = query.problem.link_func
    graph = query.mod_dict[:graph]
    dynModel = query.mod_dict[:f]
    netModel = query.mod_dict[:u]
    dyn_con_link!(query, neurons, graph, dynModel, netModel)

    reach_set = conc_reach_solve(query, digits=digits)
    return reach_set, query.problem.bounds
end

function multi_step_concreach(query::GraphPolyQuery; digits=15)
    """
    Method to solve the concrete reachability problem using MIP for multiple time steps.
    """
    input_set = query.problem.domain
    reachSets = [input_set]
    boundSets = []
    
    #t1 = Dates.now()
    
    for i = 1:query.ntime
        query.problem.domain = reachSets[end]
        reachSet, boundSet = concreach!(query; digits=digits)
        push!(reachSets, reachSet)
        push!(boundSets, boundSet)
    end
    # t2 = Dates.now()
    # println("Time for concrete reachability is ", t2-t1)
    return reachSets, boundSets
end

##################Implementing Sym Encoding###########
function encode_sym_dynamics!(symQuery::GraphPolyQuery, x_dim)
    """
    Method to encode symbolic dynamics. Takes symQuery as input
    """
    symGraph = OptiGraph()
    #####Enter time loop######
    for t_ind = 1:symQuery.ntime
        #Create a new set of nodes for each time step
        dynNodes = @optinode(symGraph, nodes[1:x_dim])
        #####Enter node loop####
        for (x_ind, sym) in enumerate(symQuery.problem.varList)
            sym_t = Meta.parse("$(sym)_$(t_ind)")
            #Get lower and upper bounds for first variable in time step
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

function encode_sym_control!(symQuery::GraphPolyQuery, reachSets)
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

function sym_link(symQuery::GraphPolyQuery, neurList, depMat)
    #First do intra-time dyn con links 
    symGraph = symQuery.mod_dict[:graph]
    for t_ind = 1:symQuery.ntime
        dynModel = symQuery.mod_dict[Meta.parse("f_$(t_ind)")]
        netModel = symQuery.mod_dict[Meta.parse("u_$(t_ind)")]

        #Link the dynamics and control first 
        symQuery.problem.link_func(symQuery, neurList[t_ind], symGraph, dynModel, netModel, t_ind)
    end

    #Next do intra tim variable dependency links 
    #TEST: do for a single time step 
    # t_ind = 1
    for t_ind = 1:symQuery.ntime
        #Iterate through models 
        #TEST: Do for a single variable
        # sym = symQuery.problem.varList[1]
        # i = 1
        for (i,sym) in enumerate(symQuery.problem.varList)
            #Identify symbol at given time step
            sym_t = Meta.parse("$(sym)_$(t_ind)")
            #Identify input vector for this symbol
            x_Vec = symQuery.var_dict[sym_t][1]
            #Identify self
            x_sym = symQuery.var_dict[sym_t][end][1]
            #Iterate over dependency matrix
            #counter for variable index
            v_ind = 1
            #TEST: Do for a single dependency
            # j = 3
            # dep_flag = depMat[i][j]
            
            for (j,dep_flag) in enumerate(depMat[i])
                #Catch self dependency or no dependency
                if j == i
                    #If we reach here, that means self dependency. Automatically increment variable index by 1
                    #println("Self dependency")
                    v_ind += 1
                elseif dep_flag == 0
                    #println("No dependency")
                    continue
                else
                    #If we reach here, that means a non-self dependency exists. Note that this creates duplicates but presolve should take care of that
                    #println("Non-self dependency")
                    
                    #Identify non-self dependent variable
                    dep_sym = symQuery.problem.varList[j]
                    #Find the dependent symbol at the given time step
                    dep_sym_t = Meta.parse("$(dep_sym)_$(t_ind)")
                    #Link function appends pertinent variable to the end!
                    x_dep_sym = symQuery.var_dict[dep_sym_t][end][1]
                    #Here we're iterating over the input vector for the symbol. Self could appear here, so we need to avoid that
                    if x_Vec[v_ind] == x_sym
                        #Avoid linking the same variable to itself. Should not be reached in practice
                        throw(ArgumentError("Self dependency detected"))
                    else
                        @linkconstraint(symGraph, x_Vec[v_ind] == x_dep_sym)
                        v_ind += 1
                    end
                end
            end
        end
    end 
    #Next link time steps
    symGraph = symQuery.mod_dict[:graph]
    for t_ind = 1:symQuery.ntime-1
        #Iterate through models and link pertinent variables 
        for sym in symQuery.problem.varList
            currSym = Meta.parse("$(sym)_$(t_ind)")
            nextSym = Meta.parse("$(sym)_$(t_ind+1)")

            xNow = symQuery.var_dict[currSym][end][1] 
            yNow = symQuery.var_dict[currSym][2][1]
            xNext = symQuery.var_dict[nextSym][end][1]

            #Define temporal self relation 
            @linkconstraint(symGraph, xNext == xNow + symQuery.dt*yNow)

        end
    end
end

function sym_reach_solve(symQuery::GraphPolyQuery, t_sym; threads=0, timeout=14400, digits=15)
    #Ensure that the time step is within bounds
    @assert t_sym <= symQuery.ntime
    #Akin to conc_reach_solve
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)
    minGraph = symQuery.mod_dict[:graph]
    i = 0

    #Compute lower bounds
    set_optimizer(minGraph, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "Threads" => threads, "TimeLimit" => timeout))
    # set_optimizer(minGraph, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))
    #NOTE: Optimize each variable separately 
    for sym in symQuery.problem.varList
        sym_t = Meta.parse("$(sym)_$(t_sym)") 
        i += 1
        v = symQuery.var_dict[sym_t][end][1]
        dv = symQuery.var_dict[sym_t][2][1]
        next_v_l = v + symQuery.dt*dv
        #NOTE: Set graph level objective directly
        @objective(minGraph, Min, next_v_l)
        @time optimize!(minGraph)
        @assert termination_status(minGraph) == MOI.OPTIMAL
        push!(lows, JuMP.objective_bound(minGraph))
    end

    #Compute upper bounds
    maxGraph = symQuery.mod_dict[:graph]
    set_optimizer(maxGraph, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "Threads" => threads, "TimeLimit" => timeout))
    # set_optimizer(maxGraph, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))
    #NOTE: Optimize each variable separately
    for sym in query.problem.varList 
        sym_t = Meta.parse("$(sym)_$(t_sym)") 
        v = symQuery.var_dict[sym_t][end][1]
        dv = symQuery.var_dict[sym_t][2][1]
        next_v_u = v + symQuery.dt*dv
        #NOTE: Set graph level objective directly
        @objective(maxGraph, Min, -next_v_u)
        @time optimize!(maxGraph)
        @assert termination_status(maxGraph) == MOI.OPTIMAL
        push!(highs, -JuMP.objective_bound(maxGraph))
    end

    #Gurobi tolerance is on the order of 1e-6, try rounding
    lows = floor.(lows, digits=digits)
    highs = ceil.(highs, digits=digits)


    reach_set = Hyperrectangle(low=lows, high=highs)
    return reach_set
end

function symreach(symQuery::GraphPolyQuery,reachSets, depMat,t_sym; threads=0, timeout=14400, digits=15)
    """
    Method to symbolically solve the reachability problem. Agnostic to how the boundSets are computed

    args:
    symQuery: GraphPolyQuery object
    depMat: Dependency matrix
    t_sym: Time step to compute the reach set
    """
    symQuery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
    symQuery.mod_dict = Dict{Symbol,Any}()

    x_dim = length(symQuery.problem.varList) #state dimension
    encode_sym_dynamics!(symQuery, x_dim)
    neurList = encode_sym_control!(symQuery, reachSets)
    sym_link(symQuery, neurList, depMat)

    sym_set = sym_reach_solve(symQuery, t_sym, threads=threads, timeout=timeout, digits=digits)
    return sym_set
end

function hybreach(symQuery::GraphPolyQuery, depMat, t_sym, reachSets = nothing, boundSets=nothing)
    """
    Hybrid symbolic method to compute reachable sets. Uses concrete reachability to compute the bounds and then uses symbolic reachability to compute the reach set
    """

    if isnothing(boundSets)
        concQuery = deepcopy(symQuery)
        reachSets,boundSets = multi_step_concreach(concQuery)
    end
    symQuery.problem.bounds = boundSets
    symQuery.ntime = t_sym
    sym_set = symreach(symQuery, reachSets, depMat, t_sym)
    return sym_set
end

function multi_step_hybreach(hybQuery, depMat, concInt)
    """
    Hybrid symbolic reachability. Requires concretization intervals 
    """

    hyb_reachSets = [hybQuery.problem.domain]
    conc_boundSets = []
    conc_reachSets = [hybQuery.problem.domain]

    cquery = deepcopy(hybQuery)
    squery = deepcopy(hybQuery)  
    for int in concInt
        #Bound the function over the desired interval
        cquery.ntime = int 
        reachSets,boundSets = multi_step_concreach(cquery)
        push!(conc_boundSets, boundSets...)
        push!(conc_reachSets, reachSets[2:end]...)

        squery.problem.bounds = boundSets
        squery.ntime = int
        hySet = symreach(squery, reachSets, depMat, int) 
        push!(hyb_reachSets, hySet)
        cquery.problem.domain = hySet
    end
    return hyb_reachSets, [conc_reachSets, conc_boundSets]
end



