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

function conc_reach_solve(query)
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)

    ###Minimization Step########
    min_dynModel = query.mod_dict[:f]
    minGraph = query.mod_dict[:graph]
    min_netModel = query.mod_dict[:u]
    i = 0
    #Compute lower bounds
    set_optimizer(minGraph, Gurobi.Optimizer)
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
    set_optimizer(maxGraph, Gurobi.Optimizer)
    for sym in query.problem.varList 
        v = query.var_dict[sym][end][1]
        dv = query.var_dict[sym][2][1]
        next_v_u = v + query.dt*dv
        #NOTE: Set graph level objective directly
        @objective(maxGraph, Min, -next_v_u)
        optimize!(maxGraph)
        #push!(highs, -objective_value(maxGraph))
        push!(highs, -JuMP.objective_bound(maxGraph))
    end
    reach_set = Hyperrectangle(low=lows, high=highs)
    return reach_set 
end

function concreach!(query::GraphPolyQuery)
    query.problem.bounds = query.problem.bound_func(query.problem)
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

    reach_set = conc_reach_solve(query)
    return reach_set, query.problem.bounds
end

function multi_step_concreach(query::GraphPolyQuery)
    """
    Method to solve the concrete reachability problem using MIP for multiple time steps.
    """
    input_set = query.problem.domain
    reachSets = [input_set]
    boundSets = []
    
    #t1 = Dates.now()
    
    for i = 1:query.ntime
        query.problem.domain = reachSets[end]
        reachSet, boundSet = concreach!(query)
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

function encode_sym_control!(symQuery::GraphPolyQuery)
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

function sym_reach_solve(symQuery::GraphPolyQuery, t_sym)
    #Ensure that the time step is within bounds
    @assert t_sym <= symQuery.ntime
    #Akin to conc_reach_solve
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)
    minGraph = symQuery.mod_dict[:graph]
    i = 0

    #Compute lower bounds
    set_optimizer(minGraph, Gurobi.Optimizer)
    #NOTE: Optimize each variable separately 
    for sym in symQuery.problem.varList
        sym_t = Meta.parse("$(sym)_$(t_sym)") 
        i += 1
        v = symQuery.var_dict[sym_t][end][1]
        dv = symQuery.var_dict[sym_t][2][1]
        next_v_l = v + symQuery.dt*dv
        #NOTE: Set graph level objective directly
        @objective(minGraph, Min, next_v_l)
        optimize!(minGraph)
        @assert termination_status(minGraph) == MOI.OPTIMAL
        push!(lows, JuMP.objective_bound(minGraph))
    end

    #Compute upper bounds
    maxGraph = symQuery.mod_dict[:graph]
    set_optimizer(maxGraph, Gurobi.Optimizer)
    #NOTE: Optimize each variable separately
    for sym in query.problem.varList 
        sym_t = Meta.parse("$(sym)_$(t_sym)") 
        v = symQuery.var_dict[sym_t][end][1]
        dv = symQuery.var_dict[sym_t][2][1]
        next_v_u = v + symQuery.dt*dv
        #NOTE: Set graph level objective directly
        @objective(maxGraph, Min, -next_v_u)
        optimize!(maxGraph)
        push!(highs, -JuMP.objective_bound(maxGraph))
    end

    reach_set = Hyperrectangle(low=lows, high=highs)
    return reach_set
end

function symreach(symQuery::GraphPolyQuery, depMat,t_sym)
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
    neurList = encode_sym_control!(symQuery)
    sym_link(symQuery, neurList, depMat)

    sym_set = sym_reach_solve(symQuery, t_sym)
    return sym_set
end

function hybreach(symQuery::GraphPolyQuery, depMat, t_sym, boundSets=nothing)
    """
    Hybrid symbolic method to compute reachable sets. Uses concrete reachability to compute the bounds and then uses symbolic reachability to compute the reach set
    """

    if isnothing(boundSets)
        concQuery = deepcopy(symQuery)
        boundSets = multi_step_concreach(concQuery)[2]
    end
    symQuery.problem.bounds = boundSets
    sym_set = symreach(symQuery, depMat, t_sym)
    return sym_set
end

function multi_step_hybreach(symQuery, depMat, concInt)
    """
    Hybrid symbolic reachability. Requires concretization intervals 
    """
    hyb_reachSets = [symQuery.problem.domain]
    for int in concInt
        hyQuery = deepcopy(symQuery)
        hyQuery.ntime = int
        hyQuery.problem.domain = hyb_reachSets[end]
        hySet = hybreach(hyQuery, depMat, int) 
        push!(hyb_reachSets, hySet)
    end
    return hyb_reachSets
end
###############Implementing Backwards Reachability####################
function breach_solve(bquery, t_idx::Union{Nothing,Int64}=nothing)
    """
    Solve contrete backwards reachability problem using MIP. Given a MIP encoding a pwl overapproximation of the dynamics, as well as MIP for the controlller, find the overapproximation of the backwards reachable set

    This version of reach_solve generalizes to multiple functions
    """
    stateVar = bquery.problem.varList
    trueInp = []
    trueOut = []
    stateVarTimed = Any[]
    
    #Compute true input and output variables 
    i = 0
    for sym in stateVar
        i += 1
        if !isnothing(t_idx)
            #Account for symbolic case where dynamics are timed
            sym_timed = Meta.parse("$(sym)_$t_idx")
            input_vars = bquery.var_dict[sym_timed][1]
            output_vars = bquery.var_dict[sym_timed][2]
            push!(stateVarTimed, sym_timed)
        else   
            input_vars = bquery.var_dict[sym][1]
            output_vars = bquery.var_dict[sym][2]
        end
        #Case 1: Multiple functions
        #Here, stateVar = varList. States are (x, y, z, etc). Simply loop over var list and find the appropriate symbol to match to the input and output variables
        if bquery.case == 1
            push!(trueInp, input_vars[i])
            push!(trueOut, output_vars)
        else
            #Case 2: Mix of single and multiple functions
            #Here, stateVar != varList. States are (x, dx, y, dy, etc)
            push!(trueInp, input_vars[i:i+1]...)
            i += 1 #increment by 1 to account for the fact that we are skipping over the derivative
            push!(trueOut, output_vars)
        end
    end

    integration_map = bquery.problem.update_rule(trueInp, trueOut)
    
    timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)

    #Loop over symbols with OVERT approximations to compute reach steps
    i = 0
    tLs, tUs = extrema(bquery.problem.domain)
    for sym in stateVar
        i += 1
        #Account for symbolic case with timed dynamics
        if !isnothing(t_idx)
            symTimed = Meta.parse("$(sym)_$t_idx")
            input_vars = bquery.var_dict[symTimed][1]
        else
            input_vars = bquery.var_dict[sym][1]
        end
        mipModel = bquery.mod_dict[sym]
        for v in input_vars 
            if v in trueInp
                dv = integration_map[v]
                next_v = v + bquery.dt*dv
                @constraint(mipModel, next_v <= tUs[i])
                @constraint(mipModel, next_v >= tLs[i])
                push!(timestep_nplus1_vars, next_v)
                @objective(mipModel, Min, v)
                JuMP.optimize!(mipModel)
                @assert termination_status(mipModel) == MOI.OPTIMAL
                objective_bound(mipModel)
                push!(lows, objective_bound(mipModel))
                @objective(mipModel, Max, v)
                JuMP.optimize!(mipModel)
                @assert termination_status(mipModel) == MOI.OPTIMAL
                objective_bound(mipModel)
                push!(highs, objective_bound(mipModel))
            end
        end
    end
    #NOTE: Hyperrectangle can plot in higher dimensions as well
    reacheable_set = Hyperrectangle(low=lows, high=highs)
    return reacheable_set
end
function back_concreach!(bquery, btarget)
    """
    Concreach equivalent for backwards reachability. Given a query and target set, compute the backwards reachable set. 

    Returns the backwards reachable set and the OVERT bounds of the problem

    Inputs
    bquery: OvertPQuery object
    btarget: Hyperrectangle of the target set
    """
    #Bound using input domain 
    bquery.problem.bounds = bquery.problem.bound_func(bquery.problem,false)
    bquery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
    bquery.mod_dict = Dict{Symbol,JuMP.Model}()
    encode_dynamics!(bquery)

    #Encode the controller if it exists
    if !isnothing(bquery.network_file)
        encode_control!(bquery)
    end

    #After encoding dynamics, replace domain with target 
    bquery.problem.domain = btarget

    reachSet = breach_solve(bquery)
    return reachSet, bquery.problem.bounds
end
function breach_refine_outer(bquery, btarget, stopRatio = 1.01)
    """
    Refine a single step backwards reachable set by iteratively solving the MIP and updating the domain. Stop when the ratio of the area of the old breach set to the new breach set is less than the stopRatio

    Returns the final breach set and a list of all breach sets computed

    Inputs
    bquery: OvertPQuery object
    btarget: Hyperrectangle of the target set
    stopRatio: Ratio to stop the refinement process
    """

    tStart = Dates.now()
    breachSets, bboundSets = back_concreach!(bquery, btarget)
    tStop = Dates.now()
    println("Time to compute initial breach set: $(tStop - tStart)")
    bloat = true
    oldBreachList = []
    breakFlag = 0
    tStart = Dates.now()
    while bloat
        bbquery = deepcopy(bquery)
        bbquery.problem.domain = breachSets
        oldBreach = deepcopy(breachSets)
        push!(oldBreachList, oldBreach)
        breachSets, bboundSets = back_concreach!(bbquery, btarget)
        if area(oldBreach)/area(breachSets) < stopRatio
            bloat = false
        end
        breakFlag += 1
        if breakFlag > 100
            print("Did not converge")
            break
        end
    end
    tStop = Dates.now()
    println("Time to compute final breach set: $(tStop - tStart)")
    return breachSets, oldBreachList, bboundSets
end
function breach_refine_inner(bquery, btarget, redRate)
    initBreachSet, _, _ = breach_refine_outer(bquery, btarget)

    #Create new candidate inner approximation 
    newCenter = deepcopy(initBreachSet.center)
    newRadius = deepcopy(initBreachSet.radius * redRate)
    
    #Define new inner approximation 
    newBreach = Hyperrectangle(newCenter, newRadius)
    
    #Test the inner approximation by computing the FRS starting from the inner approximation. If it's an inner approximation, the FRS will be contained in the target set 
    inner_query = deepcopy(bquery)
    inner_query.ntime = 1
    inner_query.problem.domain = newBreach 
    
    candReach, _ = concreach!(inner_query)

    breakCounter = 0

    while !(candReach ⊂ btarget)
        #Create new candidate inner approximation 
        newCenter = deepcopy(newBreach.center)
        newRadius = deepcopy(newBreach.radius * redRate)
        
        #Define new inner approximation 
        newBreach = Hyperrectangle(newCenter, newRadius)
        
        #Test the inner approximation by computing the FRS starting from the inner approximation. If it's an inner approximation, the FRS will be contained in the target set 
        inner_query = deepcopy(bquery)
        inner_query.ntime = 1
        inner_query.problem.domain = newBreach 
        
        candReach, _ = concreach!(inner_query)
        breakCounter += 1

        if breakCounter > 20
            print("Could not compute a maximal set")
            break
        end
    end
    return newBreach
end
function breach_refine_inner_opt(bquery, btarget, method=:UniBisection)
    """
    Refine the inner approximation of the backwards reachable set using an optimization routine. The optimization routine will try to find the largest possible inner approximation of the BRS

    Methods:
        UniBisection: Use bisection method to find a uniform scaling factor across all dimensions
    """
    initBreachSet, _, _ = breach_refine_outer(bquery, btarget)  

    if method == :UniBisection
        lRate = 0.1   # Set minimum reduction rate. NOTE: This is a heuristic value
        hRate = 1.0  # Maximum reduction rate, assuming we start with the full size
        convTol = 0.01  # Tolerance for convergence
        
        #Initialize bestBreach with the outer set 
        bestBreach = Hyperrectangle(deepcopy(initBreachSet.center), deepcopy(initBreachSet.radius)* hRate)
        bestRedRate = hRate

        inner_query = deepcopy(bquery)
        inner_query.ntime = 1 #TODO: We could do this over multiple steps
        inner_query.problem.domain = bestBreach
        # Compute the reachable set
        candReach, _ = concreach!(inner_query)
        
        while (hRate - lRate) > convTol
            midRedRate = (lRate + hRate) / 2
            newRadius = deepcopy(initBreachSet.radius * midRedRate)
            newBreach = Hyperrectangle(deepcopy(initBreachSet.center), newRadius)
            
            # Update query with new breach
            inner_query = deepcopy(bquery)
            inner_query.ntime = 1 #TODO: We could do this over multiple steps
            inner_query.problem.domain = newBreach
            
            # Compute the reachable set
            candReach, _ = concreach!(inner_query)

            if candReach ⊂ btarget
                bestBreach = newBreach
                bestRedRate = midRedRate
                lRate = midRedRate  # Increase the rate to try a larger set
            else
                hRate = midRedRate  # Decrease the rate to reduce the set size
            end
        end
    end
    return bestBreach
end


function multi_step_breach(bquery, btarget, method=:Outer)
    """
    Compute the multi-step backwards reachable set for a given query and target set
    """
    tStart = Dates.now()
    reachSets = []
    boundSets = []
    i =1
    mquery = deepcopy(bquery)
    if method == :Outer
        for i = 1:bquery.ntime
            println("Computing backwards reachable set for time step $i")
            breachSet, oldBreachSets, bboundSets = breach_refine_outer(mquery, btarget)
            push!(reachSets, breachSet)
            push!(boundSets, bboundSets)
            btarget = reachSets[end]
            mquery = deepcopy(bquery)
        end
    elseif method == :Inner
        for i = 1:bquery.ntime
            println("Computing backwards reachable set for time step $i")
            breachSet = breach_refine_inner(mquery, btarget, 0.9)
            push!(reachSets, breachSet)
            btarget = reachSets[end]
            mquery = deepcopy(bquery)
        end
    elseif method == :InnerOpt
        for i = 1:bquery.ntime
            println("Computing backwards reachable set for time step $i")
            breachSet = breach_refine_inner_opt(mquery, btarget)
            push!(reachSets, breachSet)
            btarget = reachSets[end]
            mquery = deepcopy(bquery)
        end
    end
    tStop = Dates.now()
    println("Time to compute multi-step backwards reachable set: $(tStop - tStart)")
    return reachSets, boundSets
end

