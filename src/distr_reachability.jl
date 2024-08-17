include("overtPoly_helpers.jl")
include("nn_mip_encoding.jl")
include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")
include("problems.jl")
using LazySets
using Random
using Plasmo

function encode_dynamics!(query::OvertPQuery)
    #create an optigraph to store the model
    graph = OptiGraph()
    #create a vector of models for the dynamics 
    dynNodes = @optinode(graph, nodes[1:length(query.problem.varList)])
    ind = 0
    #Iterate through elements of varList and add appropriate variables to the appropriate model 
    for sym in query.problem.varList
        ind += 1
        LB, UB = query.problem.bounds[ind]
        Tri = OA2PWA(LB)
        xS = [(tup[1:end-1]) for tup in LB]
        yUB = [tup[end] for tup in UB]
        yLB = [tup[end] for tup in LB]

        ccEncoding!(xS, yLB, yUB, Tri, query, query.problem.varList[ind], ind, dynNodes[ind])
    end

    #Reuse mod dict to store graph, dynNodes, and neural network 
    query.mod_dict[:graph] = graph
    query.mod_dict[:f] = dynNodes
end
###Create new Encode Control Function###
function encode_control!(query)
    input_set = query.problem.control_func(query.problem.domain)
    network_file = query.network_file
    netModel = @optinode(query.mod_dict[:graph])
    neurons = add_controller_constraints!(netModel, network_file, input_set, Id())
    query.mod_dict[:u] = netModel
    return neurons
end

function conc_reach_solve(query)
    max_query = deepcopy(query)
    min_query = deepcopy(query)
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)
    min_dynModel = min_query.mod_dict[:f]
    minGraph = min_query.mod_dict[:graph]
    i = 0
    #Compute lower bounds
    for sym in min_query.problem.varList 
        i += 1
        model = min_dynModel[i]
        v = min_query.var_dict[sym][end][1]
        dv = min_query.var_dict[sym][2][1]
        @variable(model, next_v)
        @constraint(model, next_v == v + query.dt*dv)
        @objective(model, Min, next_v)
    end

    set_optimizer(minGraph, Gurobi.Optimizer)
    optimize!(minGraph)
    @assert termination_status(minGraph) == MOI.OPTIMAL
    i = 0
    for _ in query.problem.varList
        i += 1
        push!(lows, value(min_dynModel[i][:next_v]))
    end

    #Compute upper bounds
    max_dynModel = max_query.mod_dict[:f]
    maxGraph = max_query.mod_dict[:graph]
    i = 0
    for sym in query.problem.varList 
        i += 1
        model = max_dynModel[i]
        v = max_query.var_dict[sym][end][1]
        dv = max_query.var_dict[sym][2][1]
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

function concreach!(query::OvertPQuery)
    query.problem.bounds = query.problem.bound_func(query.problem)
    query.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
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

function multi_step_concreach(query::OvertPQuery)
    """
    Method to solve the concrete reachability problem using MIP for multiple time steps.
    """
    input_set = query.problem.domain
    reachSets = [input_set]
    boundSets = []
    
    #t1 = Dates.now()
    for i = 1:query.ntime
        reachSet, boundSet = concreach!(query)
        push!(reachSets, reachSet)
        push!(boundSets, boundSet)
        query.problem.domain = reachSet
    end
    # t2 = Dates.now()
    # println("Time for concrete reachability is ", t2-t1)
    return reachSets, boundSets
end

##################Implementing Sym Encoding###########



 f(x1, x2) = 2x1 + 1x2
 f(x1, x2, x3) = 2x1 + 1x2 + 0x3
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

