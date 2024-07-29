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
    dyn_con_link!(query, neurons)

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

function encode_sym_dynamics!(symQuery)
    optimizer = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
    model = JuMP.Model(optimizer)
    set_silent(model)
    #Define dictionary to store MIP variables
    symQuery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
    symQuery.mod_dict = Dict{Symbol,JuMP.Model}()

    for ind = 1:symQuery.ntime
        LBVec, UBVec = Any[], Any[]
        #Iterate through the vector of bounds and add the bounds to the appropriate vector (upper or lower)
        for oATup in symQuery.problem.bounds[ind]
            LBs, UBs = oATup
            push!(LBVec, LBs)
            push!(UBVec, UBs)
        end

        #Once given the vector of upper and lower bounds, iterate through the vector to generate an overapproximation object that has all the input points present in all the bounds

        #Define universal overappoximation object
        luOA, uuOA = Any[], Any[]
        for i = 1:length(LBVec)-1
            if i == 1 
                luOA = LBVec[i]
                uuOA = UBVec[i]
            end

            #Define the next overapproximation object
            lnOA = LBVec[i+1]
            unOA = UBVec[i+1]

            #Update universal overapproximation object with inputs from the next overapproximation object
            luOA, lnOA = interpol(luOA, lnOA)
            uuOA, unOA = interpol(uuOA, unOA)
        end

        #Now ensure all overapproximation objects are defined over the same set of points 
        uLBVec, uUBVec = Any[], Any[]
        if length(LBVec) > 1
            for vec in LBVec
                #Generate vector with the universal set of points 
                uVec, _ = interpol(vec, luOA)
                push!(uLBVec, uVec)
            end

            for vec in UBVec
                #Generate vector with the universal set of points 
                uVec, _ = interpol(vec, uuOA)
                push!(uUBVec, uVec)
            end
        else
            uLBVec = LBVec
            uUBVec = UBVec
        end

        #Generate universal triangulation 
        Tri = OA2PWA(uLBVec[1])

        #Generate universal input vector 
        xS = [(tup[1:end-1]) for tup in uLBVec[1]]

        #Generate vector of outputs 
        yLBs, yUBs = Any[], Any[]
        for i = 1:length(uLBVec)
            push!(yLBs, [tup[end] for tup in uLBVec[i]])
            push!(yUBs, [tup[end] for tup in uUBVec[i]])
        end

        # sym_MIP = ccSymEncoding(xS, yLBs, yUBs, Tri, symQuery)
        model = ccSymEncoding(xS, yLBs, yUBs, Tri, symQuery, ind, model)
    end

    #Add symbolic MIP to the MIP
    symQuery.mod_dict[:sym] = model
end

function ccSymEncoding(xS, yLBs, yUBs, Tri, symQuery, ind, model)
    """
    Method to encode a piecewise affine function as a mixed integer program following the convex combination method as defined in Gessler et. al. 2012
        (https://www.dl.behinehyab.com/Ebooks/IP/IP011_655874_www.behinehyab.com.pdf#page=308)

    args:
        problem: OvertPProblem that encodes the dynamics of the system as well as some useful system info 
        xS: List of vertices of the triangulation
        yLB: List of lower bounds of the function at the vertices of the triangulation
        yUB: List of upper bounds of the function at the vertices of the triangulation
        Tri: List of simplices of the triangulation
    """

    #Following the convention from Gessler et. al. 
    m = size(xS, 1) #Number of vertices
    d = size(xS[1], 1) #Dimension of the space
    n = size(Tri, 1) #Number of simplices

    #Define indexed symbols for the convex coefficients and binary variables
    lamb_var = Meta.parse("λ_$(ind)")
    bin_var = Meta.parse("b_$(ind)")

    #Define indexed convex coefficients as a MIP variable 
    #NOTE: This is an anonymous variable. Won't appear in named model variables
    λ_var = @variable(model, [1:m], base_name = "$lamb_var")
    # symQuery.var_dict[lamb_var] = λ_var

    #Define indexed binary variables indicating with simplex is active. Use b to avoid conflict with network binary variables 
    b_var = @variable(model, [1:n], Bin, base_name = "$bin_var")
    # symQuery.var_dict[bin_var] = b_var

    #Begin constraining our auxilliary variables
    #Convex combiation constraints (Gessler et. al. eq. 3.2)
    @constraint(model, λ_var .>= 0)
    @constraint(model, sum(λ_var) == 1)

    #This is equation 3.4 from Gessler et. al.
    #Here, we iterate through all vertices. Then, we constrain the convex coefficient of each vertex to be leq the sum of the binary variables corresponding to the simplices containing that vertex

    #NOTE This relates a convex coefficient to its neighbors 
    for j in 1:m
        #Below we find all simplices where index j is present 
        @constraint(model, λ_var[j] <= sum(b_var[i] for i in findall(x -> j in x, Tri)))
    end

    #Next, enforce that at most one simplex can be active at a time (Gessler et. al. eq. 3.5)
    @constraint(model, sum(b_var) <= 1)

    #Create indices for state variables and output variables
    #Use i for input and o for output to avoid conflict with potential variables 
    #TEST: We try to keep control and input variables independent of overapproximation variables (y)
    x_ind = Meta.parse("i_$(ind)")
    u_ind = Meta.parse("u_$(ind)")

    #Now, define function variables as MIP variables
    x_var = @variable(model, [1:d], base_name = "$x_ind")
    # symQuery.var_dict[x_ind] = x_var
    u = @variable(model, [1], base_name = "$u_ind")
    # symQuery.var_dict[u_ind] = u

    #Defines the generic vertex as a convex combination of its neighbors 
    #This exploits casting. NOTE: Could be dangerous 
    @constraint(model, x_var .== sum(λ_var[i]*[xS[i]...] for i in 1:m))

    yCounter = 1
    for sym in symQuery.problem.varList
        #Define yLB and yUB for the corresponding symbol 
        yLB = yLBs[yCounter]
        yUB = yUBs[yCounter]
        yCounter += 1
        y_ind = Meta.parse("o_$(sym)_$(ind)")
        yl_ind = Meta.parse("ol_$(sym)_$(ind)")
        yu_ind = Meta.parse("ou_$(sym)_$(ind)")
        y_var = @variable(model, [1], base_name = "$y_ind")
        # symQuery.var_dict[y_ind] = y_var
        yₗ = @variable(model, [1], base_name = "$yl_ind")
        # symQuery.var_dict[yl_ind] = yₗ
        yᵤ = @variable(model, [1], base_name = "$yu_ind")
        # symQuery.var_dict[yu_ind] = yᵤ
        #Define the generic function value in terms of the convex combination of its upper and lower bounds
        #NOTE: Control is changed here. Very bad 
        #TODO: Fix how control is handled
        @constraint(model, yₗ[1] == sum(λ_var[i]*yLB[i] for i in 1:m))
        @constraint(model, yᵤ[1] == sum(λ_var[i]*yUB[i] for i in 1:m))
        @constraint(model, yₗ[1] + symQuery.problem.control_coef*u[1] <= y_var[1])
        @constraint(model, y_var[1] <= yᵤ[1] + symQuery.problem.control_coef*u[1])

        #Add model inputs and outputs to variable dictionary
        #NOTE: x_var and u are independent of sym. We simply duplicate them for each symbol
        sym_ind = Meta.parse("$(sym)_$(ind)")
        symQuery.var_dict[sym_ind] = [y_var]
    end
    #Time_Ind holds non bound variables to avoid duplication
    time_ind = Meta.parse("t_$(ind)")
    symQuery.var_dict[time_ind] = [x_var, u]

    #We will also need to define additional constraints on x and y, but those will be added later

    return model 
end

function encode_sym_control(symQuery, reachSets)
    network_file = symQuery.network_file
    #Get dictionary of MIP variables 

    #####Enter loop for time steps#################
    for i = 1:symQuery.ntime
        #Same MIP for all symbols and time steps
        sym_mip = symQuery.mod_dict[:sym]
        #Select x and u for the appropriate time step
        input_set = reachSets[i]
        time_curr = Meta.parse("t_$(i)")
        x_curr = symQuery.var_dict[time_curr][1]
        u_curr = symQuery.var_dict[time_curr][2]

        controller_bound = add_controller_constraints!(sym_mip, network_file, input_set, x_curr, u_curr)

    end
end

function encode_time(symQuery::OvertPQuery)
    """
    Method to encode the time evolution of the system as a MIP. Links distinct overappoximation objects across time steps to complete the symbolic encoding. 
    """
    ##########First loop#############
    for i = 1:symQuery.ntime-1 
        j = 0
        trueOut = Any[]
        #First, define the inputs that go into the integration map
        for sym in symQuery.problem.varList 
            j += 1
            #Use metaprogramming to get the current and next symbol
            sym_now = Meta.parse("$(sym)_$(i)")
            #Get the MIP variables associated with the symbols
            y_now = symQuery.var_dict[sym_now][1]
            push!(trueOut, y_now)
        end

        #Explicitly define the MIP model
        sym_mip = symQuery.mod_dict[:sym]
       
        #Inputs are identical across symbols, select one at random to define the current and next inputs 
        sym = symQuery.problem.varList[1]

        #Use metaprogramming to get the current and next symbol
        time_now = Meta.parse("t_$(i)")
        time_next = Meta.parse("t_$(i+1)")

        #Get the input variables associated with the symbols
        x_now = symQuery.var_dict[time_now][1]
        x_next = symQuery.var_dict[time_next][1]

        #Define the integration map 
        integration_map = symQuery.problem.update_rule(x_now, trueOut)

        #######Second loop#############
        #Iterate through each input variable, find the variable that controls its evolution, then link the input variable of the next time step to the output variable of the current time step
        for k = 1:length(x_now)
            v = x_now[k]
            dv = integration_map[v]
            next_v = x_next[k]
            @constraint(sym_mip, next_v == v + symQuery.dt*dv)
        end

    end
end

function symReach(symQuery::OvertPQuery, reachSets=nothing)
    """
    Method to solve the concrete reachability problem using MIP for multiple time steps.
    """
    #Define dictionary to store MIP variables
    symQuery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
    symQuery.mod_dict = Dict{Symbol,JuMP.Model}()

    #Encode the dynamics over the trajectory
    encode_sym_dynamics!(symQuery)

    #Encode the controller if it exists
    #TODO: Fix this 
    if !isnothing(symQuery.network_file)
        encode_sym_control(symQuery, reachSets)
    end

    #Link the time steps
    encode_time(symQuery)

    #Solve the reachability problem
    # t1 = Dates.now()
    reachSet = sym_reach_solve(symQuery, symQuery.ntime)
    # t2 = Dates.now()
    # println("Time for symbolic reach is ", t2-t1)
    return reachSet
end

function sym_reach_solve(query, t_idx::Union{Nothing,Int64}=nothing)
    """
    Solve contrete reachability problem using MIP. Given a MIP encoding a pwl overapproximation of the dynamics, as well as MIP for the controlller, find the overapproximation of the reachable set

    This version of reach_solve generalizes to multiple functions
    """
    trueInp = Any[]
    trueOut = Any[]
    
    #Compute true input and output variables 
    for sym in query.problem.varList
        #Account for symbolic case where dynamics are timed
        sym_timed = Meta.parse("$(sym)_$t_idx")
        output_vars = query.var_dict[sym_timed][1]
        push!(trueOut, output_vars)
    end
    time_sym = Meta.parse("t_$(t_idx)")
    trueInp = query.var_dict[time_sym][1]

    integration_map = query.problem.update_rule(trueInp, trueOut)
    
    timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)

    mipModel = query.mod_dict[:sym]

    for v in trueInp 
        dv = integration_map[v]
        next_v = v + query.dt*dv
        push!(timestep_nplus1_vars, next_v)
        @objective(mipModel, Min, next_v)
        JuMP.optimize!(mipModel)
        @assert termination_status(mipModel) == MOI.OPTIMAL
        objective_bound(mipModel)
        push!(lows, objective_bound(mipModel))
        @objective(mipModel, Max, next_v)
        JuMP.optimize!(mipModel)
        @assert termination_status(mipModel) == MOI.OPTIMAL
        objective_bound(mipModel)
        push!(highs, objective_bound(mipModel))
    end
    #NOTE: Hyperrectangle can plot in higher dimensions as well
    reacheable_set = Hyperrectangle(low=lows, high=highs)
    return reacheable_set
end

function multi_step_symreach(symQuery::OvertPQuery)
    """
    Method to compute symbolic reachable sets for multiple time steps using concrete (or symbolic) bounds
    """
    input_set = symQuery.problem.domain
    reachSets = [input_set]
    totTime = copy(symQuery.ntime)
    t1 = Dates.now()
    for i = 1:totTime
        symQuery.ntime = i
        reachSet = symReach(symQuery, reachSets);
        push!(reachSets, reachSet)
    end
    t2 = Dates.now()
    println("Time for symbolic reachability is ", t2-t1)
    return reachSets
end

function multi_step_hybreach(concEvery, query)
    """
    Method to compute hybrid reachable sets by using concrete reach sets as a base and then using symbolic reach sets to refine the reachable set
    """
    totalReachSets = [domain]
    symReachSets = []
    totalsteps = copy(query.ntime)
    numConc = ceil(totalsteps/concEvery)
    totalBoundSets = []
    # tStart = Dates.now()
    cumSteps = 0
    query.ntime = concEvery
    for i = 1:numConc

        if i == 1
            newDomain = domain
        else
            newDomain = symReachSets[end]
        end
        concQuery = deepcopy(query)
        concQuery.problem.domain = newDomain
        if cumSteps + concQuery.ntime > totalsteps
            concQuery.ntime = totalsteps - cumSteps
        end
        reachSets, boundSets = multi_step_concreach(concQuery)
        push!(totalBoundSets, boundSets...)
        
        symQuery = deepcopy(query)
        symQuery.problem.bounds = boundSets
        if cumSteps + symQuery.ntime > totalsteps
            symQuery.ntime = totalsteps - cumSteps
        end
        symReachSets = multi_step_symreach(symQuery)
        cumSteps += symQuery.ntime
        push!(totalReachSets, symReachSets[2:end]...)
    end
    return totalReachSets, totalBoundSets
end

function straight_shot_reach(totalReachSets, query)
    """
    Method to compute symbolic reachable sets given a set of initial (concrete or symbolic) reachable sets
    """
    symBoundSets = Any[]
    for set in totalReachSets
        itQuery = deepcopy(query)
        itQuery.problem.domain = set
        itBound = itQuery.problem.bound_func(itQuery.problem)
        push!(symBoundSets, itBound)
    end


    query.problem.bounds = symBoundSets
    reach_set = symReach(query, totalReachSets)
    return reach_set
end

function multi_shot_reach(query)
    reachSets = [query.problem.domain]
    timeHorizon = copy(query.ntime)

    for i = 1:timeHorizon
        #Compute the straight shot reach set up to the current time step
        queryCopy = deepcopy(query)
        queryCopy.ntime = i
        reach_set = straight_shot_reach(reachSets, queryCopy)
        push!(reachSets, reach_set)
    end
    return reachSets
end

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

