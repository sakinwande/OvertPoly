function sameInp(LB, UB)
    """
    Takes an overt OA and interpolates to ensure that the lower and upper bounds are over the same set of points
    """
    newUB = Any[]
    newLB = Any[]

    newXs = sort(unique(vcat([tup[1] for tup in LB], [tup[1] for tup in UB])))

    for inp in newXs
        #Check if this input has a lower bound 
        lbInd = findall(x->x[1] == inp, LB)

        #If it does, add to newLB, else interpolate
        if !isempty(lbInd)
            push!(newLB, LB[lbInd[1]])
        else
            #Find the lower bound that is closest to the input
            lbInd = findall(x->x[1] < inp, LB)
            #Interpolate
            push!(newLB, (inp, interpol(inp, LB[lbInd[end]], LB[lbInd[end]+1])))
        end

        #Check if this input has an upper bound
        ubInd = findall(x->x[1] == inp, UB)

        #Similarly, if it does, add to newUB, else interpolate
        if !isempty(ubInd)
            push!(newUB, UB[ubInd[1]])
        else
            #Find the upper bound that is closest to the input
            ubInd = findall(x->x[1] < inp, UB)
            #Interpolate
            push!(newUB, (inp, interpol(inp, UB[ubInd[end]], UB[ubInd[end]+1])))
        end

    end

    return newLB, newUB

end

function interpol(xInp, tuplb, tupub)
    """
    When it's not catching international criminals, this method interpolates the lower and upper bounds of a tuple to ensure that the bounds are over the same set of points

    Args:
        xInp: the point at which we wish to generate an approximation
        tuplb: A tuple of (xlb, y) where xlb < xInp
        tupub: A tuple of (xub, y) where xub > xInp
        
    """

    #Assert that the input is between lb and ub
    @assert xInp >= tuplb[1] && xInp <= tupub[1]

    #Assert that the upper bound and lower boud are distinct 
    @assert tuplb[1] != tupub[1]

    #Use linear interpolation to find the output given UB and LB
    yInp = tuplb[2] + (xInp - tuplb[1])*(tupub[2] - tuplb[2])/(tupub[1] - tuplb[1])

    return yInp
end

function interpol(oA1, oA2)
    """
    When it is not catching international criminals, this method interpolates two overt approximations to ensure that they are over the same set of points
    """
    #Create a new array of inputs that are the same for both approximations
    #TODO: Consider using rounding here to ensure that the points are the same
    #Unsound flag, this is removing symmetric points. Basically, 
    newInps = sort(unique(vcat([tup[1:end-1] for tup in oA1], [tup[1:end-1] for tup in oA2])))

    #Generate interpolation function for each approximation
    interp1 = gen_interpol(oA1)
    interp2 = gen_interpol(oA2)

    #Loop through both approximations and interpolate to ensure that the bounds are over the same set of points
    newOA1 = Any[(tup..., interp1(tup...)) for tup in newInps]
    newOA2 = Any[(tup..., interp2(tup...)) for tup in newInps]

    return newOA1, newOA2

end


function encode_dynamics!(query::OvertPQuery)
    """
    Method to encode the dynamics as a MIP (using the Gessing et al. method)
    """
    LB, UB = query.problem.bounds
    Tri = OA2PWA(LB)
    #These are the vertices of the triangulation
    xS = [(tup[1:end-1]) for tup in LB]
    yUB = [tup[end] for tup in UB]
    yLB = [tup[end] for tup in LB]

    mipModel = ccEncoding(xS, yLB, yUB, Tri, query)
    return mipModel
end

function encode_control!(query, mipModel)
    """
    Method to encode the controller as a MIP (using the Tjeng et al. method)
    Modifies the MIP model in place
    """
    input_set = query.problem.domain   ###For controller MIP encoding, need model, network address, input set, input variable names, output variable names
    network_file = query.network_file

    #Get dictionary of MIP variables 
    input_vars = query.var_dict[:x]
    control_vars = query.var_dict[:u][1]
    output_vars = query.var_dict[:y][1]

    controller_bound = add_controller_constraints!(mipModel, network_file, input_set, input_vars, control_vars)
end


function reach_solve(mipModel, query, t_idx::Union{Nothing,Int64}=nothing)
    """
    Solve contrete reachability problem using MIP. Given a MIP encoding a pwl overapproximation of the dynamics, as well as MIP for the controlller, find the overapproximation of the reachable set
    """
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)
    


    #Account for case where dynamics are timed
    if !isnothing(t_idx)
        input_vars = query.var_dict[Meta.parse("x_$t_idx")]
        control_vars = query.var_dict[Meta.parse("u_$t_idx")][1]
        output_vars = query.var_dict[Meta.parse("y_$t_idx")][1]
        integration_map = query.problem.update_rule(input_vars, control_vars, [output_vars])
    else
        control_vars = query.var_dict[:u][1]
        input_vars = query.var_dict[:x]
        output_vars = query.var_dict[:y][1]
        integration_map = query.problem.update_rule(input_vars, control_vars, [output_vars]) 
    end
    
    timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
    #####Enter loop here#################
    #Loop over elements of input_vars_last
    for v in input_vars
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
    
    reacheable_set = Hyperrectangle(low=lows, high=highs)
    return reacheable_set
end

function concreach!(query::OvertPQuery)
    """
    Method to solve the concrete reachability problem using MIP.
    Modifies the query object in place (specifically the bounds generated by OVERT)
    """
    query.problem.bounds = query.bound_func(query.problem)
    query.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
    query.mod_dict = Dict{Symbol,JuMP.Model}()
    encode_dynamics!(query)

    #Encode the controller if it exists
    if query.network_file != nothing
        encode_control!(query)
    end

    reachSet =  reach_solve(mipModel, query)
    return reachSet, query.problem.bounds
end


function multi_step_concreach(query::OvertPQuery)
    """
    Method to solve the concrete reachability problem using MIP for multiple time steps.
    """
    input_set = query.problem.domain
    reachSets = [input_set]
    boundSets = []
    
    for i = 1:query.ntime
        t1 = Dates.now()
        reachSet, boundSet = concreach!(query)
        t2 = Dates.now()
        println("Time for step ", i, " is ", t2-t1)
        push!(reachSets, reachSet)
        push!(boundSets, boundSet)
        query.problem.domain = reachSet
    end
    return reachSets, boundSets
end

###############Individual Sym MIP Encoding################
function ccSymEncoding(xS, yLB, yUB, Tri, symQuery, ind, model)
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
    symQuery.var_dict[lamb_var] = λ_var

    #Define indexed binary variables indicating with simplex is active. Use b to avoid conflict with network binary variables 
    b_var = @variable(model, [1:n], Bin, base_name = "$bin_var")
    symQuery.var_dict[bin_var] = b_var


    #Begin constraining our auxilliary variables
    #Convex combiation constraints (Gessler et. al. eq. 3.2)
    symQuery.var_dict
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
    x_ind = Meta.parse("x_$(ind)")
    y_ind = Meta.parse("y_$(ind)")
    yl_ind = Meta.parse("yl_$(ind)")
    yu_ind = Meta.parse("yu_$(ind)")
    u_ind = Meta.parse("u_$(ind)")

    #Now, define function variables as MIP variables
    x_var = @variable(model, [1:d], base_name = "$x_ind")
    symQuery.var_dict[x_ind] = x_var
    y_var = @variable(model, [1], base_name = "$y_ind")
    symQuery.var_dict[y_ind] = y_var
    yₗ = @variable(model, [1], base_name = "$yl_ind")
    symQuery.var_dict[yl_ind] = yₗ
    yᵤ = @variable(model, [1], base_name = "$yu_ind")
    symQuery.var_dict[yu_ind] = yᵤ
    u = @variable(model, [1], base_name = "$u_ind")
    symQuery.var_dict[u_ind] = u

    #Define the generic vertex as a convex combination of its neighbors 
    #This exploits casting. NOTE: Could be dangerous 
    @constraint(model, x_var .== sum(λ_var[i]*[xS[i]...] for i in 1:m))

    #Define the generic function value in terms of the convex combination of its upper and lower bounds
    #NOTE: Control is changed here. Very bad 
    @constraint(model, yₗ[1] == sum(λ_var[i]*yLB[i] for i in 1:m))
    @constraint(model, yᵤ[1] == sum(λ_var[i]*yUB[i] for i in 1:m))
    @constraint(model, yₗ[1] + symQuery.problem.control_coef*u[1] <= y_var[1])
    @constraint(model, y_var[1] <= yᵤ[1] + symQuery.problem.control_coef*u[1])


    #We will also need to define additional constraints on x and y, but those will be added later
    return model 

end

####### Encode Symbolic Dynamics ########
function encode_sym_dynamics(symQuery::OvertPQuery)
    """
    Function to encode the dynamics over a trajectory of overapproximations. 
    TODO: Revisit this. Can be done more efficiently by exploiting the overlap between the domains of the overapproximations. This should result in significantly fewer vertices in the final MIP encoding.
    """
    #Define symbolic MIP model
    optimizer = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
    #Define model
    symMip = Model(optimizer)

    #Define dictionary to store MIP variables
    symQuery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()

    for ind in 1:size(symQuery.problem.bounds)[1]
        boundVec = symQuery.problem.bounds[ind]
        LBs = boundVec[1]
        UBs = boundVec[2]

        Tri = OA2PWA(LBs)

        xS = [(tup[1:end-1]) for tup in LBs]
        yUB = [tup[end] for tup in UBs]
        yLB = [tup[end] for tup in LBs]

        symMip = ccSymEncoding(xS, yLB, yUB, Tri, symQuery, ind, symMip)
    end

    symMip
    return symMip
end

#######Encode Symbolic Controller########
function encode_sym_control(sym_mip, symQuery, reachSets)
    input_set = symQuery.problem.domain  
    network_file = symQuery.network_file
    ntime = symQuery.ntime
    #Get dictionary of MIP variables 
    varDict = symQuery.var_dict

    #####Enter loop for time steps#################
    for i = 1:symQuery.ntime
        input_set = reachSets[i]
        input_vars_curr = Meta.parse("x_$i") 
        control_vars_curr = Meta.parse("u_$i") 

        x_curr = varDict[input_vars_curr]
        u_curr = varDict[control_vars_curr][1]

        controller_bound = add_controller_constraints!(sym_mip, network_file, input_set, x_curr, u_curr)
    end
    return sym_mip
end

##########Symbolically link time steps########
function encode_time(sym_mip, symQuery)
    """
    Method to encode the time evolution of the system as a MIP. Links distinct overappoximation objects across time steps to complete the symbolic encoding. 
    """
    ##########First loop#############
    for i = 1:symQuery.ntime-1
        y_now = symQuery.var_dict[Meta.parse("y_$i")]
        u_now = symQuery.var_dict[Meta.parse("u_$i")]
        x_now = symQuery.var_dict[Meta.parse("x_$i")]
        x_next = symQuery.var_dict[Meta.parse("x_$(i+1)")]

        integration_map = single_pend_update_rule(x_now, u_now, y_now)

        #######Second loop#############
        for j = 1:length(x_now)
            v = x_now[j]
            dv = integration_map[v]
            next_v = x_next[j]

            @constraint(sym_mip, next_v == v + symQuery.dt*dv)
        end
    end
    return sym_mip
end

function bound_univariate(baseExpr::Expr, lb, ub; ϵ=1e-4, npoint=2, rel_error_tol=5e-3, plotflag=false)
    """
    Method to bound a univariate function

    Args: 
        Expr: A Julia symbolic expression. Assumed to have been parsed/converted to a univariate function 
        lb: Lower bound of the interval over which to bound the function
        ub: Upper bound of the interval over which to bound the function
    """

    #Found the symbolic variable in the Julia expression (for replacement)
    varBase = find_variables(baseExpr)[1]

    #Define differentiation variable
    @variables xₚ

    #Define derivative
    D = Differential(xₚ)
    #Define second derivative
    D2 = Differential(xₚ)^2

    #Replace expression variable with xₚ
    #NB:  we need to do this because the expression will eventually be parsed by Julia Symbolics. For the parser to work, the symbols of the expression must not be defined in the parsing module.

    #I choose xₚ because I can define a case to ensure we don't use this variable in input expressions

    #NOTE: you can actually do better. Convert baseExpr to symExpr in its native form, then use symbolics.get_variables to get the variables in the expression. Next, use Symbolics.substitute to replace problem specific variables with xₚ. This way, you avoid using Meta.parse on a string
    strExpr = string(baseExpr)
    strExpr = replace(strExpr, string(varBase) => "xₚ")
    standExpr = Meta.parse(strExpr) #Standardized expression with xₚ as the variable

    symExpr = Symbolics.parse_expr_to_symbolic(standExpr, Main)

    #Return a standard Julia function that can be evaluated from the expression
    fun = Symbolics.build_function(baseExpr, varBase, expression=Val{false})
    df = expand_derivatives(D(eval(symExpr)))
    dfunc = Symbolics.build_function(df, :xₚ; expression=Val{false})

    #Compute second derivative
    d2f = expand_derivatives(D2(symExpr))
    #d2f is an expression. Convert to a Julia function so IntervalRootFinding can use it
    #NB: expression=Val{false} returns a runtime gen function to avoid world age issues. This way we avoid evaluating expressions 

    #TODO: Debug this to elegantly handle case of linear functions
    if true
        d2func = Symbolics.build_function(d2f, :xₚ; expression = Val{false})

        #Then find the roots over the given interval using the function
        rootVals = IntervalRootFinding.roots(d2func, IntervalArithmetic.Interval(lb, ub))
        #TODO: This is not sound, make sound
        rootsGuess = [mid.([root.interval for root in rootVals])]
        d2f_zeros = sort(rootsGuess[1])


        convex = nothing 

        UB = bound(fun, lb, ub, npoint; rel_error_tol=rel_error_tol, conc_method="continuous", lowerbound=false, df = dfunc,d2f = d2func, d2f_zeros=d2f_zeros, convex=nothing, plot=true)
        UBpoints = unique(sort(to_pairs(UB), by = x -> x[1]))

        LB = bound(fun, lb, ub, npoint; rel_error_tol=rel_error_tol, conc_method="continuous", lowerbound=true, df = dfunc,d2f = d2func, d2f_zeros=d2f_zeros, convex=nothing, plot=true)
        LBpoints = unique(sort(to_pairs(LB), by = x -> x[1]))
    else
        #Account for linear case
        UBpoints = [(lb, fun(lb)), (ub, fun(ub))] 
        LBpoints = [(lb, fun(lb)), (ub, fun(ub))]
    end

    try 
        #Linear case 
        if d2f == 0
            UBpoints = [(lb, fun(lb)), (ub, fun(ub))] 
            LBpoints = [(lb, fun(lb)), (ub, fun(ub))]
        end
    catch
        #Nonlinear case 
        d2func = Symbolics.build_function(d2f, :xₚ; expression = Val{false})

        #Then find the roots over the given interval using the function
        rootVals = IntervalRootFinding.roots(d2func, IntervalArithmetic.Interval(lb, ub))
        #TODO: This is not sound, make sound
        rootsGuess = [mid.([root.interval for root in rootVals])]
        d2f_zeros = sort(rootsGuess[1])


        convex = nothing 

        UB = bound(fun, lb, ub, npoint; rel_error_tol=rel_error_tol, conc_method="continuous", lowerbound=false, df = dfunc,d2f = d2func, d2f_zeros=d2f_zeros, convex=nothing, plot=true)
        UBpoints = unique(sort(to_pairs(UB), by = x -> x[1]))

        LB = bound(fun, lb, ub, npoint; rel_error_tol=rel_error_tol, conc_method="continuous", lowerbound=true, df = dfunc,d2f = d2func, d2f_zeros=d2f_zeros, convex=nothing, plot=true)
        LBpoints = unique(sort(to_pairs(LB), by = x -> x[1]))
    end

    if plotflag
        plotRes2d(baseExpr, fun, lb, ub, LBpoints, UBpoints, varBase, true)
    end
    
    return UBpoints, LBpoints
end

function ccSymEncoding(xS, yLB, yUB, Tri, symQuery, ind, sym, model)
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
    lamb_var = Meta.parse("λ_$(sym)_$(ind)")
    bin_var = Meta.parse("b_$(sym)_$(ind)")

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
    x_ind = Meta.parse("x_$(sym)_$(ind)")
    y_ind = Meta.parse("y_$(sym)_$(ind)")
    yl_ind = Meta.parse("yl_$(sym)_$(ind)")
    yu_ind = Meta.parse("yu_$(sym)_$(ind)")
    u_ind = Meta.parse("u_$(sym)_$(ind)")

    #Now, define function variables as MIP variables
    x_var = @variable(model, [1:d], base_name = "$x_ind")
    # symQuery.var_dict[x_ind] = x_var
    y_var = @variable(model, [1], base_name = "$y_ind")
    # symQuery.var_dict[y_ind] = y_var
    yₗ = @variable(model, [1], base_name = "$yl_ind")
    # symQuery.var_dict[yl_ind] = yₗ
    yᵤ = @variable(model, [1], base_name = "$yu_ind")
    # symQuery.var_dict[yu_ind] = yᵤ
    u = @variable(model, [1], base_name = "$u_ind")
    # symQuery.var_dict[u_ind] = u

    #Define the generic vertex as a convex combination of its neighbors 
    #This exploits casting. NOTE: Could be dangerous 
    @constraint(model, x_var .== sum(λ_var[i]*[xS[i]...] for i in 1:m))

    #Define the generic function value in terms of the convex combination of its upper and lower bounds
    #NOTE: Control is changed here. Very bad 
    @constraint(model, yₗ[1] == sum(λ_var[i]*yLB[i] for i in 1:m))
    @constraint(model, yᵤ[1] == sum(λ_var[i]*yUB[i] for i in 1:m))
    @constraint(model, yₗ[1] + symQuery.problem.control_coef*u[1] <= y_var[1])
    @constraint(model, y_var[1] <= yᵤ[1] + symQuery.problem.control_coef*u[1])

    #Add model inputs and outputs to variable dictionary
    sym_ind = Meta.parse("$(sym)_$(ind)")
    symQuery.var_dict[sym_ind] = [x_var, y_var, u]

    #We will also need to define additional constraints on x and y, but those will be added later
    return model 

end

####### Encode Symbolic Dynamics ########
function encode_sym_dynamics!(symQuery::OvertPQuery)
    """
    Function to encode the dynamics over a trajectory of overapproximations. 
    TODO: Revisit this. Can be done more efficiently by exploiting the overlap between the domains of the overapproximations. This should result in significantly fewer vertices in the final MIP encoding.
    """
    #Define symbolic MIP model
    optimizer = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
    #Define dictionary to store MIP variables
    symQuery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
    symQuery.mod_dict = Dict{Symbol,JuMP.Model}()

    for ind in 1:symQuery.ntime
        #index used to iterate over variables with overapproximations
        i = 0
        for sym in symQuery.problem.varList
            i += 1
            LBs, UBs = symQuery.problem.bounds[ind][i]
            Tri = OA2PWA(LBs)
            xS = [(tup[1:end-1]) for tup in LBs]
            yUB = [tup[end] for tup in UBs]
            yLB = [tup[end] for tup in LBs]

            #If this is the first index, initialize model, otherwise use the model associated with the symbol
            if ind == 1
                symMip = Model(optimizer)
            else
                symMip = symQuery.mod_dict[sym]
            end

            symQuery.mod_dict[sym] = ccSymEncoding(xS, yLB, yUB, Tri, symQuery, ind, sym, symMip)
        end
    end
end

#######Encode Symbolic Controller########
function encode_sym_control(symQuery, reachSets)
    network_file = symQuery.network_file
    #Get dictionary of MIP variables 

    #####Enter loop for time steps#################
    for i = 1:symQuery.ntime
        for sym in symQuery.problem.varList
            sym_mip = symQuery.mod_dict[sym]
            input_set = reachSets[i]
            sym_curr = Meta.parse("$(sym)_$(i)")
            x_curr = symQuery.var_dict[sym_curr][1]
            u_curr = symQuery.var_dict[sym_curr][3]
            y_curr = symQuery.var_dict[sym_curr][2]

            controller_bound = add_controller_constraints!(sym_mip, network_file, input_set, x_curr, u_curr)
        end
    end
end

##########Symbolically link time steps########
function encode_time(symQuery::OvertPQuery)
    """
    Method to encode the time evolution of the system as a MIP. Links distinct overappoximation objects across time steps to complete the symbolic encoding. 
    """
    ##########First loop#############
    for i = 1:symQuery.ntime-1 
        j = 0
        trueInp = Any[]
        trueOut = Any[]
        #First, define the inputs that go into the integration map
        for sym in symQuery.problem.varList 
            j += 1
            #Use metaprogramming to get the current and next symbol
            sym_now = Meta.parse("$(sym)_$(i)")
            sym_next = Meta.parse("$(sym)_$(i+1)")
            #Get the MIP variables associated with the symbols
            y_now = symQuery.var_dict[sym_now][2]
            u_now = symQuery.var_dict[sym_now][3]
            x_now = symQuery.var_dict[sym_now][1]
            x_next = symQuery.var_dict[sym_next][1]

            #Account for different cases 
            if query.case == 1
                push!(trueInp, x_now[j])
                push!(trueOut, y_now)
            else
                push!(trueInp, x_now[j:j+1])
                push!(trueOut, y_now)
            end
        end

        #Define the integration map 
        integration_map = symQuery.problem.update_rule(trueInp, trueOut)
        #Iterate through each MIP model and link current and future time steps
        for sym in symQuery.problem.varList
            #Use metaprogramming to get the current and next symbol
            sym_now = Meta.parse("$(sym)_$(i)")
            sym_next = Meta.parse("$(sym)_$(i+1)")
            #Get the MIP variables associated with the symbols
            y_now = symQuery.var_dict[sym_now][2]
            u_now = symQuery.var_dict[sym_now][3]
            x_now = symQuery.var_dict[sym_now][1]
            x_next = symQuery.var_dict[sym_next][1]

            sym_mip = symQuery.mod_dict[sym]

            #######Second loop#############
            #Iterate through each input variable, find the variable that controls its evolution, then link the input variable of the next time step to the output variable of the current time step
            for k = 1:length(x_now)
                v = x_now[k]
                if v in trueInp
                    dv = integration_map[v][1]
                    next_v = x_next[k]

                    @constraint(sym_mip, next_v == v + symQuery.dt*dv)
                end
            end
        end
    end
end

function symReach(symQuery::OvertPQuery)
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
        encode_sym_control(symQuery)
    end

    #Link the time steps
    encode_time(symQuery)

    #Solve the reachability problem
    t1 = Dates.now()
    reachSet = reach_solve(symQuery, symQuery.ntime)
    t2 = Dates.now()
    println("Time for symbolic reach is ", t2-t1)
    return reachSet
end


function ccSymEncoding(xS, yLB, yUB, Tri, symQuery, ind, sym, model)
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
    lamb_var = Meta.parse("λ_$(sym)_$(ind)")
    bin_var = Meta.parse("b_$(sym)_$(ind)")

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
    x_ind = Meta.parse("x_$(sym)_$(ind)")
    y_ind = Meta.parse("y_$(sym)_$(ind)")
    yl_ind = Meta.parse("yl_$(sym)_$(ind)")
    yu_ind = Meta.parse("yu_$(sym)_$(ind)")
    u_ind = Meta.parse("u_$(sym)_$(ind)")

    #Now, define function variables as MIP variables
    x_var = @variable(model, [1:d], base_name = "$x_ind")
    # symQuery.var_dict[x_ind] = x_var
    y_var = @variable(model, [1], base_name = "$y_ind")
    # symQuery.var_dict[y_ind] = y_var
    yₗ = @variable(model, [1], base_name = "$yl_ind")
    # symQuery.var_dict[yl_ind] = yₗ
    yᵤ = @variable(model, [1], base_name = "$yu_ind")
    # symQuery.var_dict[yu_ind] = yᵤ
    u = @variable(model, [1], base_name = "$u_ind")
    # symQuery.var_dict[u_ind] = u

    #Define the generic vertex as a convex combination of its neighbors 
    #This exploits casting. NOTE: Could be dangerous 
    @constraint(model, x_var .== sum(λ_var[i]*[xS[i]...] for i in 1:m))

    #Define the generic function value in terms of the convex combination of its upper and lower bounds
    #NOTE: Control is changed here. Very bad 
    @constraint(model, yₗ[1] == sum(λ_var[i]*yLB[i] for i in 1:m))
    @constraint(model, yᵤ[1] == sum(λ_var[i]*yUB[i] for i in 1:m))
    @constraint(model, yₗ[1] + symQuery.problem.control_coef*u[1] <= y_var[1])
    @constraint(model, y_var[1] <= yᵤ[1] + symQuery.problem.control_coef*u[1])

    #Add model inputs and outputs to variable dictionary
    sym_ind = Meta.parse("$(sym)_$(ind)")
    symQuery.var_dict[sym_ind] = [x_var, y_var, u]

    #We will also need to define additional constraints on x and y, but those will be added later
    return model 

end

###### Encode Symbolic Dynamics ########
function encode_sym_dynamics!(symQuery::OvertPQuery)
    """
    Function to encode the dynamics over a trajectory of overapproximations. 
    TODO: Revisit this. Can be done more efficiently by exploiting the overlap between the domains of the overapproximations. This should result in significantly fewer vertices in the final MIP encoding.
    """
    #Define symbolic MIP model
    optimizer = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
    #Define dictionary to store MIP variables
    symQuery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
    symQuery.mod_dict = Dict{Symbol,JuMP.Model}()

    for ind in 1:symQuery.ntime
        #index used to iterate over variables with overapproximations
        i = 0
        for sym in symQuery.problem.varList
            i += 1
            LBs, UBs = symQuery.problem.bounds[ind][i]
            Tri = OA2PWA(LBs)
            xS = [(tup[1:end-1]) for tup in LBs]
            yUB = [tup[end] for tup in UBs]
            yLB = [tup[end] for tup in LBs]

            #If this is the first index, initialize model, otherwise use the model associated with the symbol
            if ind == 1
                symMip = Model(optimizer)
            else
                symMip = symQuery.mod_dict[sym]
            end

            symQuery.mod_dict[sym] = ccSymEncoding(xS, yLB, yUB, Tri, symQuery, ind, sym, symMip)
        end
    end
end

#######Encode Symbolic Controller########
function encode_sym_control(symQuery, reachSets)
    network_file = symQuery.network_file
    #Get dictionary of MIP variables 

    #####Enter loop for time steps#################
    for i = 1:symQuery.ntime
        for sym in symQuery.problem.varList
            sym_mip = symQuery.mod_dict[sym]
            input_set = reachSets[i]
            sym_curr = Meta.parse("$(sym)_$(i)")
            x_curr = symQuery.var_dict[sym_curr][1]
            u_curr = symQuery.var_dict[sym_curr][3]
            y_curr = symQuery.var_dict[sym_curr][2]

            controller_bound = add_controller_constraints!(sym_mip, network_file, input_set, x_curr, u_curr)
        end
    end
end

##########Symbolically link time steps########
function encode_time(symQuery::OvertPQuery)
    """
    Method to encode the time evolution of the system as a MIP. Links distinct overappoximation objects across time steps to complete the symbolic encoding. 
    """
    ##########First loop#############
    for i = 1:symQuery.ntime-1 
        j = 0
        trueInp = Any[]
        trueOut = Any[]
        #First, define the inputs that go into the integration map
        for sym in symQuery.problem.varList 
            j += 1
            #Use metaprogramming to get the current and next symbol
            sym_now = Meta.parse("$(sym)_$(i)")
            sym_next = Meta.parse("$(sym)_$(i+1)")
            #Get the MIP variables associated with the symbols
            y_now = symQuery.var_dict[sym_now][2]
            u_now = symQuery.var_dict[sym_now][3]
            x_now = symQuery.var_dict[sym_now][1]
            x_next = symQuery.var_dict[sym_next][1]

            #Account for different cases 
            if query.case == 1
                push!(trueInp, x_now[j])
                push!(trueOut, y_now)
            else
                push!(trueInp, x_now[j:j+1])
                push!(trueOut, y_now)
            end
        end

        #Define the integration map 
        integration_map = symQuery.problem.update_rule(trueInp, trueOut)
        #Iterate through each MIP model and link current and future time steps
        for sym in symQuery.problem.varList
            #Use metaprogramming to get the current and next symbol
            sym_now = Meta.parse("$(sym)_$(i)")
            sym_next = Meta.parse("$(sym)_$(i+1)")
            #Get the MIP variables associated with the symbols
            y_now = symQuery.var_dict[sym_now][2]
            u_now = symQuery.var_dict[sym_now][3]
            x_now = symQuery.var_dict[sym_now][1]
            x_next = symQuery.var_dict[sym_next][1]

            sym_mip = symQuery.mod_dict[sym]

            #######Second loop#############
            #Iterate through each input variable, find the variable that controls its evolution, then link the input variable of the next time step to the output variable of the current time step
            for k = 1:length(x_now)
                v = x_now[k]
                if v in trueInp
                    dv = integration_map[v][1]
                    next_v = x_next[k]

                    @constraint(sym_mip, next_v == v + symQuery.dt*dv)
                end
            end
        end
    end
end

function symReach(symQuery::OvertPQuery)
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
        encode_sym_control(symQuery)
    end

    #Link the time steps
    encode_time(symQuery)

    #Solve the reachability problem
    t1 = Dates.now()
    reachSet = reach_solve(symQuery, symQuery.ntime)
    t2 = Dates.now()
    println("Time for symbolic reach is ", t2-t1)
    return reachSet
end

function addDim(vec, dim, zeroVal = 1e-12)
    """
    Add a dimension to each tuple in a vector of tuples. This is equivalent to lifting a n-d polytope to a dimention nd+1

    Equiv to a cartesian product with the zero vector in the new dimension
    """
    newVec = Any[]
    for tup in vec
        newTup = Tuple(if i < dim tup[i] elseif i == dim zeroVal else tup[i-1] end for i in 1:length(tup)+1)
        push!(newVec, newTup)
    end
    return newVec
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

function reach_solve(query, t_idx::Union{Nothing,Int64}=nothing)
    """
    Solve contrete reachability problem using MIP. Given a MIP encoding a pwl overapproximation of the dynamics, as well as MIP for the controlller, find the overapproximation of the reachable set

    This version of reach_solve generalizes to multiple functions
    """
    stateVar = query.problem.varList
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
            input_vars = query.var_dict[sym_timed][1]
            output_vars = query.var_dict[sym_timed][2]
            push!(stateVarTimed, sym_timed)
        else   
            input_vars = query.var_dict[sym][1]
            output_vars = query.var_dict[sym][2]
        end
        #Case 1: Multiple functions
        #Here, stateVar = varList. States are (x, y, z, etc). Simply loop over var list and find the appropriate symbol to match to the input and output variables
        #TODO: To this more cleverly
        if query.case == 1
            push!(trueInp, input_vars[i])
            push!(trueOut, output_vars)
        elseif query.case == 2
            #Case 2: Mix of single and multiple functions
            #Here, stateVar != varList. States are (x, dx, y, dy, etc)
            push!(trueInp, input_vars[i:i+1]...)
            i += 1 #increment by 1 to account for the fact that we are skipping over the derivative
            push!(trueOut, output_vars)
        elseif query.case == 3
            #Case 3: Mix of single and multiple functions
            #Here, stateVar != varList. States are (x, dx, ddx, y, dy, ddy etc)
            push!(trueInp, input_vars[i:i+2]...)
            i += 2 #increment by 1 to account for the fact that we are skipping over the derivative
            push!(trueOut, output_vars)
        end
    end

    integration_map = query.problem.update_rule(trueInp, trueOut)
    
    timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)

    #Loop over symbols with OVERT approximations to compute reach steps
    for sym in stateVar
        #Account for symbolic case with timed dynamics
        if !isnothing(t_idx)
            symTimed = Meta.parse("$(sym)_$t_idx")
            input_vars = query.var_dict[symTimed][1]
        else
            input_vars = query.var_dict[sym][1]
        end
        mipModel = query.mod_dict[sym]
        for v in input_vars 
            if v in trueInp
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
        end
    end
    #NOTE: Hyperrectangle can plot in higher dimensions as well
    reacheable_set = Hyperrectangle(low=lows, high=highs)
    return reacheable_set
end

function concreach!(query::OvertPQuery)
    """
    Method to solve the concrete reachability problem using MIP.
    Modifies the query object in place (specifically the bounds generated by OVERT)
    """
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
end

#Find how much to shift each pair of bounds to ensure log is valid 
s_x9p1sp1 = inpShiftLog(lb_x9_p1_sp1, ub_x9_p1_sp1,bounds=x9_p1_sp1_LB)
s_x9p1sp2 = inpShiftLog(lb_x9_p1_sp2, ub_x9_p1_sp2,bounds=x9_p1_sp2_LB)

#Apply log
x9_p1_sp1_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x9p1sp1)) for tup in x9_p1_sp1_LB]
x9_p1_sp1_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x9p1sp1)) for tup in x9_p1_sp1_UB]

x9_p1_sp2_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x9p1sp2)) for tup in x9_p1_sp2_LB]
x9_p1_sp2_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x9p1sp2)) for tup in x9_p1_sp2_UB]

#Lift bounds of sp1 and sp2 to the same space of (x₇, x₈)
emptyList = [2] #Sub part 1 missing x₈
currList = [1]
lbList = [lbs[7], lbs[8]]
ubList = [ubs[7], ubs[8]]

l_x9_p1_sp1_LB_l, l_x9_p1_sp1_UB_l = lift_OA(emptyList, currList, x9_p1_sp1_LB_l,x9_p1_sp1_UB_l, lbList, ubList)

#Lift sub part 2 to space of (x₇, x₈, x₁₁)
emptyList = [1] #Sub part 2 missing x₇ and x₁₁
currList = [2]

l_x9_p1_sp2_LB_l, l_x9_p1_sp2_UB_l = lift_OA(emptyList, currList, x9_p1_sp2_LB_l,x9_p1_sp2_UB_l, lbList, ubList)

#Experimenting here, instead of taking a Minkowski sum, let's just sumBounds. My hypothesis is that MinkSum and sumBounds are equivalent
#So... sum the lifted bounds 

x9_p1_sp4_LB_l, x9_p1_sp4_UB_l = sumBounds(l_x9_p1_sp1_LB_l, l_x9_p1_sp1_UB_l, l_x9_p1_sp2_LB_l, l_x9_p1_sp2_UB_l, true)

if sanityFlag
    validBounds(:(log(sin(x7) + $s_x9p1sp2) - log(cos(x8) + $s_x9p1sp2)), [:x7, :x8], x9_p1_sp4_LB_l, x9_p1_sp4_UB_l, true)
end

#Compute exp to get bounds for (sin(x₇) + s_x9p1sp2)/(cos(x₈) + s_x9p1sp2)
x9_p1_sp4_LB_s = [(tup[1:end-1]..., floor_n(exp(tup[end]))) for tup in x9_p1_sp4_LB_l]
x9_p1_sp4_UB_s = [(tup[1:end-1]..., ceil_n(exp(tup[end]))) for tup in x9_p1_sp4_UB_l]

if sanityFlag 
    validBounds(:((sin(x7)+$s_x9p1sp2)/(cos(x8) + $s_x9p1sp2)), [:x7, :x8], x9_p1_sp4_LB_s, x9_p1_sp4_UB_s, true)
end

#Need to shift the bounds to recover sin(x₇)/cos(x₈)
#f1/f2 = ((f1 + s1)/(f2 + s2) * (f2 + s2) - s1)/((f2 + s2)/)
x9_p1_sp4_LB = []
x9_p1_sp4_UB = []

function prodBounds(lb1, ub1, lb2, ub2)
    #Vector for outputs
    prodLB = []
    prodUB = []

    #Find the union of the inputs 
    #NOTE: Assumes lb and ub have the same inputs 

    bound1Inps = [tup[1:end-1] for tup in lb1]
    bound2Inps = [tup[1:end-1] for tup in lb2]

    #Find the union of the inputs
    unionInps = sort(unique(vcat(bound1Inps, bound2Inps), dims=1))

    #interpolate the bounds to ensure they are defined over the same set of points 
    lb1_i, lb2_i = interpol_nd(lb1, lb2)
    ub1_i, ub2_i = interpol_nd(ub1, ub2)

    #Find the bounds of the product
    for inp in unionInps
        #Find the bounds of the first function
        ind1 = findall(x->x[1:end-1] == inp, lb1_i)[1]
        lb1 = lb1_i[ind1][end]
        ub1 = ub1_i[ind1][end]

        #Find the bounds of the second function
        ind2 = findall(x->x[1:end-1] == inp, lb2_i)[1]
        lb2 = lb2_i[ind2][end]
        ub2 = ub2_i[ind2][end]
        
        #Compute the bounds of the product. Use interval arithmetic
        lb = min(lb1*lb2, lb1*ub2, ub1*lb2, ub1*ub2)
        ub = max(lb1*lb2, lb1*ub2, ub1*lb2, ub1*ub2)
        #Push the bounds to the output list
        push!(prodLB, (inp..., lb))
        push!(prodUB, (inp..., ub))
    end

    return prodLB, prodUB
end