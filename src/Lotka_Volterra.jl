include("overtPoly_helpers.jl")
include("nn_mip_encoding.jl")
include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")
include("problems.jl")
include("reachability.jl")
using LazySets
using Dates
using Logging

Logging.disable_logging(Logging.Warn)

#Define update rule for Lotka-Volterra
function lotka_volterra_update_rule(input_vars, 
    overt_output_vars)
    dx = overt_output_vars[1]
    dy = overt_output_vars[2]
    integration_map = Dict(input_vars[1] => dx, input_vars[2] => dy)
    return integration_map
end

function bound_lv(LotkaVolterra, ϵ=0.001, Nᵢ=2, plotFlag=false)
    """
    Method to compute bounds for the Lotka Volterra Problem

    Here we observe that the main nonlinearity is x*y. We focus on overapproximating this nonlinearity via exp-log decomposition

    x*y = exp(log(x) + log(y)). Approximate log(x) and log(y) separately and then combine to get x*y

    Takes optional parameters ϵ and Nᵢ to control the buffer to add to the bounds and the number of points to interpolate over. Additional flag to determine if we want to plot the bounds
    """

    #Get input bounds 
    lbs, ubs = extrema(LotkaVolterra.domain)
    baseFunc1 = :(log(x))
    xDec2f1 = Symbolics.build_function(baseFunc1, find_variables(baseFunc1)...,expression=Val{false})
    lbx,ubx = lbs[1], ubs[1]

    #Compute extrema of the nonlinear term
    sX = inpShiftLog(lbx, ubx) #Shift value to account for negative values
    baseFunc1UB = [(lbx, xDec2f1(lbx + sX)), (ubx, xDec2f1(ubx + sX))]
    baseFunc1LB = [(lbx, xDec2f1(lbx + sX)), (ubx, xDec2f1(ubx + sX))]

    baseFunc2 = :(log(y))
    xDec2f2 = Symbolics.build_function(baseFunc2, find_variables(baseFunc2)...,expression=Val{false})
    lby,uby = lbs[2], ubs[2]

    sY = inpShiftLog(lby, uby) #Shift value to account for negative values
    baseFunc2UB = [(lby, xDec2f2(lby + sY)), (uby, xDec2f2(uby + sY))]
    baseFunc2LB = [(lby, xDec2f2(lby + sY)), (uby, xDec2f2(uby + sY))]

    #Add a dimension to prepare for Minkowski sum
    #Add empty y-dimension to log(x)
    baseFunc1U = addDim(baseFunc1UB, 2)
    baseFunc1L = addDim(baseFunc1LB, 2)
    #Add empty x-dimension to log(y)
    baseFunc2U = addDim(baseFunc2UB, 1)
    baseFunc2L = addDim(baseFunc2LB, 1)

    #Combine to get log(x*y)
    lxyU = MinkSum(baseFunc1U, baseFunc2U)
    lxyL = MinkSum(baseFunc1L, baseFunc2L)

    #Compute the exp to get x*y
    xyUs = [(tup[1:end-1]..., exp(tup[end])) for tup in lxyU]
    xyLs = [(tup[1:end-1]..., exp(tup[end])) for tup in lxyL]

    #Account for the shift 
    xyU = Any[]
    xyL = Any[]

    #Shift down by sX and sY
    for tup in xyUs
        #First find the corresponding f(x) and f(y) values
        xInd = findall(x->x[1] == tup[1], baseFunc1LB)
        yInd = findall(y->y[1] == tup[2], baseFunc2LB)

        #Quadratic shift down 
        newXY = tup[end] - sY * baseFunc1LB[xInd][1][1] - sX * baseFunc2LB[yInd][1][1] - sX*sY

        push!(xyU, (tup[1:end-1]..., newXY))
    end

    for tup in xyLs
        #First find the corresponding f(x) and f(y) values
        xInd = findall(x->x[1] == tup[1], baseFunc1LB)
        yInd = findall(y->y[1] == tup[2], baseFunc2LB)

        #Quadratic shift down 
        newXY = tup[end] - sY * baseFunc1LB[xInd][1][1] - sX * baseFunc2LB[yInd][1][1] - sX*sY

        push!(xyL, (tup[1:end-1]..., newXY))
    end


    #Define bounds for f(x) = x
    xLinUB = [(tup[1:end-1]..., tup[1]) for tup in xyU]
    xLinLB = [(tup[1:end-1]..., tup[1]) for tup in xyL]

    xFuncLB = Any[]
    xFuncUB = Any[]

    #Compute 3*x - 3*x*y for upper and lower bounds
    for i = 1:size(xyL)[1]
        newtup = (xyL[i][1:end-1]..., 3*-xyL[i][end] + 3*xLinLB[i][end])
        push!(xFuncLB, newtup)
    end

    for i = 1:size(xyU)[1]
        newtup = (xyU[i][1:end-1]..., 3*-xyU[i][end] + 3*xLinUB[i][end])
        push!(xFuncUB, newtup)
    end

    #Similary, compute bounds for second function 

    #Define bounds for f(y) = y
    yLinUB = [(tup[1:end-1]..., tup[2]) for tup in xyU]
    yLinLB = [(tup[1:end-1]..., tup[2]) for tup in xyL]

    yFuncLB = Any[]
    yFuncUB = Any[]

    #Compute x*y - y for upper and lower bounds
    for i = 1:size(xyL)[1]
        newtup = (xyL[i][1:end-1]..., xyL[i][end] - yLinLB[i][end])
        push!(yFuncLB, newtup)
    end

    for i = 1:size(xyU)[1]
        newtup = (xyU[i][1:end-1]..., xyL[i][end] - yLinUB[i][end])
        push!(yFuncUB, newtup)
    end

    ###Interpolate between extrema to get piecewise affine approx over full domain 
    xLBInt = gen_interpol(xFuncLB)
    xUBInt = gen_interpol(xFuncUB)
    yLBInt = gen_interpol(yFuncLB)
    yUBInt = gen_interpol(yFuncUB)

    ##Define domain to interpolate over 
    xRange = range(lbs[1], stop=ubs[1], length=Nᵢ)
    yRange = range(lbs[2], stop=ubs[2], length=Nᵢ)

    #Interpolate over domain
    dxUB = Any[]
    dxLB = Any[]
    dyUB = Any[]
    dyLB = Any[]

    #NOTE: that the bounds are exact beforehand. Aftewards, we add a small buffer to ensure that the bounds are conservative
    ϵ = 0.001
    for (x,y) in Iterators.product(xRange, yRange)
        push!(dxLB, (x,y,xLBInt(x,y)-ϵ))
        push!(dxUB, (x,y,xUBInt(x,y)+ϵ))
        push!(dyLB, (x,y,yLBInt(x,y)-ϵ))
        push!(dyUB, (x,y,yUBInt(x,y)+ϵ))
    end

    #Plot to compare to real functions
    if plotFlag
        xS = Any[tup[1] for tup in baseFunc1LB]
        yS = Any[tup[1] for tup in baseFunc2LB]

        surfDim = (size(yRange)[1], size(xRange)[1])

        plotSurf(xExpr, sort(dxLB), sort(dxUB), surfDim, xRange, yRange, true)
        plotSurf(yExpr, sort(dyLB), sort(dyUB), surfDim, xRange, yRange, true)
    end
    return [[dxLB, dxUB], [dyLB, dyUB]]
end

function lotka_volterra_dynamics(x,dt)
    "Method to compute a single time step of the Lotka-Volterra dynamics"
    dx = 3*x[1] - 3*x[1]*x[2]
    dy = x[1]*x[2] - x[2]

    xNew = [x[1] + dt*dx, x[2] + dt*dy]
    return xNew
end

# #Define equations of motion 
xExpr = :(3*x - 3*x*y)
yExpr = :(x*y - y)
expr = [xExpr, yExpr]

nsteps = 100
dt = 0.008

ϵ = 0.02
domain = Hyperrectangle(low=[1.3-ϵ, 1-ϵ/2], high=[1.3 + ϵ,1 + ϵ/2])

LotkaVolterra = OvertPProblem(
    expr, #list of equations 
    nothing, #Decomposed form of dynamics. Done manually
    0, #Control coefficients. Not used in this case
    domain, #Domain of the problem
    [:x, :y], #List of variables with OVERT bounds
    nothing, #undefined bounds to start 
    lotka_volterra_update_rule, #Update rule for the system
    lotka_volterra_dynamics, #Dynamics function
    bound_lv #Bound function
)

#Define OVERT query
query = OvertPQuery(
    LotkaVolterra, #Problem to solve
    nothing, #Network file
    nothing, #Last layer activation
    "MIP", #Solver
    nsteps, #Number of time steps
    dt, #Time step size
    2, #Number of overapproximation points
    nothing, #Variable dictionary
    nothing, #Model dictionary
    1 #Case of variables
)

bound_lv(LotkaVolterra, true)
#Simulate first because multi_step_concreach updates starting domain
simTraj = simulateTraj(query,1000)
xVals = [x[1] for x in simTraj]
yVals = [x[2] for x in simTraj]

#perform multi-step concrete reachability
reachSets, boundSets = multi_step_concreach(query)

# Plot the results and compare to simulated trajectories 


plot(reachSets[end], title="Lotka_Volterra_Concrete_$(nsteps)", label="Concrete Reach Set")
scatter!(xVals, yVals, label="Simulated Trajectory")
# Define symbolic problem
symLotkaVolterra = OvertPProblem(
    expr, #list of equations 
    nothing, #Decomposed form of dynamics. Done manually
    0, #Control coefficients. Not used in this case
    domain, #Domain of the problem
    [:x, :y], #List of variables
    boundSets, #Bounds from concrete problem
    lotka_volterra_update_rule, #Update rule for the system
    lotka_volterra_dynamics, #Dynamics function
    bound_lv #Bound function
)
#Define symbolic query
symQuery = OvertPQuery(
    symLotkaVolterra, #Problem to solve
    nothing, #No controller file
    nothing, #Last layer activation
    "MIP", #Solver
    nsteps, #Number of time steps
    dt, #Time step size
    2, #Number of overapproximation points
    nothing, #Variable dictionary
    nothing, #Model dictionary
    1 #Case of variables
)

reach_set = symReach(symQuery)
plot!(reach_set, label="Sym Reach Set")
# plot!(reachSets[end], label="Conc Reach Set")

# symReachSets = multi_step_symreach(symQuery)
# symReachSets[end]

# # plot(reachSets, title="Comparing_LV_Concrete_and_Symbolic_$(nsteps)", fillcolor=:blue)
# plot!(symReachSets[end], label="Symbolic Reach Set")

#####################Debugging Symbolic Reach##########################
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
        time_curr = Meta.parse("$t_$(i)")
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
            dv = integration_map[v][1]
            next_v = x_next[k]
            @constraint(sym_mip, next_v == v + symQuery.dt*dv)
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
        dv = integration_map[v][1]
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
    Method to solve the concrete reachability problem using MIP for multiple time steps.
    """
    input_set = symQuery.problem.domain
    reachSets = []
    totTime = copy(symQuery.ntime)
    t1 = Dates.now()
    for i = 1:totTime
        symQuery.ntime = i
        reachSet = symReach(symQuery);
        push!(reachSets, reachSet)
    end
    t2 = Dates.now()
    println("Time for symbolic reachability is ", t2-t1)
    return reachSets
end

##############################################################################
######Defining hybrid symbolic loop######
totalReachSets = [domain]
symReachSets = []
numConc = ceil(totalsteps/nsteps)
totalBoundSets = []
totalConcReachSets = [domain]


# tStart = Dates.now()
cumSteps = 0
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
    push!(totalConcReachSets, reachSets...)
    push!(totalBoundSets, boundSets...)
    symLotkaVolterra = OvertPProblem(
        expr, #list of equations 
        nothing, #Decomposed form of dynamics. Done manually
        0, #Control coefficients. Not used in this case
        newDomain, #Domain of the problem
        [:x, :y], #List of variables
        boundSets, #Bounds from concrete problem
        lotka_volterra_update_rule #Update rule for the system
    )
    symQuery = deepcopy(query)
    symQuery.problem = symLotkaVolterra
    if cumSteps + symQuery.ntime > totalsteps
        symQuery.ntime = totalsteps - cumSteps
    end
    symReachSets = multi_step_symreach(symQuery)
    cumSteps += symQuery.ntime
    push!(totalReachSets, symReachSets...)
    
end

# tEnd = Dates.now()
# print("Time to compute $(totalsteps) steps:", tEnd-tStart)
plot(totalReachSets, title="Hybrid_LV_Concrete_and_Symbolic_$(cumSteps)", fillcolor=:blue)

totalReachSets
cumSteps
totalBoundSets

#######Try to do straight shot with hybrid-symbolic bounds##############
# Define symbolic problem
symBoundSets = Any[]
for set in totalReachSets
    itQuery = deepcopy(query)
    itQuery.problem.domain = set
    itBound = itQuery.bound_func(itQuery.problem)
    push!(symBoundSets, itBound)
end

symLotkaVolterra = OvertPProblem(
    expr, #list of equations 
    nothing, #Decomposed form of dynamics. Done manually
    0, #Control coefficients. Not used in this case
    Hyperrectangle(low=[1.28, 0.99], high=[1.312,1.01]), #Domain of the problem
    [:x, :y], #List of variables
    symBoundSets, #Bounds from concrete problem
    lotka_volterra_update_rule #Update rule for the system
)
#Define symbolic query
symQuery = OvertPQuery(
    symLotkaVolterra, #Problem to solve
    bound_lv, #Bound function
    nothing, #No controller file
    nothing, #Last layer activation
    "MIP", #Solver
    totalsteps, #Number of time steps
    dt, #Time step size
    2, #Number of overapproximation points
    nothing, #Variable dictionary
    nothing, #Model dictionary
    1 #Case of variables
)
reach_set = symReach(symQuery)
extrema(reach_set)
a = area(reach_set)




