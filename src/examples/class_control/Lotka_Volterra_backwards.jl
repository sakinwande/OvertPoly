include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../reachability.jl")
using LazySets
using Dates
using Logging

Logging.disable_logging(Logging.Warn)
coraMinFile = "matData/min_lotkaVolterra_Cora_100.txt"
coraMaxFile = "matData/max_lotkaVolterra_Cora_100.txt"


#Define update rule for Lotka-Volterra
function lotka_volterra_update_rule(input_vars, 
    overt_output_vars)
    dx = overt_output_vars[1][1]
    dy = overt_output_vars[2][1]
    integration_map = Dict(input_vars[1] => dx, input_vars[2] => dy)
    return integration_map
end

function bound_lv(LotkaVolterra, plotFlag=false, ϵ=1e-10, Nᵢ=2)
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
    #TEST: swap order
    # return [[dxLB, dyLB], [dxUB, dyUB]]
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

nsteps = 5
totTime = 100
dt = 0.008

ϵ = 0.02
target = Hyperrectangle(low=[1.3-ϵ, 1-ϵ/2], high=[1.3 + ϵ,1 + ϵ/2])
domain = Hyperrectangle(low=[-5, -5], high=[5, 5])
# domain = Hyperrectangle(low=[1.27, 0.89], high=[1.33, 1.02])
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

###Compute forward reachable sets 
fquery = deepcopy(query)
fquery.problem.domain = target
simTraj = simulateTraj(fquery, 1000)

xVals = [x[1] for x in simTraj]
yVals = [x[2] for x in simTraj]
reachSets, boundSets = multi_step_concreach(fquery);



############Implementing Backwards Reachability#################
#Bound the function over the full domain

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
function breach_refine_outer(bquery, btarget, stopRatio = 1.1)
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


btarget = reachSets[end]
trueBreach = reachSets[1]
bquery_outer = deepcopy(query)
breachSet_outer, oldBreachSets, bboundSets = breach_refine_outer(bquery_outer, btarget)

bquery_inner = deepcopy(query)
breachSet_inner = breach_refine_inner(bquery_inner, btarget, 0.9)
#breachSets, bboundSets = multi_step_breach(bquery, btarget)
plot(trueBreach, label="true back reach set", legend=:bottomright, title="Backwards Reachable Set For Lotka_Volterra")
plot!(breachSet_outer, label="Minimal back reach set")
plot!(breachSet_inner, label="Maximal back reach set")
trueBreach ⊂ breachSet


bquery = deepcopy(query)

function multi_step_breach(bquery, btarget, outerFlag=true)
    """
    Compute the multi-step backwards reachable set for a given query and target set
    """
    tStart = Dates.now()
    reachSets = []
    boundSets = []
    i =1
    mquery = deepcopy(bquery)
    if outerFlag
        for i = 1:bquery.ntime
            println("Computing backwards reachable set for time step $i")
            breachSet, oldBreachSets, bboundSets = breach_refine_outer(mquery, btarget)
            push!(reachSets, breachSet)
            push!(boundSets, bboundSets)
            btarget = reachSets[end]
            mquery = deepcopy(bquery)
        end
    else
        for i = 1:bquery.ntime
            println("Computing backwards reachable set for time step $i")
            breachSet = breach_refine_inner(mquery, btarget, 0.9)
            push!(reachSets, breachSet)
            btarget = reachSets[end]
            mquery = deepcopy(bquery)
        end
    end
    tStop = Dates.now()
    println("Time to compute multi-step backwards reachable set: $(tStop - tStart)")
    return reachSets, boundSets
end

breachSets, boundSets = multi_step_breach(bquery, btarget, false)

plot([set for set in breachSets][5], fillcolor=:blue, title="Inner BRS vs True BRS for Lotka_Volterra", label="Outer BRS", legend=:bottomright)
plot!([set for set in breachSets][2:end], fillcolor=:blue)
trueSet = deepcopy(reachSets[1:5])
trueSet = reverse(trueSet)
plot!(trueSet[5], fillcolor=:red, label="True BRS")
plot!(trueSet[2:end], fillcolor=:red)

"""
TODO: Implement Minkowski difference (instead of constant scaling)
TODO: Implement optimization routine to find scaling factor 
"""