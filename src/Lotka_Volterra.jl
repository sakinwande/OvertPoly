include("overtPoly_helpers.jl")
include("nn_mip_encoding.jl")
include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")
include("problems.jl")
include("reachability.jl")
using LazySets
using Dates
import Logging

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

# #Define equations of motion 
xExpr = :(3*x - 3*x*y)
yExpr = :(x*y - y)
expr = [xExpr, yExpr]

totalsteps = 450
nsteps = 50
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
    lotka_volterra_update_rule #Update rule for the system
)

#Define OVERT query
query = OvertPQuery(
    LotkaVolterra, #Problem to solve
    bound_lv, #Bound function
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

# reachSets, boundSets = multi_step_concreach(query)

# # Plot the results
# # plot(reachSets, title="Lotka_Volterra_Concrete_$(nsteps)")

# # Define symbolic problem
# symLotkaVolterra = OvertPProblem(
#     expr, #list of equations 
#     nothing, #Decomposed form of dynamics. Done manually
#     0, #Control coefficients. Not used in this case
#     Hyperrectangle(low=[1.28, 0.99], high=[1.312,1.01]), #Domain of the problem
#     [:x, :y], #List of variables
#     boundSets, #Bounds from concrete problem
#     lotka_volterra_update_rule #Update rule for the system
# )
# #Define symbolic query
# symQuery = OvertPQuery(
#     symLotkaVolterra, #Problem to solve
#     bound_lv, #Bound function
#     nothing, #No controller file
#     nothing, #Last layer activation
#     "MIP", #Solver
#     nsteps, #Number of time steps
#     dt, #Time step size
#     2, #Number of overapproximation points
#     nothing, #Variable dictionary
#     nothing, #Model dictionary
#     1 #Case of variables
# )

# # reach_set = symReach(symQuery)
# # plot(reach_set, title="Symbolic_Reachable_Set_t=$(nsteps)", label="Sym Reach Set")
# # plot!(reachSets[end], label="Conc Reach Set")

# symReachSets = multi_step_symreach(symQuery)
# symReachSets[end]

# plot(reachSets, title="Comparing_LV_Concrete_and_Symbolic_$(nsteps)", fillcolor=:blue)
# plot!(symReachSets, fillcolor=:red)


totalReachSets = [domain]
symReachSets = []
numConc = ceil(totalsteps/nsteps)
totalBoundSets = []
totalConcReachSets = [domain]

######Defining hybrid symbolic loop######
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

# totalReachSets
# plot(totalReachSets)

# extrema(totalReachSets[456])

a = area(totalReachSets[end])
extrema(totalReachSets[end])

b = (1.33815 - 1.29095) * (1.0109 - 0.985862)

a/b

totalBoundSets
totalReachSets
##############Try to do straight shot with hybrid-symbolic bounds##############
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

#b = (1.23936 - 1.18530) * (1.13026 - 1.09866)
#b = (0.861708 - 0.816375) * (1.14685 - 1.11876)
#b = (0.779331 - 0.742348) * (0.953271 - 0.929644)
b = (1.03734 - 1.00236) * (0.855737 - 0.835147)

b/a

plot(reach_set)

######################Simulate random trajectories
using Random
ntraj = 1000
traj = []
for i = 1:ntraj
    set = Hyperrectangle(low=[1.3-ϵ, 1-ϵ/2], high=[1.3 + ϵ,1 + ϵ/2])
    x0 = [0,0]
    x0[1] = extrema(set)[1][1] + rand()*(extrema(set)[2][1] - extrema(set)[1][1])
    x0[2] = extrema(set)[1][2] + rand()*(extrema(set)[2][2] - extrema(set)[1][2])
    for j = 1:totalsteps
        xNew = [x0[1] + dt*(3*x0[1] - 3*x0[1]*x0[2]), x0[2] + dt*(x0[1]*x0[2] - x0[2])]
        x0 = xNew
    end
    push!(traj, x0)
end


xVals = [x[1] for x in traj]
yVals = [x[2] for x in traj]
plot(xVals, yVals)

minimum(xVals)
maximum(xVals)

minimum(yVals)
maximum(yVals)

set = Hyperrectangle(low=[1.3-ϵ, 1-ϵ/2], high=[1.3 + ϵ,1 + ϵ/2])
x0 = rand(extrema(set))
extrema(set)[1][1]

rand((2,2.1))

randn()
