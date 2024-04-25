include("overtPoly_helpers.jl")
include("nn_mip_encoding.jl")
include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")
include("problems.jl")
include("reachability.jl")
using LazySets
using Dates

#Define parameters 
μ = 1

#Define nonlinear expressions 
expr1 = :($(μ) * (1 - x₁^2)*y₁ + b * (x₂ - x₁) - x₁)
expr2 = :($(μ) * (1 - x₂^2)*y₂ + b * (x₁ - x₂) - x₂)
#NOTE: y₁ ≡ ̇x₁ and y₂ ≡ ̇x₂

expr = [expr1, expr2]

function cvdp_update_rule(input_vars, overt_output_vars)
    dy₁ = overt_output_vars[1]
    dy₂ = overt_output_vars[2]
    integration_map = Dict(input_vars[1] => input_vars[2], input_vars[2] => dy₁, input_vars[3] => input_vars[4], input_vars[4] => dy₂, input_vars[5] => input_vars[5])
    return integration_map
end

nsteps = 10
dt = 0.01


CVDP = OvertPProblem(
    expr, #List of nonlinear equations 
    nothing, #No decomposed dynamics. Done manually here
    0, #No control coefficients
    Hyperrectangle(low=[1.25, 2.35, 1.25, 2.35, 1], high=[1.55, 2.45, 1.55, 2.45, 3]), #input domain 
    [:y₁, :y₂], #List of variables with OVERT bounds
    nothing, #No bounds to start
    cvdp_update_rule #Update rule for the system
)

############Implementing bound_cvdp############
lbs, ubs = extrema(CVDP.domain)

baseFunc1 = :($(μ) * (1 - x₁^2)*y₁)
lbs1, ubs1 = lbs[1:2], ubs[1:2]

#Decompose by hand. f(x)*f(y) = log(f(x)) + log(f(y))
bF1sub1 = :(1- x₁^2)
bF1sub2 = :($(μ)*y₁)

#Define bounds for the first expression
bF1s1LB, bF1s1UB = bound_univariate(bF1sub1, lbs1[1], ubs1[1], plotflag=true)
bF1s2LB, bF1s2UB = bound_univariate(bF1sub2, lbs1[2], ubs1[2], plotflag=true)

bF1s1LB, bF1s1UB = interpol(bF1s1LB, bF1s1UB)
bF1s2LB, bF1s2UB = interpol(bF1s2LB, bF1s2UB)

#Convert to log in preparation for Minkowski sum
sₓ₁ = inpShiftLog(lbs1[1], ubs1[1]; bounds=bF1s1LB)
sᵧ₁ = inpShiftLog(lbs1[2], ubs1[2]; bounds=bF1s2LB)

lbF1s1LB = [(tup[1:end-1]..., log(tup[end] + sₓ₁)) for tup in bF1s1LB]
lbF1s1UB = [(tup[1:end-1]..., log(tup[end] + sₓ₁)) for tup in bF1s1UB]

lbF1s2LB = [(tup[1:end-1]..., log(tup[end] + sᵧ₁)) for tup in bF1s2LB]
lbF1s2UB = [(tup[1:end-1]..., log(tup[end] + sᵧ₁)) for tup in bF1s2UB]

#Add a dimension to prepare for Minkowski sum
lbF1s1LB_l = addDim(lbF1s1LB, 2)
lbF1s1UB_l = addDim(lbF1s1UB, 2)

lbF1s2LB_l = addDim(lbF1s2LB, 1)
lbF1s2UB_l = addDim(lbF1s2UB, 1)

#Combine to get log(f(x)) + log(f(y))
lbF1LB = MinkSum(lbF1s1LB_l, lbF1s2LB_l)
lbF1UB = MinkSum(lbF1s1UB_l, lbF1s2UB_l)

#Compute the exp to get f(x)*f(y)
bF1LB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in lbF1LB]
bF1UB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in lbF1UB]

#Account for the shift
bF1LB = Any[]
bF1UB = Any[]

#Shift down by sₓ₁ and sᵧ₁
for tup in bF1LB_s
    #First find the corresponding f(x) and f(y) values
    xInd = findall(x->x[1] == tup[1], bF1s1LB)
    yInd = findall(y->y[1] == tup[2], bF1s2LB)

    #Quadratic shift down
    newXY = tup[end] - sᵧ₁ * bF1s1LB[xInd][1][1] - sₓ₁ * bF1s2LB[yInd][1][1] - sₓ₁*sᵧ₁

    push!(bF1LB, (tup[1:end-1]..., newXY))
end

for tup in bF1UB_s
    #First find the corresponding f(x) and f(y) values
    xInd = findall(x->x[1] == tup[1], bF1s1LB)
    yInd = findall(y->y[1] == tup[2], bF1s2LB)

    #Quadratic shift down
    newXY = tup[end] - sᵧ₁ * bF1s1LB[xInd][1][1] - sₓ₁ * bF1s2LB[yInd][1][1] - sₓ₁*sᵧ₁

    push!(bF1UB, (tup[1:end-1]..., newXY))
end

plotFlag = false
####Plot the surface
if plotFlag
    xS = Any[tup[1] for tup in bF1s1LB]
    yS = Any[tup[2] for tup in bF1s2LB]

    surfDim = (size(yS)[1], size(xS)[1])

    plotSurf(baseFunc1, bF1LB, bF1UB, surfDim, xS, yS, true)
end
###############################################

query = OvertPQuery(
    CVDP, #Problem to solve
    bound_cvdp, #Bound function
    nothing, #No network file
    nothing, #No last layer activation
    "MIP", #Solver
    nsteps, #Number of time steps
    dt, #Time step size
    2, #Number of overapproximation points
    nothing, #Variable dictionary
    nothing, #Model dictionary
    2 #CVDP has form (x, dx, y, dy, z)
)


