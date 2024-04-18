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

#Decompose by hand
bF1sub1 = :(1- x₁^2)
bF1sub2 = :($(μ)*y₁)

#Define bounds for the first expression
bF1s1LB, bF1s1UB = bound_univariate(bF1sub1, lbs1[1], ubs1[1], plotflag=true)
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

#############Implementing bound univariate#################




UBPoints