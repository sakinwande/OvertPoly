include("../src/overapprox_nd.jl")
include("../src/overt_utils.jl")
include("overtPoly_helpers.jl")
using IntervalArithmetic, IntervalRootFinding, Symbolics

#Define problem params
ϵ=1e-1
rel_error_tol=1e-1
lb, ub, npoint = -pi, pi, 1

#Define messy multivariate function
baseFunc = :(cos(x)cos(y)x*y^2 + sin(x)cos(y)y)

v5LB, v5UB = bound_multiariate(baseFunc, lb, ub)


##Traingulation starts here###
#Combine overapprox into 1 vector 
v6 = vcat(v5LB, v5UB)

test = [[tup[1:end-1]...] for tup in v6]
using DelaunayTriangulation
DelaunayTriangulation.triangulate(test)

using MiniQhull
3*416
Mat = delaunay(test)

Mat[1,:]