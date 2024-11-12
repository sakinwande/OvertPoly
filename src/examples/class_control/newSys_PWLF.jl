include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

expr = [:(x2 - sin(x1)), :(-x2 - 0.3*x1 - (0.5^x1)^3)]
domain = Hyperrectangle(low=[-1, -1], high=[1, 1])

npoint = 2

#function bound_func(npoint)
    lbs, ubs = extrema(domain)
    #Part 1: Bound the x2 - sin(x1) term
    p1 = :(1*x)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[2], ubs[2], npoint=npoint)

    p2 = :(-sin(x))
    p1_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[1], ubs[1], npoint=npoint)