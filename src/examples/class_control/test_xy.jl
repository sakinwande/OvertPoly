include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

#Define problem parameters
expr = [:(x*y)]
domain = Hyperrectangle(low=[0, 0], high=[1, 1])
# domain1 = Hyperrectangle(low=[-pi2_round, -pi2_round], high=[0, 0])
# domain2 = Hyperrectangle(low=[0, 0], high=[pi2_round, pi2_round])
npoint=1
#Bounding the pendulum. Break into smaller chunks
function bound_xy_ia(npoint) 
    lbs, ubs = extrema(domain)
    p1 = :(x*cos(x))
    p1_LB_1_1, p1_UB_1_1 = interpol_nd(bound_univariate(p1, lbs[1], ubs[1],npoint=npoint)...)
    p2 = :(y*sin(y))
    p2_LB_1_1, p2_UB_1_1 = interpol_nd(bound_univariate(p2, lbs[2], ubs[2],npoint=npoint)...)

    l_p1_LB_11, l_p1_UB_11 = lift_OA([2], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs) 
    l_p2_LB_11, l_p2_UB_11 = lift_OA([1], [2], p2_LB_1_1, p2_UB_1_1, lbs, ubs)
    return l_p1_LB_11, l_p1_UB_11, l_p2_LB_11, l_p2_UB_11
end

lb1, ub1, lb2, ub2 = bound_xy_ia(1)

