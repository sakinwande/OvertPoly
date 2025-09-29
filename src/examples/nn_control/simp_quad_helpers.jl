include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

function bound_quadx1(QUAD, plotFlag = false, sanityFlag = true, npoint=1)
    """
    Function to bound ̇x₁ = cos(x₅)*cos(x₆) + sin(x₄)*sin(x₅)*cos(x₆) - cos(x₄)*sin(x₆) + cos(x₄)*sin(x₅)*cos(x₆) + sin(x₄)*sin(x₆)

    The need for a separate bounding function is self evident 

    Args:
        QUAD: Quadrotor dynamics
        plotFlag: Flag to plot the bounds
        sanityFlag: Flag to check the validity of the bounds
    """
    lbs, ubs = extrema(Quad.domain)

    #Follow a similar strategy to the x1 bounds. Break the initial function into 7 parts then combine parts to regain full bounds 

    #Part 1: cos(x₅)*cos(x₆)
    
    
    return x2_p12_LB_l, x2_p12_UB_l
end