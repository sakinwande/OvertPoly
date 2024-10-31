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
expr = [:(x2), :(sin(x1) - x2)]
pi2_round = ceil(pi/2, digits=6)
domain = Hyperrectangle(low=[0, 0], high=[pi2_round, pi2_round])
# domain1 = Hyperrectangle(low=[-pi2_round, -pi2_round], high=[0, 0])
# domain2 = Hyperrectangle(low=[0, 0], high=[pi2_round, pi2_round])
npoint=2
#Bounding the pendulum. Break into smaller chunks
# function bound_pend(npoint) 
    lbs, ubs = extrema(domain)
    #Part 1: Bound the sin(x1) term
    p1 = :(sin(x))
    p1_LB_1_1, p1_UB_1_1 = interpol(bound_univariate(p1, lbs[1], ubs[1],npoint=npoint)...)
    # p1_LB_1_2, p1_UB_1_2 = interpol(bound_univariate(p1, lbs2[1], ubs2[1],npoint=npoint)...)

    #Part 2: Bound the x2 term
    p2 = :(1*x)
    p2_LB_1_1, p2_UB_1_1 = interpol(bound_univariate(p2, lbs[2], ubs[2],npoint=npoint)...)
    # p2_LB_1_2, p2_UB_1_2 = interpol(bound_univariate(p2, lbs2[2], ubs2[2],npoint=npoint)...)


    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first 
    #Part 1 missing x2
    l_p1_LB_11, l_p1_UB_11 = lift_OA([2], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs) 
    #l_p1_LB_12, l_p1_UB_12 = lift_OA([2], [1], p1_LB_1_2, p1_UB_1_2, lbs2, ubs2)
    # # l_p1_LB_13, l_p1_UB_13 = lift_OA([2], [1], p1_LB_1_1, p1_UB_1_1, lbs2, ubs2)
    # # l_p1_LB_14, l_p1_UB_14 = lift_OA([2], [1], p1_LB_1_2, p1_UB_1_2, lbs1, ubs1)

    # l_p1_LB_1 = sort(unique(vcat(l_p1_LB_11, l_p1_LB_12, l_p1_LB_13, l_p1_LB_14); dims =1))
    # l_p1_UB_1 = sort(unique(vcat(l_p1_UB_11, l_p1_UB_12, l_p1_UB_13, l_p1_UB_14); dims =1))

    #Lift part 2
    #Part 2 missing x1
    l_p2_LB_11, l_p2_UB_11 = lift_OA([1], [2], p2_LB_1_1, p2_UB_1_1, lbs, ubs)
    # l_p2_LB_12, l_p2_UB_12 = lift_OA([1], [2], p2_LB_1_2, p2_UB_1_2, lbs2, ubs2)
    # l_p2_LB_13, l_p2_UB_13 = lift_OA([1], [2], p2_LB_1_1, p2_UB_1_1, lbs2, ubs2)
    # l_p2_LB_14, l_p2_UB_14 = lift_OA([1], [2], p2_LB_1_2, p2_UB_1_2, lbs1, ubs1)

    # l_p2_LB_1 = sort(unique(vcat(l_p2_LB_11, l_p2_LB_12, l_p2_LB_13, l_p2_LB_14); dims =1))
    # l_p2_UB_1 = sort(unique(vcat(l_p2_UB_11, l_p2_UB_12, l_p2_UB_13, l_p2_UB_14); dims =1))


    #Sum the bounds to recover 2d bounds 
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_11, l_p2_UB_11, true)

    # LB_1
    # #Plot surface from this 
    # baseFunc = :(sin(x1) - x2)
    # xS = unique([tup[1] for tup in LB_1])
    # yS = unique([tup[2] for tup in LB_1])
    # surfDim = (length(yS), length(xS))

    plotSurf(baseFunc, LB_1, UB_1, surfDim, xS, yS, true)

    LB_1_inps = [tup[1:end-1] for tup in LB_1]
    LB_1_Tri = OA2PWA(LB_1)

    #Write to file 
    open("pend_n$(npoint)_bounds.txt", "w") do file 
        for tup in LB_1_inps
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end

    open("pend_n$(npoint)_tri.txt", "w") do file 
        for tup in LB_1_Tri
            write(file, string(tup[1], ",", tup[2], ",", tup[3], "\n"))
        end
    end

    return validBounds(:(sin(x1) - x2), [:x1, :x2],LB_1, UB_1, true)
# end

#n = 2
bound_pend(2)
bound_pend(5)
bound_pend(10)
bound_pend(50)
bound_pend(100)
bound_pend(200)
bound_pend(500)
bound_pend(1000)