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
    Function to bound ̇x₁ = cos(x₈)*cos(x₉)*x₄ + (sin(x₇)*sin(x₈)*cos(x₉) - cos(x₇)*sin(x₉))*x₅ + (cos(x₇)*sin(x₈)*cos(x₉) + sin(x₇)*sin(x₉))*x₆

    The need for a separate bounding function is self evident 

    Args:
        QUAD: Quadrotor dynamics
        plotFlag: Flag to plot the bounds
        sanityFlag: Flag to check the validity of the bounds
    """
    lbs, ubs = extrema(Quad.domain)

    #Follow a similar strategy to the x1 bounds. Break the initial function into 7 parts then combine parts to regain full bounds 

    #Part 1: f₁(x₄, x₈, x₉) = x₄*cos(x₈)*cos(x₉)
    #K-A decomposition is exp(log(x₄) + log(cos(x₈)) + log(cos(x₉)))

    #Sub-part 1 = x₄
    x2_p1_sp1 = :(1*x)
    lb_x2_p1_sp1 = lbs[4]
    ub_x2_p1_sp1 = ubs[4]

    x2_p1_sp1_LB, x2_p1_sp1_UB = interpol_nd(bound_univariate(x2_p1_sp1, lb_x2_p1_sp1, ub_x2_p1_sp1, npoint=npoint)...)

    #Sub-part 2 = cos(x₈)
    x2_p1_sp2 = :(cos(x))
    lb_x2_p1_sp2 = lbs[8]
    ub_x2_p1_sp2 = ubs[8]

    #Bounding trig functions over very thin intervals is tricky. Widen for now
    if ub_x2_p1_sp2 - lb_x2_p1_sp2 < 1e-5
        lb_x2_p1_sp2 = lb_x2_p1_sp2 - 1e-5
        ub_x2_p1_sp2 = ub_x2_p1_sp2 + 1e-5
        lbs[8] = lb_x2_p1_sp2
        ubs[8] = ub_x2_p1_sp2
    end

    x2_p1_sp2_LB, x2_p1_sp2_UB = interpol_nd(bound_univariate(x2_p1_sp2, lb_x2_p1_sp2, ub_x2_p1_sp2,npoint=npoint)...)

    #Sub-part 3 = cos(x₉)
    x2_p1_sp3 = :(cos(x))
    lb_x2_p1_sp3 = lbs[9]
    ub_x2_p1_sp3 = ubs[9]

    #Bounding trig functions over very thin intervals is tricky. Widen for now
    if ub_x2_p1_sp3 - lb_x2_p1_sp3 < 1e-5
        lb_x2_p1_sp3 = lb_x2_p1_sp3 - 1e-5
        ub_x2_p1_sp3 = ub_x2_p1_sp3 + 1e-5
        lbs[9] = lb_x2_p1_sp3
        ubs[9] = ub_x2_p1_sp3
    end

    x2_p1_sp3_LB, x2_p1_sp3_UB = interpol_nd(bound_univariate(x2_p1_sp3, lb_x2_p1_sp3, ub_x2_p1_sp3, npoint=npoint)...)

    #Avoid Minksum and just directly evaluate the products of the bounds
    #To take the products directly, lift each bound to the same set of dimensions, then multiply. Equiv to exp-log trick

    #So.. first lift sub part 1 to space of (x₄, x₈, x₉)
    emptyList = [2,3] #Since x₈, x₉ come after x₄
    currList = [1]
    lbList = lbs[8:9]
    pushfirst!(lbList, lbs[4])
    ubList = ubs[8:9]
    pushfirst!(ubList, ubs[4])

    l_x2_p1_sp1_LB, l_x2_p1_sp1_UB = lift_OA(emptyList, currList, x2_p1_sp1_LB, x2_p1_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₄, x₈, x₉)
    emptyList = [1,3] #Since x₄, x₉ come after x₈
    currList = [2]

    l_x2_p1_sp2_LB, l_x2_p1_sp2_UB = lift_OA(emptyList, currList, x2_p1_sp2_LB, x2_p1_sp2_UB, lbList, ubList)

    #Now lift sub part 3 to space of (x₄, x₈, x₉)
    emptyList = [1,2] #Since x₄, x₈ come after x₉
    currList = [3]

    l_x2_p1_sp3_LB, l_x2_p1_sp3_UB = lift_OA(emptyList, currList, x2_p1_sp3_LB, x2_p1_sp3_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p1_LB_i, x2_p1_UB_i = prodBounds(l_x2_p1_sp1_LB, l_x2_p1_sp1_UB, l_x2_p1_sp2_LB, l_x2_p1_sp2_UB)
    x2_p1_LB, x2_p1_UB = prodBounds(x2_p1_LB_i, x2_p1_UB_i, l_x2_p1_sp3_LB, l_x2_p1_sp3_UB)

    # if sanityFlag
    #     validBounds(:(x4*cos(x8)*cos(x9)), [:x4, :x8, :x9], x2_p1_LB, x2_p1_UB)
    # end
    #Part 2: f₂(x₈, x₉) = sin(x₈)*cos(x₉)
    #Sub part 1 = sin(x₈)
    x2_p2_sp1 = :(sin(x))
    lb_x2_p2_sp1 = lbs[8]
    ub_x2_p2_sp1 = ubs[8]
    
    if ub_x2_p2_sp1 - lb_x2_p2_sp1 < 1e-5
        lb_x2_p2_sp1 = lb_x2_p2_sp1 - 1e-5
        ub_x2_p2_sp1 = ub_x2_p2_sp1 + 1e-5
        lbs[8] = lb_x2_p2_sp1
        ubs[8] = ub_x2_p2_sp1
    end

    x2_p2_sp1_LB, x2_p2_sp1_UB = interpol_nd(bound_univariate(x2_p2_sp1, lb_x2_p2_sp1, ub_x2_p2_sp1, npoint=npoint)...)

    #Sub part 2 = sin(x₉)
    x2_p2_sp2 = :(cos(x))
    lb_x2_p2_sp2 = lbs[9]
    ub_x2_p2_sp2 = ubs[9]

    if ub_x2_p2_sp2 - lb_x2_p2_sp2 < 1e-5
        lb_x2_p2_sp2 = lb_x2_p2_sp2 - 1e-5
        ub_x2_p2_sp2 = ub_x2_p2_sp2 + 1e-5
        lbs[9] = lb_x2_p2_sp2
        ubs[9] = ub_x2_p2_sp2
    end

    x2_p2_sp2_LB, x2_p2_sp2_UB = interpol_nd(bound_univariate(x2_p2_sp2, lb_x2_p2_sp2, ub_x2_p2_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₉ comes after x₈
    currList = [1]
    lbList = lbs[8:9]
    ubList = ubs[8:9]

    l_x2_p2_sp1_LB, l_x2_p2_sp1_UB = lift_OA(emptyList, currList, x2_p2_sp1_LB, x2_p2_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₈, x₉)
    emptyList = [1] #Since x₈ comes before x₉
    currList = [2]

    l_x2_p2_sp2_LB, l_x2_p2_sp2_UB = lift_OA(emptyList, currList, x2_p2_sp2_LB, x2_p2_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p2_LB, x2_p2_UB = prodBounds(l_x2_p2_sp1_LB, l_x2_p2_sp1_UB, l_x2_p2_sp2_LB, l_x2_p2_sp2_UB)

    # if sanityFlag
    #     validBounds(:(sin(x8)*sin(x9)), [:x8, :x9], x2_p2_LB, x2_p2_UB)
    # end
    #Part 3: f₃(x₅,x₇) = x₅*sin(x₇) 
    #Sub part 1: x₅
    x2_p3_sp1 = :(1*x)
    lb_x2_p3_sp1 = lbs[5]
    ub_x2_p3_sp1 = ubs[5]

    x2_p3_sp1_LB, x2_p3_sp1_UB = interpol_nd(bound_univariate(x2_p3_sp1, lb_x2_p3_sp1, ub_x2_p3_sp1, npoint=npoint)...)

    #Sub part 2: sin(x₇)
    x2_p3_sp2 = :(sin(x))
    lb_x2_p3_sp2 = lbs[7]
    ub_x2_p3_sp2 = ubs[7]

    if ub_x2_p3_sp2 - lb_x2_p3_sp2 < 1e-5
        lb_x2_p3_sp2 = lb_x2_p3_sp2 - 1e-5
        ub_x2_p3_sp2 = ub_x2_p3_sp2 + 1e-5
        lbs[7] = lb_x2_p3_sp2
        ubs[7] = ub_x2_p3_sp2
    end

    x2_p3_sp2_LB, x2_p3_sp2_UB = interpol_nd(bound_univariate(x2_p3_sp2, lb_x2_p3_sp2, ub_x2_p3_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₇ comes after x₅
    currList = [1]
    lbList = [lbs[5]]
    push!(lbList, lbs[7])
    ubList = [ubs[5]]
    push!(ubList, ubs[7])

    l_x2_p3_sp1_LB, l_x2_p3_sp1_UB = lift_OA(emptyList, currList, x2_p3_sp1_LB, x2_p3_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₅, x₇)
    emptyList = [1] #Since x₅ comes before x₇
    currList = [2]

    l_x2_p3_sp2_LB, l_x2_p3_sp2_UB = lift_OA(emptyList, currList, x2_p3_sp2_LB, x2_p3_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p3_LB, x2_p3_UB = prodBounds(l_x2_p3_sp1_LB, l_x2_p3_sp1_UB, l_x2_p3_sp2_LB, l_x2_p3_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x5*sin(x7)), [:x5, :x7], x2_p3_LB, x2_p3_UB)
    # end
    #Part 4: f₄(x₅,x₇) = x₆*cos(x₇)
    #Sub part 1: x₆
    x2_p4_sp1 = :(1*x)
    lb_x2_p4_sp1 = lbs[6]
    ub_x2_p4_sp1 = ubs[6]

    x2_p4_sp1_LB, x2_p4_sp1_UB = interpol_nd(bound_univariate(x2_p4_sp1, lb_x2_p4_sp1, ub_x2_p4_sp1, npoint=npoint)...)

    #Sub part 2: cos(x₇)
    x2_p4_sp2 = :(cos(x))
    lb_x2_p4_sp2 = lbs[7]
    ub_x2_p4_sp2 = ubs[7]

    if ub_x2_p3_sp2 - lb_x2_p3_sp2 < 1e-5
        lb_x2_p4_sp2 = lb_x2_p4_sp2 - 1e-5
        ub_x2_p4_sp2 = ub_x2_p4_sp2 + 1e-5
        lbs[7] = lb_x2_p4_sp2
        ubs[7] = ub_x2_p4_sp2
    end

    x2_p4_sp2_LB, x2_p4_sp2_UB = interpol_nd(bound_univariate(x2_p4_sp2, lb_x2_p4_sp2, ub_x2_p4_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₇ comes after x₆
    currList = [1]
    lbList = lbs[6:7]
    ubList = ubs[6:7]

    l_x2_p4_sp1_LB, l_x2_p4_sp1_UB = lift_OA(emptyList, currList, x2_p4_sp1_LB, x2_p4_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₆, x₇)
    emptyList = [1] #Since x₆ comes before x₇
    currList = [2]

    l_x2_p4_sp2_LB, l_x2_p4_sp2_UB = lift_OA(emptyList, currList, x2_p4_sp2_LB, x2_p4_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p4_LB, x2_p4_UB = prodBounds(l_x2_p4_sp1_LB, l_x2_p4_sp1_UB, l_x2_p4_sp2_LB, l_x2_p4_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x6*cos(x7)), [:x6, :x7], x2_p4_LB, x2_p4_UB)
    # end

    #Part 5: f₅(x₉) = sin(x₉)
    x2_p5 = :(sin(x))
    lb_x2_p5 = lbs[9]
    ub_x2_p5 = ubs[9]

    if ub_x2_p5 - lb_x2_p5 < 1e-5
        lb_x2_p5 = lb_x2_p5 - 1e-5
        ub_x2_p5 = ub_x2_p5 + 1e-5
        lbs[9] = lb_x2_p5
        ubs[9] = ub_x2_p5
    end

    x2_p5_LB, x2_p5_UB = interpol_nd(bound_univariate(x2_p5, lb_x2_p5, ub_x2_p5, npoint=npoint)...)

    # if sanityFlag
    #     validBounds(:(cos(x9)), [:x9], x2_p5_LB, x2_p5_UB)
    # end

   #Part 6: f₆(x₅, x₇) = x₅*cos(x₇)
    #Sub part 1: x₅
    x2_p6_sp1 = :(1*x)
    lb_x2_p6_sp1 = lbs[5]
    ub_x2_p6_sp1 = ubs[5]

    x2_p6_sp1_LB, x2_p6_sp1_UB = interpol_nd(bound_univariate(x2_p6_sp1, lb_x2_p6_sp1, ub_x2_p6_sp1, npoint=npoint)...)

    #Sub part 2: cos(x₇)
    x2_p6_sp2 = :(cos(x))
    lb_x2_p6_sp2 = lbs[7]
    ub_x2_p6_sp2 = ubs[7]

    if ub_x2_p6_sp2 - lb_x2_p6_sp2 < 1e-5
        lb_x2_p6_sp2 = lb_x2_p6_sp2 - 1e-5
        ub_x2_p6_sp2 = ub_x2_p6_sp2 + 1e-5
        lbs[7] = lb_x2_p6_sp2
        ubs[7] = ub_x2_p6_sp2
    end

    x2_p6_sp2_LB, x2_p6_sp2_UB = interpol_nd(bound_univariate(x2_p6_sp2, lb_x2_p6_sp2, ub_x2_p6_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₇ comes after x₅
    currList = [1]
    lbList = [lbs[5]]
    push!(lbList, lbs[7])
    ubList = [ubs[5]]
    push!(ubList, ubs[7])

    l_x2_p6_sp1_LB, l_x2_p6_sp1_UB = lift_OA(emptyList, currList, x2_p6_sp1_LB, x2_p6_sp1_UB, lbList, ubList)


    #Now lift sub part 2 to space of (x₅, x₇)
    emptyList = [1] #Since x₅ comes before x₇
    currList = [2]

    l_x2_p6_sp2_LB, l_x2_p6_sp2_UB = lift_OA(emptyList, currList, x2_p6_sp2_LB, x2_p6_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p6_LB, x2_p6_UB = prodBounds(l_x2_p6_sp1_LB, l_x2_p6_sp1_UB, l_x2_p6_sp2_LB, l_x2_p6_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x5*cos(x7)), [:x5, :x7], x2_p6_LB, x2_p6_UB)
    # end

    #Part 7: f₇(x₅, x₇) = x₆*sin(x₇)
    #Sub part 1: x₆
    x2_p7_sp1 = :(1*x)
    lb_x2_p7_sp1 = lbs[6]
    ub_x2_p7_sp1 = ubs[6]

    x2_p7_sp1_LB, x2_p7_sp1_UB = interpol_nd(bound_univariate(x2_p7_sp1, lb_x2_p7_sp1, ub_x2_p7_sp1, npoint=npoint)...)

    #Sub part 2: sin(x₇)
    x2_p7_sp2 = :(sin(x))
    lb_x2_p7_sp2 = lbs[7]
    ub_x2_p7_sp2 = ubs[7]

    if ub_x2_p7_sp2 - lb_x2_p7_sp2 < 1e-5
        lb_x2_p7_sp2 = lb_x2_p7_sp2 - 1e-5
        ub_x2_p7_sp2 = ub_x2_p7_sp2 + 1e-5
        lbs[7] = lb_x2_p7_sp2
        ubs[7] = ub_x2_p7_sp2
    end

    x2_p7_sp2_LB, x2_p7_sp2_UB = interpol_nd(bound_univariate(x2_p7_sp2, lb_x2_p7_sp2, ub_x2_p7_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₇ comes after x₆
    currList = [1]
    lbList = [lbs[6]]
    push!(lbList, lbs[7])
    ubList = [ubs[6]]
    push!(ubList, ubs[7])

    l_x2_p7_sp1_LB, l_x2_p7_sp1_UB = lift_OA(emptyList, currList, x2_p7_sp1_LB, x2_p7_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₆, x₇)
    emptyList = [1] #Since x₆ comes before x₇
    currList = [2]

    l_x2_p7_sp2_LB, l_x2_p7_sp2_UB = lift_OA(emptyList, currList, x2_p7_sp2_LB, x2_p7_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p7_LB, x2_p7_UB = prodBounds(l_x2_p7_sp1_LB, l_x2_p7_sp1_UB, l_x2_p7_sp2_LB, l_x2_p7_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x6*sin(x7)), [:x6, :x7], x2_p7_LB, x2_p7_UB)
    # end

    #Now beging combining chunks to get the full bounds
    #Define f₈(x₅, x₆, x₇) =  f₇(x₆, x₇) - f₆(x₅, x₇)
    #Lift f₆(x₅, x₇) to space of (x₅, x₆, x₇)
    emptyList = [2] #Since x₇ comes after x₅
    currList = [1, 3]
    lbList = lbs[5:7]
    ubList = ubs[5:7]

    l_x2_p6_LB, l_x2_p6_UB = lift_OA(emptyList, currList, x2_p6_LB, x2_p6_UB, lbList, ubList)

    #Lift f₇(x₆, x₇) to space of (x₅, x₆, x₇)
    emptyList = [1] #Since x₅ comes before x₆
    currList = [2, 3]

    l_x2_p7_LB, l_x2_p7_UB = lift_OA(emptyList, currList, x2_p7_LB, x2_p7_UB, lbList, ubList)

    #Now subtract the lifted bounds
    x2_p8_LB, x2_p8_UB = sumBounds(l_x2_p7_LB, l_x2_p7_UB, l_x2_p6_LB, l_x2_p6_UB,true)

    # if sanityFlag
    #     validBounds(:(x5*cos(x7) - x6*sin(x7)), [:x5, :x6, :x7], x2_p8_LB, x2_p8_UB)
    # end

    #Define f₉(x₅, x₆, x₇, x₉) = f₅(x₉) * f₈(x₅, x₆, x₇)
    #Lift f₅(x₉) to space of (x₅, x₆, x₇, x₉)
    emptyList = [1, 2, 3] #Since x₉ comes after x₅, x₆, x₇
    currList = [4]
    lbList = [lbs[5:7]..., lbs[9]]
    ubList = [ubs[5:7]..., ubs[9]]

    l_x2_p5_LB, l_x2_p5_UB = lift_OA(emptyList, currList, x2_p5_LB, x2_p5_UB, lbList, ubList)

    #Lift f₈(x₅, x₆, x₇) to space of (x₅, x₆, x₇, x₉)
    emptyList = [4] #Since x₉ comes after x₅, x₆, x₇
    currList = [1, 2, 3]

    l_x2_p8_LB, l_x2_p8_UB = lift_OA(emptyList, currList, x2_p8_LB, x2_p8_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p9_LB, x2_p9_UB = prodBounds(l_x2_p5_LB, l_x2_p5_UB, l_x2_p8_LB, l_x2_p8_UB)

    # if sanityFlag
    #     validBounds(:(cos(x9)*(x5*cos(x7) - x6*sin(x7))), [:x5, :x6, :x7, :x9], x2_p9_LB, x2_p9_UB)
    # end

    #Define f₁₀(x₅, x₆, x₇) = f₃(x₅, x₇) + f₄(x₆, x₇)
    #Lift f₃(x₅, x₇) to space of (x₅, x₆, x₇)
    emptyList = [2] #Since x₆ comes after x₅
    currList = [1, 3]
    lbList = lbs[5:7]
    ubList = ubs[5:7]

    l_x2_p3_LB, l_x2_p3_UB = lift_OA(emptyList, currList, x2_p3_LB, x2_p3_UB, lbList, ubList)

    #Lift f₄(x₆, x₇) to space of (x₅, x₆, x₇)
    emptyList = [1] #Since x₅ comes before x₆
    currList = [2, 3]

    l_x2_p4_LB, l_x2_p4_UB = lift_OA(emptyList, currList, x2_p4_LB, x2_p4_UB, lbList, ubList)

    #Now add the lifted bounds
    x2_p10_LB, x2_p10_UB = sumBounds(l_x2_p3_LB, l_x2_p3_UB, l_x2_p4_LB, l_x2_p4_UB, false)

    # if sanityFlag
    #     validBounds(:(x5*sin(x7) + x6*cos(x7)), [:x5, :x6, :x7], x2_p10_LB, x2_p10_UB)
    # end

    #Define f₁₁(x₅, x₆, x₇, x₈, x₉) = f₂(x₈, x₉) * f₁₀(x₅, x₆, x₇)
    #Lift f₂(x₈, x₉) to space of (x₅, x₆, x₇, x₈, x₉)
    emptyList = [1, 2, 3] #Since x₈, x₉ come after x₅, x₆, x₇
    currList = [4, 5]
    lbList = lbs[5:9]
    ubList = ubs[5:9]

    l_x2_p2_LB, l_x2_p2_UB = lift_OA(emptyList, currList, x2_p2_LB, x2_p2_UB, lbList, ubList)

    #Lift f₁₀(x₅, x₆, x₇) to space of (x₅, x₆, x₇, x₈, x₉)
    emptyList = [4, 5] #Since x₈, x₉ come after x₅, x₆, x₇
    currList = [1, 2, 3]

    l_x2_p10_LB, l_x2_p10_UB = lift_OA(emptyList, currList, x2_p10_LB, x2_p10_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p11_LB, x2_p11_UB = prodBounds(l_x2_p2_LB, l_x2_p2_UB, l_x2_p10_LB, l_x2_p10_UB)

    # if sanityFlag
    #     validBounds(:(sin(x8)*sin(x9)*(x5*sin(x7) + x6*cos(x7))), [:x5, :x6, :x7, :x8, :x9], x2_p11_LB, x2_p11_UB)
    # end

    #Combine 3 distinct chunks to get f₁₂(x₄, x₅, x₆, x₇, x₈, x₉) = f₁(x₄, x₈, x₉) + f₁₁(x₅, x₆, x₇, x₈, x₉) + f₉(x₅, x₆, x₇, x₉)

    #Lift f₁(x₄, x₈, x₉) to space of (x₄, x₅, x₆, x₇, x₈, x₉)
    emptyList = [2, 3, 4] #Since x₅, x₆, x₇, come after x₄ but before x₈, x₉
    currList = [1, 5, 6]
    lbList = lbs[4:9]
    ubList = ubs[4:9]

    l_x2_p1_LB, l_x2_p1_UB = lift_OA(emptyList, currList, x2_p1_LB, x2_p1_UB, lbList, ubList)

    #Lift f₁₁(x₅, x₆, x₇, x₈, x₉) to space of (x₄, x₅, x₆, x₇, x₈, x₉)
    emptyList = [1] #Since x₄ comes before x₅
    currList = [2, 3, 4, 5, 6]

    l_x2_p11_LB, l_x2_p11_UB = lift_OA(emptyList, currList, x2_p11_LB, x2_p11_UB, lbList, ubList)

    #Lift f₉(x₅, x₆, x₇, x₉) to space of (x₄, x₅, x₆, x₇, x₈, x₉)
    emptyList = [1, 5] #Since x₄ comes before x₅, x₆, x₇, and x₈ comes before x₉
    currList = [2, 3, 4, 6]

    l_x2_p9_LB, l_x2_p9_UB = lift_OA(emptyList, currList, x2_p9_LB, x2_p9_UB, lbList, ubList)

    #Combine the three lifted chunks to get f₁₂(x₄, x₅, x₆, x₇, x₈, x₉)
    x2_p12_LB_i, x2_p12_UB_i = sumBounds(l_x2_p1_LB, l_x2_p1_UB, l_x2_p11_LB, l_x2_p11_UB, false)
    x2_p12_LB, x2_p12_UB = sumBounds(x2_p12_LB_i, x2_p12_UB_i, l_x2_p9_LB, l_x2_p9_UB, false)

    if sanityFlag
        @assert validBounds(:(x4*cos(x8)*cos(x9) + sin(x8)*cos(x9)*(x5*sin(x7) + x6*cos(x7)) + sin(x9)*(-x5*cos(x7) + x6*sin(x7))), [:x4, :x5, :x6, :x7, :x8, :x9], x2_p12_LB, x2_p12_UB) "Invalid bounds for x2"
    end

    #Finally, bounds for x2 have to be a function of x2. Lift bounds to space of (x₂, x₄, x₅, x₆, x₇, x₈, x₉)
    emptyList = [1] #Since x₂ comes after x₄, x₅, x₆, x₇, x₈, x₉
    currList = [2, 3, 4, 5, 6, 7]
    lbList = [lbs[2], lbs[4:9]...]
    ubList = [ubs[2], ubs[4:9]...]

    x2_p12_LB_l, x2_p12_UB_l = lift_OA(emptyList, currList, x2_p12_LB, x2_p12_UB, lbList, ubList)
    
    return x2_p12_LB_l, x2_p12_UB_l
end

function bound_quadx2(Quad, plotFlag = false, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Follow a similar strategy to the x1 bounds. Break the initial function into 7 parts then combine parts to regain full bounds 

    #Part 1: f₁(x₄, x₈, x₉) = x₄*cos(x₈)*sin(x₉)
    #K-A decomposition is exp(log(x₄) + log(cos(x₈)) + log(cos(x₉)))

    #Sub-part 1 = x₄
    x2_p1_sp1 = :(1*x)
    lb_x2_p1_sp1 = lbs[4]
    ub_x2_p1_sp1 = ubs[4]

    x2_p1_sp1_LB, x2_p1_sp1_UB = interpol_nd(bound_univariate(x2_p1_sp1, lb_x2_p1_sp1, ub_x2_p1_sp1, npoint=npoint)...)

    #Sub-part 2 = cos(x₈)
    x2_p1_sp2 = :(cos(x))
    lb_x2_p1_sp2 = lbs[8]
    ub_x2_p1_sp2 = ubs[8]

    #Bounding trig functions over very thin intervals is tricky. Widen for now
    if ub_x2_p1_sp2 - lb_x2_p1_sp2 < 1e-5
        lb_x2_p1_sp2 = lb_x2_p1_sp2 - 1e-5
        ub_x2_p1_sp2 = ub_x2_p1_sp2 + 1e-5
        lbs[8] = lb_x2_p1_sp2
        ubs[8] = ub_x2_p1_sp2
    end

    x2_p1_sp2_LB, x2_p1_sp2_UB = interpol_nd(bound_univariate(x2_p1_sp2, lb_x2_p1_sp2, ub_x2_p1_sp2, npoint=npoint)...)

    #Sub-part 3 = sin(x₉)
    x2_p1_sp3 = :(sin(x))
    lb_x2_p1_sp3 = lbs[9]
    ub_x2_p1_sp3 = ubs[9]

    #Bounding trig functions over very thin intervals is tricky. Widen for now
    if ub_x2_p1_sp3 - lb_x2_p1_sp3 < 1e-5
        lb_x2_p1_sp3 = lb_x2_p1_sp3 - 1e-5
        ub_x2_p1_sp3 = ub_x2_p1_sp3 + 1e-5
        lbs[9] = lb_x2_p1_sp3
        ubs[9] = ub_x2_p1_sp3
    end

    x2_p1_sp3_LB, x2_p1_sp3_UB = interpol_nd(bound_univariate(x2_p1_sp3, lb_x2_p1_sp3, ub_x2_p1_sp3, npoint=npoint)...)

    #Avoid Minksum and just directly evaluate the products of the bounds
    #To take the products directly, lift each bound to the same set of dimensions, then multiply. Equiv to exp-log trick

    #So.. first lift sub part 1 to space of (x₄, x₈, x₉)
    emptyList = [2,3] #Since x₈, x₉ come after x₄
    currList = [1]
    lbList = lbs[8:9]
    pushfirst!(lbList, lbs[4])
    ubList = ubs[8:9]
    pushfirst!(ubList, ubs[4])

    l_x2_p1_sp1_LB, l_x2_p1_sp1_UB = lift_OA(emptyList, currList, x2_p1_sp1_LB, x2_p1_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₄, x₈, x₉)
    emptyList = [1,3] #Since x₄, x₉ come after x₈
    currList = [2]

    l_x2_p1_sp2_LB, l_x2_p1_sp2_UB = lift_OA(emptyList, currList, x2_p1_sp2_LB, x2_p1_sp2_UB, lbList, ubList)

    #Now lift sub part 3 to space of (x₄, x₈, x₉)
    emptyList = [1,2] #Since x₄, x₈ come after x₉
    currList = [3]

    l_x2_p1_sp3_LB, l_x2_p1_sp3_UB = lift_OA(emptyList, currList, x2_p1_sp3_LB, x2_p1_sp3_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p1_LB_i, x2_p1_UB_i = prodBounds(l_x2_p1_sp1_LB, l_x2_p1_sp1_UB, l_x2_p1_sp2_LB, l_x2_p1_sp2_UB)
    x2_p1_LB, x2_p1_UB = prodBounds(x2_p1_LB_i, x2_p1_UB_i, l_x2_p1_sp3_LB, l_x2_p1_sp3_UB)

    # if sanityFlag
    #     validBounds(:(x4*cos(x8)*cos(x9)), [:x4, :x8, :x9], x2_p1_LB, x2_p1_UB)
    # end


    #Part 2: f₂(x₈, x₉) = sin(x₈)*sin(x₉)
    #Sub part 1 = sin(x₈)
    x2_p2_sp1 = :(sin(x))
    lb_x2_p2_sp1 = lbs[8]
    ub_x2_p2_sp1 = ubs[8]

    if ub_x2_p2_sp1 - lb_x2_p2_sp1 < 1e-5
        lb_x2_p2_sp1 = lb_x2_p2_sp1 - 1e-5
        ub_x2_p2_sp1 = ub_x2_p2_sp1 + 1e-5
        lbs[8] = lb_x2_p2_sp1
        ubs[8] = ub_x2_p2_sp1
    end

    x2_p2_sp1_LB, x2_p2_sp1_UB = interpol_nd(bound_univariate(x2_p2_sp1, lb_x2_p2_sp1, ub_x2_p2_sp1, npoint=npoint)...)

    #Sub part 2 = sin(x₉)
    x2_p2_sp2 = :(sin(x))
    lb_x2_p2_sp2 = lbs[9]
    ub_x2_p2_sp2 = ubs[9]

    if ub_x2_p2_sp2 - lb_x2_p2_sp2 < 1e-5
        lb_x2_p2_sp2 = lb_x2_p2_sp2 - 1e-5
        ub_x2_p2_sp2 = ub_x2_p2_sp2 + 1e-5
        lbs[9] = lb_x2_p2_sp2
        ubs[9] = ub_x2_p2_sp2
    end

    x2_p2_sp2_LB, x2_p2_sp2_UB = interpol_nd(bound_univariate(x2_p2_sp2, lb_x2_p2_sp2, ub_x2_p2_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₉ comes after x₈
    currList = [1]
    lbList = lbs[8:9]
    ubList = ubs[8:9]

    l_x2_p2_sp1_LB, l_x2_p2_sp1_UB = lift_OA(emptyList, currList, x2_p2_sp1_LB, x2_p2_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₈, x₉)
    emptyList = [1] #Since x₈ comes before x₉
    currList = [2]

    l_x2_p2_sp2_LB, l_x2_p2_sp2_UB = lift_OA(emptyList, currList, x2_p2_sp2_LB, x2_p2_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p2_LB, x2_p2_UB = prodBounds(l_x2_p2_sp1_LB, l_x2_p2_sp1_UB, l_x2_p2_sp2_LB, l_x2_p2_sp2_UB)

    # if sanityFlag
    #     validBounds(:(sin(x8)*sin(x9)), [:x8, :x9], x2_p2_LB, x2_p2_UB)
    # end

    #Part 3: f₃(x₅,x₇) = x₅*sin(x₇) 
    #Sub part 1: x₅
    x2_p3_sp1 = :(1*x)
    lb_x2_p3_sp1 = lbs[5]
    ub_x2_p3_sp1 = ubs[5]

    x2_p3_sp1_LB, x2_p3_sp1_UB = interpol_nd(bound_univariate(x2_p3_sp1, lb_x2_p3_sp1, ub_x2_p3_sp1, npoint=npoint)...)

    #Sub part 2: sin(x₇)
    x2_p3_sp2 = :(sin(x))
    lb_x2_p3_sp2 = lbs[7]
    ub_x2_p3_sp2 = ubs[7]

    if ub_x2_p3_sp2 - lb_x2_p3_sp2 < 1e-5
        lb_x2_p3_sp2 = lb_x2_p3_sp2 - 1e-5
        ub_x2_p3_sp2 = ub_x2_p3_sp2 + 1e-5
        lbs[7] = lb_x2_p3_sp2
        ubs[7] = ub_x2_p3_sp2
    end

    x2_p3_sp2_LB, x2_p3_sp2_UB = interpol_nd(bound_univariate(x2_p3_sp2, lb_x2_p3_sp2, ub_x2_p3_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₇ comes after x₅
    currList = [1]
    lbList = [lbs[5]]
    push!(lbList, lbs[7])
    ubList = [ubs[5]]
    push!(ubList, ubs[7])

    l_x2_p3_sp1_LB, l_x2_p3_sp1_UB = lift_OA(emptyList, currList, x2_p3_sp1_LB, x2_p3_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₅, x₇)
    emptyList = [1] #Since x₅ comes before x₇
    currList = [2]

    l_x2_p3_sp2_LB, l_x2_p3_sp2_UB = lift_OA(emptyList, currList, x2_p3_sp2_LB, x2_p3_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p3_LB, x2_p3_UB = prodBounds(l_x2_p3_sp1_LB, l_x2_p3_sp1_UB, l_x2_p3_sp2_LB, l_x2_p3_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x5*sin(x7)), [:x5, :x7], x2_p3_LB, x2_p3_UB)
    # end

    #Part 4: f₄(x₅,x₇) = x₆*cos(x₇)
    #Sub part 1: x₆
    x2_p4_sp1 = :(1*x)
    lb_x2_p4_sp1 = lbs[6]
    ub_x2_p4_sp1 = ubs[6]

    x2_p4_sp1_LB, x2_p4_sp1_UB = interpol_nd(bound_univariate(x2_p4_sp1, lb_x2_p4_sp1, ub_x2_p4_sp1, npoint=npoint)...)

    #Sub part 2: cos(x₇)
    x2_p4_sp2 = :(cos(x))
    lb_x2_p4_sp2 = lbs[7]
    ub_x2_p4_sp2 = ubs[7]

    if ub_x2_p3_sp2 - lb_x2_p3_sp2 < 1e-5
        lb_x2_p4_sp2 = lb_x2_p4_sp2 - 1e-5
        ub_x2_p4_sp2 = ub_x2_p4_sp2 + 1e-5
        lbs[7] = lb_x2_p4_sp2
        ubs[7] = ub_x2_p4_sp2
    end

    x2_p4_sp2_LB, x2_p4_sp2_UB = interpol_nd(bound_univariate(x2_p4_sp2, lb_x2_p4_sp2, ub_x2_p4_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₇ comes after x₆
    currList = [1]
    lbList = lbs[6:7]
    ubList = ubs[6:7]

    l_x2_p4_sp1_LB, l_x2_p4_sp1_UB = lift_OA(emptyList, currList, x2_p4_sp1_LB, x2_p4_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₆, x₇)
    emptyList = [1] #Since x₆ comes before x₇
    currList = [2]

    l_x2_p4_sp2_LB, l_x2_p4_sp2_UB = lift_OA(emptyList, currList, x2_p4_sp2_LB, x2_p4_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p4_LB, x2_p4_UB = prodBounds(l_x2_p4_sp1_LB, l_x2_p4_sp1_UB, l_x2_p4_sp2_LB, l_x2_p4_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x6*cos(x7)), [:x6, :x7], x2_p4_LB, x2_p4_UB)
    # end

    #Part 5: f₅(x₉) = cos(x₉)
    x2_p5 = :(cos(x))
    lb_x2_p5 = lbs[9]
    ub_x2_p5 = ubs[9]

    if ub_x2_p5 - lb_x2_p5 < 1e-5
        lb_x2_p5 = lb_x2_p5 - 1e-5
        ub_x2_p5 = ub_x2_p5 + 1e-5
        lbs[9] = lb_x2_p5
        ubs[9] = ub_x2_p5
    end

    x2_p5_LB, x2_p5_UB = interpol_nd(bound_univariate(x2_p5, lb_x2_p5, ub_x2_p5, npoint=npoint)...)

    # if sanityFlag
    #     validBounds(:(cos(x9)), [:x9], x2_p5_LB, x2_p5_UB)
    # end

    #Part 6: f₆(x₅, x₇) = x₅*cos(x₇)
    #Sub part 1: x₅
    x2_p6_sp1 = :(1*x)
    lb_x2_p6_sp1 = lbs[5]
    ub_x2_p6_sp1 = ubs[5]

    x2_p6_sp1_LB, x2_p6_sp1_UB = interpol_nd(bound_univariate(x2_p6_sp1, lb_x2_p6_sp1, ub_x2_p6_sp1, npoint=npoint)...)

    #Sub part 2: cos(x₇)
    x2_p6_sp2 = :(cos(x))
    lb_x2_p6_sp2 = lbs[7]
    ub_x2_p6_sp2 = ubs[7]

    if ub_x2_p6_sp2 - lb_x2_p6_sp2 < 1e-5
        lb_x2_p6_sp2 = lb_x2_p6_sp2 - 1e-5
        ub_x2_p6_sp2 = ub_x2_p6_sp2 + 1e-5
        lbs[7] = lb_x2_p6_sp2
        ubs[7] = ub_x2_p6_sp2
    end

    x2_p6_sp2_LB, x2_p6_sp2_UB = interpol_nd(bound_univariate(x2_p6_sp2, lb_x2_p6_sp2, ub_x2_p6_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₇ comes after x₅
    currList = [1]
    lbList = [lbs[5]]
    push!(lbList, lbs[7])
    ubList = [ubs[5]]
    push!(ubList, ubs[7])

    l_x2_p6_sp1_LB, l_x2_p6_sp1_UB = lift_OA(emptyList, currList, x2_p6_sp1_LB, x2_p6_sp1_UB, lbList, ubList)


    #Now lift sub part 2 to space of (x₅, x₇)
    emptyList = [1] #Since x₅ comes before x₇
    currList = [2]

    l_x2_p6_sp2_LB, l_x2_p6_sp2_UB = lift_OA(emptyList, currList, x2_p6_sp2_LB, x2_p6_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p6_LB, x2_p6_UB = prodBounds(l_x2_p6_sp1_LB, l_x2_p6_sp1_UB, l_x2_p6_sp2_LB, l_x2_p6_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x5*cos(x7)), [:x5, :x7], x2_p6_LB, x2_p6_UB)
    # end

    #Part 7: f₇(x₅, x₇) = x₆*sin(x₇)
    #Sub part 1: x₆
    x2_p7_sp1 = :(1*x)
    lb_x2_p7_sp1 = lbs[6]
    ub_x2_p7_sp1 = ubs[6]

    x2_p7_sp1_LB, x2_p7_sp1_UB = interpol_nd(bound_univariate(x2_p7_sp1, lb_x2_p7_sp1, ub_x2_p7_sp1, npoint=npoint)...)

    #Sub part 2: sin(x₇)
    x2_p7_sp2 = :(sin(x))
    lb_x2_p7_sp2 = lbs[7]
    ub_x2_p7_sp2 = ubs[7]

    if ub_x2_p7_sp2 - lb_x2_p7_sp2 < 1e-5
        lb_x2_p7_sp2 = lb_x2_p7_sp2 - 1e-5
        ub_x2_p7_sp2 = ub_x2_p7_sp2 + 1e-5
        lbs[7] = lb_x2_p7_sp2
        ubs[7] = ub_x2_p7_sp2
    end

    x2_p7_sp2_LB, x2_p7_sp2_UB = interpol_nd(bound_univariate(x2_p7_sp2, lb_x2_p7_sp2, ub_x2_p7_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₇ comes after x₆
    currList = [1]
    lbList = [lbs[6]]
    push!(lbList, lbs[7])
    ubList = [ubs[6]]
    push!(ubList, ubs[7])

    l_x2_p7_sp1_LB, l_x2_p7_sp1_UB = lift_OA(emptyList, currList, x2_p7_sp1_LB, x2_p7_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₆, x₇)
    emptyList = [1] #Since x₆ comes before x₇
    currList = [2]

    l_x2_p7_sp2_LB, l_x2_p7_sp2_UB = lift_OA(emptyList, currList, x2_p7_sp2_LB, x2_p7_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p7_LB, x2_p7_UB = prodBounds(l_x2_p7_sp1_LB, l_x2_p7_sp1_UB, l_x2_p7_sp2_LB, l_x2_p7_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x6*sin(x7)), [:x6, :x7], x2_p7_LB, x2_p7_UB)
    # end

    #Now beging combining chunks to get the full bounds
    #Define f₈(x₅, x₆, x₇) = f₆(x₅, x₇) - f₇(x₆, x₇)
    #Lift f₆(x₅, x₇) to space of (x₅, x₆, x₇)
    emptyList = [2] #Since x₇ comes after x₅
    currList = [1, 3]
    lbList = lbs[5:7]
    ubList = ubs[5:7]

    l_x2_p6_LB, l_x2_p6_UB = lift_OA(emptyList, currList, x2_p6_LB, x2_p6_UB, lbList, ubList)

    #Lift f₇(x₆, x₇) to space of (x₅, x₆, x₇)
    emptyList = [1] #Since x₅ comes before x₆
    currList = [2, 3]

    l_x2_p7_LB, l_x2_p7_UB = lift_OA(emptyList, currList, x2_p7_LB, x2_p7_UB, lbList, ubList)

    #Now subtract the lifted bounds
    x2_p8_LB, x2_p8_UB = sumBounds(l_x2_p6_LB, l_x2_p6_UB, l_x2_p7_LB, l_x2_p7_UB, true)

    # if sanityFlag
    #     validBounds(:(x5*cos(x7) - x6*sin(x7)), [:x5, :x6, :x7], x2_p8_LB, x2_p8_UB)
    # end

    #Define f₉(x₅, x₆, x₇, x₉) = f₅(x₉) * f₈(x₅, x₆, x₇)
    #Lift f₅(x₉) to space of (x₅, x₆, x₇, x₉)
    emptyList = [1, 2, 3] #Since x₉ comes after x₅, x₆, x₇
    currList = [4]
    lbList = [lbs[5:7]..., lbs[9]]
    ubList = [ubs[5:7]..., ubs[9]]

    l_x2_p5_LB, l_x2_p5_UB = lift_OA(emptyList, currList, x2_p5_LB, x2_p5_UB, lbList, ubList)

    #Lift f₈(x₅, x₆, x₇) to space of (x₅, x₆, x₇, x₉)
    emptyList = [4] #Since x₉ comes after x₅, x₆, x₇
    currList = [1, 2, 3]

    l_x2_p8_LB, l_x2_p8_UB = lift_OA(emptyList, currList, x2_p8_LB, x2_p8_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p9_LB, x2_p9_UB = prodBounds(l_x2_p5_LB, l_x2_p5_UB, l_x2_p8_LB, l_x2_p8_UB)

    # if sanityFlag
    #     validBounds(:(cos(x9)*(x5*cos(x7) - x6*sin(x7))), [:x5, :x6, :x7, :x9], x2_p9_LB, x2_p9_UB)
    # end

    #Define f₁₀(x₅, x₆, x₇) = f₃(x₅, x₇) + f₄(x₆, x₇)
    #Lift f₃(x₅, x₇) to space of (x₅, x₆, x₇)
    emptyList = [2] #Since x₆ comes after x₅
    currList = [1, 3]
    lbList = lbs[5:7]
    ubList = ubs[5:7]

    l_x2_p3_LB, l_x2_p3_UB = lift_OA(emptyList, currList, x2_p3_LB, x2_p3_UB, lbList, ubList)

    #Lift f₄(x₆, x₇) to space of (x₅, x₆, x₇)
    emptyList = [1] #Since x₅ comes before x₆
    currList = [2, 3]

    l_x2_p4_LB, l_x2_p4_UB = lift_OA(emptyList, currList, x2_p4_LB, x2_p4_UB, lbList, ubList)

    #Now add the lifted bounds
    x2_p10_LB, x2_p10_UB = sumBounds(l_x2_p3_LB, l_x2_p3_UB, l_x2_p4_LB, l_x2_p4_UB, false)

    # if sanityFlag
    #     validBounds(:(x5*sin(x7) + x6*cos(x7)), [:x5, :x6, :x7], x2_p10_LB, x2_p10_UB)
    # end

    #Define f₁₁(x₅, x₆, x₇, x₈, x₉) = f₂(x₈, x₉) * f₁₀(x₅, x₆, x₇)
    #Lift f₂(x₈, x₉) to space of (x₅, x₆, x₇, x₈, x₉)
    emptyList = [1, 2, 3] #Since x₈, x₉ come after x₅, x₆, x₇
    currList = [4, 5]
    lbList = lbs[5:9]
    ubList = ubs[5:9]

    l_x2_p2_LB, l_x2_p2_UB = lift_OA(emptyList, currList, x2_p2_LB, x2_p2_UB, lbList, ubList)

    #Lift f₁₀(x₅, x₆, x₇) to space of (x₅, x₆, x₇, x₈, x₉)
    emptyList = [4, 5] #Since x₈, x₉ come after x₅, x₆, x₇
    currList = [1, 2, 3]

    l_x2_p10_LB, l_x2_p10_UB = lift_OA(emptyList, currList, x2_p10_LB, x2_p10_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x2_p11_LB, x2_p11_UB = prodBounds(l_x2_p2_LB, l_x2_p2_UB, l_x2_p10_LB, l_x2_p10_UB)

    # if sanityFlag
    #     validBounds(:(sin(x8)*sin(x9)*(x5*sin(x7) + x6*cos(x7))), [:x5, :x6, :x7, :x8, :x9], x2_p11_LB, x2_p11_UB)
    # end

    #Combine 3 distinct chunks to get f₁₂(x₄, x₅, x₆, x₇, x₈, x₉) = f₁(x₄, x₈, x₉) + f₁₁(x₅, x₆, x₇, x₈, x₉) + f₉(x₅, x₆, x₇, x₉)

    #Lift f₁(x₄, x₈, x₉) to space of (x₄, x₅, x₆, x₇, x₈, x₉)
    emptyList = [2, 3, 4] #Since x₅, x₆, x₇, come after x₄ but before x₈, x₉
    currList = [1, 5, 6]
    lbList = lbs[4:9]
    ubList = ubs[4:9]

    l_x2_p1_LB, l_x2_p1_UB = lift_OA(emptyList, currList, x2_p1_LB, x2_p1_UB, lbList, ubList)

    #Lift f₁₁(x₅, x₆, x₇, x₈, x₉) to space of (x₄, x₅, x₆, x₇, x₈, x₉)
    emptyList = [1] #Since x₄ comes before x₅
    currList = [2, 3, 4, 5, 6]

    l_x2_p11_LB, l_x2_p11_UB = lift_OA(emptyList, currList, x2_p11_LB, x2_p11_UB, lbList, ubList)

    #Lift f₉(x₅, x₆, x₇, x₉) to space of (x₄, x₅, x₆, x₇, x₈, x₉)
    emptyList = [1, 5] #Since x₄ comes before x₅, x₆, x₇, and x₈ comes before x₉
    currList = [2, 3, 4, 6]

    l_x2_p9_LB, l_x2_p9_UB = lift_OA(emptyList, currList, x2_p9_LB, x2_p9_UB, lbList, ubList)

    #Combine the three lifted chunks to get f₁₂(x₄, x₅, x₆, x₇, x₈, x₉)
    x2_p12_LB_i, x2_p12_UB_i = sumBounds(l_x2_p1_LB, l_x2_p1_UB, l_x2_p11_LB, l_x2_p11_UB, false)
    x2_p12_LB, x2_p12_UB = sumBounds(x2_p12_LB_i, x2_p12_UB_i, l_x2_p9_LB, l_x2_p9_UB, false)

    if sanityFlag
        @assert validBounds(:(x4*cos(x8)*sin(x9) + sin(x8)*sin(x9)*(x5*sin(x7) + x6*cos(x7)) + cos(x9)*(x5*cos(x7) - x6*sin(x7))), [:x4, :x5, :x6, :x7, :x8, :x9], x2_p12_LB, x2_p12_UB) "Invalid bounds for x2"
    end

    #Finally, bounds for x2 have to be a function of x2. Lift bounds to space of (x₂, x₄, x₅, x₆, x₇, x₈, x₉)
    emptyList = [1] #Since x₂ comes after x₄, x₅, x₆, x₇, x₈, x₉
    currList = [2, 3, 4, 5, 6, 7]
    lbList = [lbs[2], lbs[4:9]...]
    ubList = [ubs[2], ubs[4:9]...]

    x2_p12_LB_l, x2_p12_UB_l = lift_OA(emptyList, currList, x2_p12_LB, x2_p12_UB, lbList, ubList)
    
    return x2_p12_LB_l, x2_p12_UB_l
end

function bound_quadx3(Quad, plotFlag = false, sanityFlag = true, npoint=1)

    lbs, ubs = extrema(Quad.domain)

    #Part 1: f₁(x₄, x₈) = x₄*sin(x₈)
    #Sub-part 1: x₄
    x3_p1_sp1 = :(1*x)
    lb_x3_p1_sp1 = lbs[4]
    ub_x3_p1_sp1 = ubs[4]

    x3_p1_sp1_LB, x3_p1_sp1_UB = interpol_nd(bound_univariate(x3_p1_sp1, lb_x3_p1_sp1, ub_x3_p1_sp1, npoint=npoint)...)

    #Sub-part 2: sin(x₈)
    x3_p1_sp2 = :(sin(x₈))
    lb_x3_p1_sp2 = lbs[8]
    ub_x3_p1_sp2 = ubs[8]

    if ub_x3_p1_sp2 - lb_x3_p1_sp2 < 1e-5
        lb_x3_p1_sp2 = lb_x3_p1_sp2 - 1e-5
        ub_x3_p1_sp2 = ub_x3_p1_sp2 + 1e-5  
        lbs[8] = lb_x3_p1_sp2
        ubs[8] = ub_x3_p1_sp2
    end

    x3_p1_sp2_LB, x3_p1_sp2_UB = interpol_nd(bound_univariate(x3_p1_sp2, lb_x3_p1_sp2, ub_x3_p1_sp2, npoint=npoint)...)

    #Evalute the product of the bounds
    #First, lift sub part 1 to a space of (x₄, x₈)
    emptyList = [2] #because x8 comes after x4
    currList = [1]
    lbList = [lbs[4], lbs[8]]
    ubList = [ubs[4], ubs[8]]

    l_x3_p1_sp1_LB, l_x3_p1_sp1_UB  = lift_OA(emptyList, currList, x3_p1_sp1_LB, x3_p1_sp1_UB,  lbList, ubList)  

    #Next, lift sub part 2 to a space of (x₄, x₈)
    emptyList = [1] #because x4 comes before x8
    currList = [2]

    l_x3_p1_sp2_LB, l_x3_p1_sp2_UB  = lift_OA(emptyList, currList, x3_p1_sp2_LB, x3_p1_sp2_UB,  lbList, ubList)

    #Now, multiply the lifted bounds
    x3_p1_LB, x3_p1_UB = prodBounds(l_x3_p1_sp1_LB, l_x3_p1_sp1_UB, l_x3_p1_sp2_LB, l_x3_p1_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x₄*sin(x₈)), [:x₄, :x₈], x3_p1_LB, x3_p1_UB)
    # end
    
    #Part 2: f₂(x₅, x₇, x₈) = x₅*sin(x₇)*cos(x₈) 
    #Sub-part 1: x₅
    x3_p2_sp1 = :(1*x)
    lb_x3_p2_sp1 = lbs[5]
    ub_x3_p2_sp1 = ubs[5]

    x3_p2_sp1_LB, x3_p2_sp1_UB = interpol_nd(bound_univariate(x3_p2_sp1, lb_x3_p2_sp1, ub_x3_p2_sp1, npoint=npoint)...)

    #Sub-part 2: sin(x₇)
    x3_p2_sp2 = :(sin(x₇))
    lb_x3_p2_sp2 = lbs[7]
    ub_x3_p2_sp2 = ubs[7]

    if ub_x3_p2_sp2 - lb_x3_p2_sp2 < 1e-5
        lb_x3_p2_sp2 = lb_x3_p2_sp2 - 1e-5
        ub_x3_p2_sp2 = ub_x3_p2_sp2 + 1e-5  
        lbs[7] = lb_x3_p2_sp2
        ubs[7] = ub_x3_p2_sp2
    end

    x3_p2_sp2_LB, x3_p2_sp2_UB = interpol_nd(bound_univariate(x3_p2_sp2, lb_x3_p2_sp2, ub_x3_p2_sp2, npoint=npoint)...)
    
    #Sub-part 3: cos(x₈)
    x3_p2_sp3 = :(cos(x₈))
    lb_x3_p2_sp3 = lbs[8]
    ub_x3_p2_sp3 = ubs[8]

    if ub_x3_p2_sp3 - lb_x3_p2_sp3 < 1e-5
        lb_x3_p2_sp3 = lb_x3_p2_sp3 - 1e-5
        ub_x3_p2_sp3 = ub_x3_p2_sp3 + 1e-5  
        lbs[8] = lb_x3_p2_sp3
        ubs[8] = ub_x3_p2_sp3
    end

    x3_p2_sp3_LB, x3_p2_sp3_UB = interpol_nd(bound_univariate(x3_p2_sp3, lb_x3_p2_sp3, ub_x3_p2_sp3, npoint=npoint)...)

    #Evalute the product of the bounds
    #First, lift sub part 1 to a space of (x₅, x₇, x₈)
    emptyList = [2,3] #because x7,x8 comes after x5
    currList = [1]
    lbList = [lbs[5], lbs[7], lbs[8]]
    ubList = [ubs[5], ubs[7], ubs[8]]

    l_x3_p2_sp1_LB, l_x3_p2_sp1_UB  = lift_OA(emptyList, currList, x3_p2_sp1_LB, x3_p2_sp1_UB,  lbList, ubList)

    #Next, lift sub part 2 to a space of (x₅, x₇, x₈)
    emptyList = [1,3] #because x5 comes before x7 and x8 comes after
    currList = [2]

    l_x3_p2_sp2_LB, l_x3_p2_sp2_UB  = lift_OA(emptyList, currList, x3_p2_sp2_LB, x3_p2_sp2_UB,  lbList, ubList)

    #Next, lift sub part 3 to a space of (x₅, x₇, x₈)
    emptyList = [1,2] #because x5 comes before x7 and x7 comes before x8
    currList = [3]

    l_x3_p2_sp3_LB, l_x3_p2_sp3_UB  = lift_OA(emptyList, currList, x3_p2_sp3_LB, x3_p2_sp3_UB,  lbList, ubList)

    #Now, multiply the lifted bounds
    x3_p2_LB_i, x3_p2_UB_i = prodBounds(l_x3_p2_sp1_LB, l_x3_p2_sp1_UB, l_x3_p2_sp2_LB, l_x3_p2_sp2_UB)
    x3_p2_LB, x3_p2_UB = prodBounds(x3_p2_LB_i, x3_p2_UB_i, l_x3_p2_sp3_LB, l_x3_p2_sp3_UB)

    # if sanityFlag
    #     validBounds(:(x₅*sin(x₇)*cos(x₈)), [:x₅, :x₇, :x₈], x3_p2_LB, x3_p2_UB)
    # end
    
    #Part 3: f₃(x₆,x₇,x₈) = x₆*cos(x₇)*cos(x₈)
    #Sub-part 1: x₆
    x3_p3_sp1 = :(1*x)
    lb_x3_p3_sp1 = lbs[6]
    ub_x3_p3_sp1 = ubs[6]

    x3_p3_sp1_LB, x3_p3_sp1_UB = interpol_nd(bound_univariate(x3_p3_sp1, lb_x3_p3_sp1, ub_x3_p3_sp1, npoint=npoint)...)

    #Sub-part 2: cos(x₇)
    x3_p3_sp2 = :(cos(x₇))
    lb_x3_p3_sp2 = lbs[7]
    ub_x3_p3_sp2 = ubs[7]

    if ub_x3_p3_sp2 - lb_x3_p3_sp2 < 1e-5
        lb_x3_p3_sp2 = lb_x3_p3_sp2 - 1e-5
        ub_x3_p3_sp2 = ub_x3_p3_sp2 + 1e-5  
        lbs[7] = lb_x3_p3_sp2
        ubs[7] = ub_x3_p3_sp2
    end

    x3_p3_sp2_LB, x3_p3_sp2_UB = interpol_nd(bound_univariate(x3_p3_sp2, lb_x3_p3_sp2, ub_x3_p3_sp2, npoint=npoint)...)

    #Sub-part 3: cos(x₈)
    x3_p3_sp3 = :(cos(x₈))
    lb_x3_p3_sp3 = lbs[8]
    ub_x3_p3_sp3 = ubs[8]

    if ub_x3_p3_sp3 - lb_x3_p3_sp3 < 1e-5
        lb_x3_p3_sp3 = lb_x3_p3_sp3 - 1e-5
        ub_x3_p3_sp3 = ub_x3_p3_sp3 + 1e-5  
        lbs[8] = lb_x3_p3_sp3
        ubs[8] = ub_x3_p3_sp3
    end

    x3_p3_sp3_LB, x3_p3_sp3_UB = interpol_nd(bound_univariate(x3_p3_sp3, lb_x3_p3_sp3, ub_x3_p3_sp3, npoint=npoint)...)
    
    #Evalute the product of the bounds
    #First, lift sub part 1 to a space of (x₆, x₇, x₈)
    emptyList = [2,3] #because x7,x8 comes after x6
    currList = [1]
    lbList = lbs[6:8]
    ubList = ubs[6:8]

    l_x3_p3_sp1_LB, l_x3_p3_sp1_UB  = lift_OA(emptyList, currList, x3_p3_sp1_LB, x3_p3_sp1_UB,  lbList, ubList)

    #Next, lift sub part 2 to a space of (x₆, x₇, x₈)
    emptyList = [1,3] #because x6 comes before x7 and x8 comes after
    currList = [2]

    l_x3_p3_sp2_LB, l_x3_p3_sp2_UB  = lift_OA(emptyList, currList, x3_p3_sp2_LB, x3_p3_sp2_UB,  lbList, ubList)

    #Next, lift sub part 3 to a space of (x₆, x₇, x₈)
    emptyList = [1,2] #because x6 comes before x7 and x7 comes before x8
    currList = [3]

    l_x3_p3_sp3_LB, l_x3_p3_sp3_UB  = lift_OA(emptyList, currList, x3_p3_sp3_LB, x3_p3_sp3_UB,  lbList, ubList)

    #Now, multiply the lifted bounds
    x3_p3_LB_i, x3_p3_UB_i = prodBounds(l_x3_p3_sp1_LB, l_x3_p3_sp1_UB, l_x3_p3_sp2_LB, l_x3_p3_sp2_UB)
    x3_p3_LB, x3_p3_UB = prodBounds(x3_p3_LB_i, x3_p3_UB_i, l_x3_p3_sp3_LB, l_x3_p3_sp3_UB)

    # if sanityFlag
    #     validBounds(:(x₆*cos(x₇)*cos(x₈)), [:x₆, :x₇, :x₈], x3_p3_LB, x3_p3_UB)
    # end

    #Finally, combine the bounds to get f(x₄, x₅, x₆, x₇, x₈) = x₄*sin(x₈) - x₅*sin(x₇)*cos(x₈) - x₆*cos(x₇)*cos(x₈)

    #First, lift part 1 to a space of (x₄, x₅, x₆, x₇, x₈)
    emptyList = [2,3,4] #because x5,x6,x7 comes after x4 and before x8
    currList = [1,5]
    lbList = lbs[4:8]
    ubList = ubs[4:8]

    l_x3_p1_LB, l_x3_p1_UB  = lift_OA(emptyList, currList, x3_p1_LB, x3_p1_UB,  lbList, ubList)

    #Next, lift part 2 to a space of (x₄, x₅, x₆, x₇, x₈)
    emptyList = [1,3] #because x4 and x6 are missing 
    currList = [2,4,5]

    l_x3_p2_LB, l_x3_p2_UB  = lift_OA(emptyList, currList, x3_p2_LB, x3_p2_UB,  lbList, ubList)

    #Next, lift part 3 to a space of (x₄, x₅, x₆, x₇, x₈)
    emptyList = [1,2] #because x4 and x5 are missing
    currList = [3,4,5]

    l_x3_p3_LB, l_x3_p3_UB  = lift_OA(emptyList, currList, x3_p3_LB, x3_p3_UB,  lbList, ubList)
    
    x3_p4_LB_i, x3_p4_UB_i = sumBounds(l_x3_p1_LB, l_x3_p1_UB, l_x3_p2_LB, l_x3_p2_UB, true)
    x3_p4_LB, x3_p4_UB = sumBounds(x3_p4_LB_i, x3_p4_UB_i, l_x3_p3_LB, l_x3_p3_UB, true)

    if sanityFlag
        @assert validBounds(:(x4*sin(x8) - x5*sin(x7)*cos(x8) - x6*cos(x7)*cos(x8)), [:x4, :x5, :x6, :x7, :x8], x3_p4_LB, x3_p4_UB) "Invalid bounds for x3"
    end

    #Finally, bounds for x3 have to be a function of x3. Lift bounds to space of (x₃, x₄, x₅, x₆, x₇, x₈)
    emptyList = [1] #Since x3 comes before x4, x5, x6, x7, x8
    currList = [2, 3, 4, 5, 6]
    lbList = lbs[3:8]
    ubList = ubs[3:8]

    x3_p4_LB_l, x3_p4_UB_l = lift_OA(emptyList, currList, x3_p4_LB, x3_p4_UB, lbList, ubList)

    return x3_p4_LB_l, x3_p4_UB_l
end

function bound_quadx4(Quad, plotFlag = false, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Bounding x₅*x₁₂ -x₆*x₁₁ - g*sin(x₈)
    #Part 1: f₁(x₅,x₁₂) = x₅*x₁₂
    #Sub-part 1: x₅
    x4_p1_sp1 = :(1*x)
    lb_x4_p1_sp1 = lbs[5]
    ub_x4_p1_sp1 = ubs[5]

    x4_p1_sp1_LB, x4_p1_sp1_UB = interpol_nd(bound_univariate(x4_p1_sp1, lb_x4_p1_sp1, ub_x4_p1_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₂
    x4_p1_sp2 = :(1*x)
    lb_x4_p1_sp2 = lbs[12]
    ub_x4_p1_sp2 = ubs[12]

    x4_p1_sp2_LB, x4_p1_sp2_UB = interpol_nd(bound_univariate(x4_p1_sp2, lb_x4_p1_sp2, ub_x4_p1_sp2, npoint=npoint)...)

    #Evaluate the product of the bounds 
    #First, lift sub part 1 to a space of (x₅,x₁₂)
    emptyList = [2] #x12 is missing 
    currList = [1] 
    lbList = [lbs[5], lbs[12]]
    ubList = [ubs[5], ubs[12]]

    l_x4_p1_sp1_LB, l_x4_p1_sp1_UB = lift_OA(emptyList, currList, x4_p1_sp1_LB, x4_p1_sp1_UB, lbList, ubList)

    #Next, lift sub part 2 to a space of (x₅,x₁₂)
    emptyList = [1] #x5 is missing
    currList = [2]

    l_x4_p1_sp2_LB, l_x4_p1_sp2_UB = lift_OA(emptyList, currList, x4_p1_sp2_LB, x4_p1_sp2_UB, lbList, ubList)

    #Now, multiply the lifted bounds
    x4_p1_LB, x4_p1_UB = prodBounds(l_x4_p1_sp1_LB, l_x4_p1_sp1_UB, l_x4_p1_sp2_LB, l_x4_p1_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x₅*x₁₂),[:x₅,:x₁₂], x4_p1_LB, x4_p1_UB)
    # end

    #Part 2: f₂(x₆, x₁₁) = x₆*x₁₁
    #Sub-part 1: x₆
    x4_p2_sp1 = :(1*x)
    lb_x4_p2_sp1 = lbs[6]
    ub_x4_p2_sp1 = ubs[6]

    x4_p2_sp1_LB, x4_p2_sp1_UB = interpol_nd(bound_univariate(x4_p2_sp1, lb_x4_p2_sp1, ub_x4_p2_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₁
    x4_p2_sp2 = :(1*x)
    lb_x4_p2_sp2 = lbs[11]
    ub_x4_p2_sp2 = ubs[11]

    x4_p2_sp2_LB, x4_p2_sp2_UB = interpol_nd(bound_univariate(x4_p2_sp2, lb_x4_p2_sp2, ub_x4_p2_sp2, npoint=npoint)...)

    #Evaluate the product of the bounds
    #First, lift sub part 1 to a space of (x₆,x₁₁)
    emptyList = [2] #x11 is missing
    currList = [1]
    lbList = [lbs[6], lbs[11]]
    ubList = [ubs[6], ubs[11]]

    l_x4_p2_sp1_LB, l_x4_p2_sp1_UB = lift_OA(emptyList, currList, x4_p2_sp1_LB, x4_p2_sp1_UB, lbList, ubList)

    #Next, lift sub part 2 to a space of (x₆,x₁₁)
    emptyList = [1] #x6 is missing
    currList = [2]

    l_x4_p2_sp2_LB, l_x4_p2_sp2_UB = lift_OA(emptyList, currList, x4_p2_sp2_LB, x4_p2_sp2_UB, lbList, ubList)

    #Now, multiply the lifted bounds
    x4_p2_LB, x4_p2_UB = prodBounds(l_x4_p2_sp1_LB, l_x4_p2_sp1_UB, l_x4_p2_sp2_LB, l_x4_p2_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x₆*x₁₁),[:x₆,:x₁₁], x4_p2_LB, x4_p2_UB)
    # end

    #Part 3: f₃(x₈) = -g*sin(x₈)
    x4_p3 = :($g*sin(x₈))
    lb_x4_p3 = lbs[8]
    ub_x4_p3 = ubs[8]

    if ub_x4_p3 - lb_x4_p3 < 1e-5
        lb_x4_p3 = lb_x4_p3 - 1e-5
        ub_x4_p3 = ub_x4_p3 + 1e-5
        lbs[8] = lb_x4_p3
        ubs[8] = ub_x4_p3
    end

    #NOTE:Artifically intervene
    x4_p3_LB, x4_p3_UB = interpol_nd(bound_univariate(x4_p3,lb_x4_p3,ub_x4_p3)...)
    
    #Recover complete bounds by combining each set of bounds 
    #Complete bounds are f(x₅,x₆,x₈,x₁₁,x₁₂) = x₅*x₁₂ -x₆*x₁₁ - g*sin(x₈)
    #First lift each part to a space of (x₅,x₆,x₈,x₁₁,x₁₂)
    #Lift part 1
    emptyList = [2,3,4] #x6,x8,x11 are missing
    currList = [1,5]
    lbList = [lbs[5], lbs[6], lbs[8], lbs[11], lbs[12]]
    ubList = [ubs[5], ubs[6], ubs[8], ubs[11], ubs[12]]

    l_x4_p1_LB, l_x4_p1_UB = lift_OA(emptyList, currList, x4_p1_LB, x4_p1_UB, lbList, ubList)

    #Lift part 2
    emptyList = [1,3,5] #x5,x8,x12 are missing
    currList = [2,4]

    l_x4_p2_LB, l_x4_p2_UB = lift_OA(emptyList, currList, x4_p2_LB, x4_p2_UB, lbList, ubList)

    #Lift part 3
    emptyList = [1,2,4,5] #x5,x6,x11,x12 are missing
    currList = [3]

    l_x4_p3_LB, l_x4_p3_UB = lift_OA(emptyList, currList, x4_p3_LB, x4_p3_UB, lbList, ubList)
    
    #Combine the lifted bounds
    x4_LB_i, x4_UB_i = sumBounds(l_x4_p1_LB, l_x4_p1_UB, l_x4_p2_LB, l_x4_p2_UB,true)
    x4_LB, x4_UB = sumBounds(x4_LB_i, x4_UB_i, l_x4_p3_LB, l_x4_p3_UB,true)

    if sanityFlag
        @assert validBounds(:(x₅*x₁₂ - x₆*x₁₁ - $g*sin(x₈)),[:x₅,:x₆,:x₈,:x₁₁,:x₁₂], x4_LB, x4_UB,true) "Invalid bounds for x4"
    end

    #Finally, bounds for x4 have to be a function of x4. Lift bounds to space of (x₄,x₅,x₆,x₈,x₁₁,x₁₂)

    emptyList = [1] #Since x4 comes before x5,x6,x8,x11,x12
    currList = [2,3,4,5,6]
    lbList = [lbs[4], lbs[5], lbs[6], lbs[8], lbs[11], lbs[12]]
    ubList = [ubs[4], ubs[5], ubs[6], ubs[8], ubs[11], ubs[12]]

    x4_LB_l, x4_UB_l = lift_OA(emptyList, currList, x4_LB, x4_UB, lbList, ubList)
    return x4_LB_l, x4_UB_l
end

function bound_quadx5(Quad, plotFlag = false, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Bounding x₆*x₁₀ - x₄*x₁₂ + g*sin(x₇)*cos(x₈)
    #Part 1: x₆*x₁₀
    #Sub-part 1: x₆
    x5_p1_sp1 = :(1*x)
    lb_x5_p1_sp1 = lbs[6]
    ub_x5_p1_sp1 = ubs[6]

    x5_p1_sp1_LB, x5_p1_sp1_UB = interpol_nd(bound_univariate(x5_p1_sp1, lb_x5_p1_sp1, ub_x5_p1_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₀
    x5_p1_sp2 = :(1*x)
    lb_x5_p1_sp2 = lbs[10]
    ub_x5_p1_sp2 = ubs[10]

    x5_p1_sp2_LB, x5_p1_sp2_UB = interpol_nd(bound_univariate(x5_p1_sp2, lb_x5_p1_sp2, ub_x5_p1_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₁₀ comes after x₆
    currList = [1]
    lbList = [lbs[6], lbs[10]]
    ubList = [ubs[6], ubs[10]]

    l_x5_p1_sp1_LB, l_x5_p1_sp1_UB = lift_OA(emptyList, currList, x5_p1_sp1_LB, x5_p1_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₆, x₁₀)
    emptyList = [1] #Since x₆ comes before x₁₀
    currList = [2]

    l_x5_p1_sp2_LB, l_x5_p1_sp2_UB = lift_OA(emptyList, currList, x5_p1_sp2_LB, x5_p1_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x5_p1_LB, x5_p1_UB = prodBounds(l_x5_p1_sp1_LB, l_x5_p1_sp1_UB, l_x5_p1_sp2_LB, l_x5_p1_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x6*x10), [:x6, :x10], x5_p1_LB, x5_p1_UB)
    # end


    #Part 2: x₄*x₁₂
    #Sub-part 1: x₄
    x5_p2_sp1 = :(1*x)
    lb_x5_p2_sp1 = lbs[4]
    ub_x5_p2_sp1 = ubs[4]

    x5_p2_sp1_LB, x5_p2_sp1_UB = interpol_nd(bound_univariate(x5_p2_sp1, lb_x5_p2_sp1, ub_x5_p2_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₂
    x5_p2_sp2 = :(1*x)
    lb_x5_p2_sp2 = lbs[12]
    ub_x5_p2_sp2 = ubs[12]

    x5_p2_sp2_LB, x5_p2_sp2_UB = interpol_nd(bound_univariate(x5_p2_sp2, lb_x5_p2_sp2, ub_x5_p2_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₁₂ comes after x₄
    currList = [1]
    lbList = [lbs[4], lbs[12]]
    ubList = [ubs[4], ubs[12]]

    l_x5_p2_sp1_LB, l_x5_p2_sp1_UB = lift_OA(emptyList, currList, x5_p2_sp1_LB, x5_p2_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₄, x₁₂)
    emptyList = [1] #Since x₄ comes before x₁₂
    currList = [2]

    l_x5_p2_sp2_LB, l_x5_p2_sp2_UB = lift_OA(emptyList, currList, x5_p2_sp2_LB, x5_p2_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x5_p2_LB, x5_p2_UB = prodBounds(l_x5_p2_sp1_LB, l_x5_p2_sp1_UB, l_x5_p2_sp2_LB, l_x5_p2_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x4*x12), [:x4, :x12], x5_p2_LB, x5_p2_UB)
    # end

    #Part 3: g*sin(x₇)*cos(x₈)
    #Sub-part 1: g*sin(x₇)
    x5_p3_sp1 = :($g*sin(x))
    lb_x5_p3_sp1 = lbs[7]
    ub_x5_p3_sp1 = ubs[7]

    if ub_x5_p3_sp1 - lb_x5_p3_sp1 < 1e-5
        lb_x5_p3_sp1 = lb_x5_p3_sp1 - 1e-5
        ub_x5_p3_sp1 = ub_x5_p3_sp1 + 1e-5
        lbs[7] = lb_x5_p3_sp1
        ubs[7] = ub_x5_p3_sp1
    end

    x5_p3_sp1_LB, x5_p3_sp1_UB = interpol_nd(bound_univariate(x5_p3_sp1, lb_x5_p3_sp1, ub_x5_p3_sp1)...)

    #Sub-part 2: cos(x₈)
    x5_p3_sp2 = :(cos(x))
    lb_x5_p3_sp2 = lbs[8]
    ub_x5_p3_sp2 = ubs[8]

    if ub_x5_p3_sp2 - lb_x5_p3_sp2 < 1e-5
        lb_x5_p3_sp2 = lb_x5_p3_sp2 - 1e-5
        ub_x5_p3_sp2 = ub_x5_p3_sp2 + 1e-5
        lbs[8] = lb_x5_p3_sp2
        ubs[8] = ub_x5_p3_sp2
    end

    x5_p3_sp2_LB, x5_p3_sp2_UB = interpol_nd(bound_univariate(x5_p3_sp2, lb_x5_p3_sp2, ub_x5_p3_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₈ comes after x₇
    currList = [1]
    lbList = [lbs[7], lbs[8]]
    ubList = [ubs[7], ubs[8]]

    l_x5_p3_sp1_LB, l_x5_p3_sp1_UB = lift_OA(emptyList, currList, x5_p3_sp1_LB, x5_p3_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₇, x₈)
    emptyList = [1] #Since x₇ comes before x₈
    currList = [2]

    l_x5_p3_sp2_LB, l_x5_p3_sp2_UB = lift_OA(emptyList, currList, x5_p3_sp2_LB, x5_p3_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x5_p3_LB, x5_p3_UB = prodBounds(l_x5_p3_sp1_LB, l_x5_p3_sp1_UB, l_x5_p3_sp2_LB, l_x5_p3_sp2_UB)

    # if sanityFlag
    #     validBounds(:($g*sin(x7)*cos(x8)), [:x7, :x8], x5_p3_LB, x5_p3_UB)
    # end

    #Now add the bounds to obtain f(x₄,x₆,x₇,x₈,x₁₀,x₁₂) = x₆*x₁₀ - x₄*x₁₂ + g*sin(x₇)*cos(x₈)
    #Lift the bounds to the same space
    #First lift part 1 to the space of (x₄,x₆,x₇,x₈,x₁₀,x₁₂)
    emptyList = [1,3,4,6] #Mising x4,x7,x8, and x12
    currList = [2,5] 
    lbList = [lbs[4], lbs[6], lbs[7], lbs[8], lbs[10], lbs[12]]
    ubList = [ubs[4], ubs[6], ubs[7], ubs[8], ubs[10], ubs[12]]

    l_x5_p1_LB, l_x5_p1_UB = lift_OA(emptyList, currList, x5_p1_LB, x5_p1_UB, lbList, ubList)

    #Now lift part 2 to the space of (x₄,x₆,x₇,x₈,x₁₀,x₁₂)
    emptyList = [2,3,4,5] #Missing x6, x7, x8, and x10
    currList = [1,6]

    l_x5_p2_LB, l_x5_p2_UB = lift_OA(emptyList, currList, x5_p2_LB, x5_p2_UB, lbList, ubList)

    #Now lift part 3 to the space of (x₄,x₆,x₇,x₈,x₁₀,x₁₂)
    emptyList = [1,2,5,6] #Missing x4, x6, x10, and x12
    currList = [3,4]

    l_x5_p3_LB, l_x5_p3_UB = lift_OA(emptyList, currList, x5_p3_LB, x5_p3_UB, lbList, ubList)

    #Now add the lifted bounds
    x5_LB_i, x5_UB_i = sumBounds(l_x5_p1_LB, l_x5_p1_UB, l_x5_p2_LB, l_x5_p2_UB,true)
    x5_LB, x5_UB = sumBounds(x5_LB_i, x5_UB_i, l_x5_p3_LB, l_x5_p3_UB,false)

    if sanityFlag
        @assert validBounds(:($g*sin(x7)*cos(x8) + x6*x10 - x4*x12), [:x4, :x6, :x7, :x8, :x10, :x12], x5_LB, x5_UB) "Invalid bounds for x5"
    end

    #Finally, bounds for x5 have to be a function of x5. Lift bounds to space of (x₄,x₅,x₆,x₇,x₈,x₁₀,x₁₂)
    emptyList = [2] #Since x5 comes after x4 but before x6, x7, x8, x10, x12
    currList = [1,3,4,5,6,7]
    lbList = [lbs[4], lbs[5], lbs[6], lbs[7], lbs[8], lbs[10], lbs[12]]
    ubList = [ubs[4], ubs[5], ubs[6], ubs[7], ubs[8], ubs[10], ubs[12]]

    x5_LB_l, x5_UB_l = lift_OA(emptyList, currList, x5_LB, x5_UB, lbList, ubList)
    return x5_LB_l, x5_UB_l
end

function bound_quadx6(Quad, plotFlag, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Bounding x₄*x₁₁ - x₅*x₁₀ + g*cos(x₇)*cos(x₈) - g
    #Part 1: x₄*x₁₁
    #Sub-part 1: x₄
    x6_p1_sp1 = :(1*x)
    lb_x6_p1_sp1 = lbs[4]
    ub_x6_p1_sp1 = ubs[4]

    x6_p1_sp1_LB, x6_p1_sp1_UB = interpol_nd(bound_univariate(x6_p1_sp1, lb_x6_p1_sp1, ub_x6_p1_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₁
    x6_p1_sp2 = :(1*x)
    lb_x6_p1_sp2 = lbs[11]
    ub_x6_p1_sp2 = ubs[11]

    x6_p1_sp2_LB, x6_p1_sp2_UB = interpol_nd(bound_univariate(x6_p1_sp2, lb_x6_p1_sp2, ub_x6_p1_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₁₁ comes after x₄
    currList = [1]
    lbList = [lbs[4], lbs[11]]
    ubList = [ubs[4], ubs[11]]

    l_x6_p1_sp1_LB, l_x6_p1_sp1_UB = lift_OA(emptyList, currList, x6_p1_sp1_LB, x6_p1_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₄, x₁₁)
    emptyList = [1] #Since x₄ comes before x₁₁
    currList = [2]

    l_x6_p1_sp2_LB, l_x6_p1_sp2_UB = lift_OA(emptyList, currList, x6_p1_sp2_LB, x6_p1_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x6_p1_LB, x6_p1_UB = prodBounds(l_x6_p1_sp1_LB, l_x6_p1_sp1_UB, l_x6_p1_sp2_LB, l_x6_p1_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x4*x11), [:x4, :x11], x6_p1_LB, x6_p1_UB)
    # end

    #Part 2: x₅*x₁₀
    #Sub-part 1: x₅
    x6_p2_sp1 = :(1*x)
    lb_x6_p2_sp1 = lbs[5]
    ub_x6_p2_sp1 = ubs[5]

    x6_p2_sp1_LB, x6_p2_sp1_UB = interpol_nd(bound_univariate(x6_p2_sp1, lb_x6_p2_sp1, ub_x6_p2_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₀
    x6_p2_sp2 = :(1*x)
    lb_x6_p2_sp2 = lbs[10]
    ub_x6_p2_sp2 = ubs[10]

    x6_p2_sp2_LB, x6_p2_sp2_UB = interpol_nd(bound_univariate(x6_p2_sp2, lb_x6_p2_sp2, ub_x6_p2_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₁₀ comes after x₅
    currList = [1]
    lbList = [lbs[5], lbs[10]]
    ubList = [ubs[5], ubs[10]]

    l_x6_p2_sp1_LB, l_x6_p2_sp1_UB = lift_OA(emptyList, currList, x6_p2_sp1_LB, x6_p2_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₅, x₁₀)
    emptyList = [1] #Since x₅ comes before x₁₀
    currList = [2]

    l_x6_p2_sp2_LB, l_x6_p2_sp2_UB = lift_OA(emptyList, currList, x6_p2_sp2_LB, x6_p2_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x6_p2_LB, x6_p2_UB = prodBounds(l_x6_p2_sp1_LB, l_x6_p2_sp1_UB, l_x6_p2_sp2_LB, l_x6_p2_sp2_UB)

    # if sanityFlag
    #     validBounds(:(x5*x10), [:x5, :x10], x6_p2_LB, x6_p2_UB)
    # end

    #Part 3: g*cos(x₇)*cos(x₈) - g
    #Sub-part 1: g*cos(x₇)
    x6_p3_sp1 = :($g*cos(x))
    lb_x6_p3_sp1 = lbs[7]
    ub_x6_p3_sp1 = ubs[7]

    if ub_x6_p3_sp1 - lb_x6_p3_sp1 < 1e-5
        lb_x6_p3_sp1 = lb_x6_p3_sp1 - 1e-5
        ub_x6_p3_sp1 = ub_x6_p3_sp1 + 1e-5
        lbs[7] = lb_x6_p3_sp1
        ubs[7] = ub_x6_p3_sp1
    end

    x6_p3_sp1_LB, x6_p3_sp1_UB = interpol_nd(bound_univariate(x6_p3_sp1, lb_x6_p3_sp1, ub_x6_p3_sp1, npoint=npoint)...)

    #Sub-part 2: cos(x₈)
    x6_p3_sp2 = :(cos(x))
    lb_x6_p3_sp2 = lbs[8]
    ub_x6_p3_sp2 = ubs[8]

    if ub_x6_p3_sp2 - lb_x6_p3_sp2 < 1e-5
        lb_x6_p3_sp2 = lb_x6_p3_sp2 - 1e-5
        ub_x6_p3_sp2 = ub_x6_p3_sp2 + 1e-5
        lbs[8] = lb_x6_p3_sp2
        ubs[8] = ub_x6_p3_sp2
    end

    x6_p3_sp2_LB, x6_p3_sp2_UB = interpol_nd(bound_univariate(x6_p3_sp2, lb_x6_p3_sp2, ub_x6_p3_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Since x₈ comes after x₇
    currList = [1]
    lbList = [lbs[7], lbs[8]]
    ubList = [ubs[7], ubs[8]]

    l_x6_p3_sp1_LB, l_x6_p3_sp1_UB = lift_OA(emptyList, currList, x6_p3_sp1_LB, x6_p3_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₇, x₈)
    emptyList = [1] #Since x₇ comes before x₈
    currList = [2]

    l_x6_p3_sp2_LB, l_x6_p3_sp2_UB = lift_OA(emptyList, currList, x6_p3_sp2_LB, x6_p3_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x6_p3_LB, x6_p3_UB = prodBounds(l_x6_p3_sp1_LB, l_x6_p3_sp1_UB, l_x6_p3_sp2_LB, l_x6_p3_sp2_UB)

    #Subtract g
    x6_p3_LB = [(tup[1:end-1]..., tup[end] - g) for tup in x6_p3_LB]
    x6_p3_UB = [(tup[1:end-1]..., tup[end] - g) for tup in x6_p3_UB]

    # if sanityFlag
    #     validBounds(:($g*cos(x7)*cos(x8) - $g), [:x7, :x8], x6_p3_LB, x6_p3_UB)
    # end

    #Now add the bounds to recover f(x₄, x₅, x₇, x₈, x₁₀, x₁₁)
    #Lift the bounds to the same space
    #First lift part 1 to the space of (x₄, x₅, x₇, x₈, x₁₀, x₁₁)
    emptyList = [2,3,4,5] #Missing x₅,x₇, x₈ and x₁₀
    currList = [1,6]
    lbList = [lbs[4], lbs[5], lbs[7], lbs[8], lbs[10], lbs[11]]
    ubList = [ubs[4], ubs[5], ubs[7], ubs[8], ubs[10], ubs[11]]

    l_x6_p1_LB, l_x6_p1_UB = lift_OA(emptyList, currList, x6_p1_LB, x6_p1_UB, lbList, ubList)

    #Now lift part 2 to the space of (x₄, x₅, x₇, x₈, x₁₀, x₁₁)
    emptyList = [1,3,4,6] #Missing x₄,x₇, x₈ and x₁₁
    currList = [2,5]

    l_x6_p2_LB, l_x6_p2_UB = lift_OA(emptyList, currList, x6_p2_LB, x6_p2_UB, lbList, ubList)

    #Now lift part 3 to the space of (x₄, x₅, x₇, x₈, x₁₀, x₁₁)
    emptyList = [1,2,5,6] #Missing x₄,x₅, x₁₀ and x₁₁
    currList = [3,4]

    l_x6_p3_LB, l_x6_p3_UB = lift_OA(emptyList, currList, x6_p3_LB, x6_p3_UB, lbList, ubList)

    #Now add the lifted bounds
    x6_LB_i, x6_UB_i = sumBounds(l_x6_p1_LB, l_x6_p1_UB, l_x6_p2_LB, l_x6_p2_UB,true)
    x6_LB, x6_UB = sumBounds(x6_LB_i, x6_UB_i, l_x6_p3_LB, l_x6_p3_UB,false)

    if sanityFlag
        @assert validBounds(:($g*cos(x7)*cos(x8) - $g + x4*x11 - x5*x10), [:x4, :x5, :x7, :x8, :x10, :x11], x6_LB, x6_UB) "Invalid bounds for x6"
    end

    #Finally, bounds for x6 have to be a function of x6. Lift bounds to space of (x₄,x₅,x₆,x₇,x₈,x₁₀,x₁₁)
    emptyList = [3] #Since x6 comes after x4 and x5 but before x7, x8, x10, x11
    currList = [1,2,4,5,6,7]
    lbList = [lbs[4], lbs[5], lbs[6], lbs[7], lbs[8], lbs[10], lbs[11]]
    ubList = [ubs[4], ubs[5], ubs[6], ubs[7], ubs[8], ubs[10], ubs[11]]

    x6_LB_l, x6_UB_l = lift_OA(emptyList, currList, x6_LB, x6_UB, lbList, ubList)
    return x6_LB_l, x6_UB_l
end

function bound_quadx7(Quad, plotFlag, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)
    #Bounding x₁₀ + sin(x₇)*tan(x₈)*x₁₁ + cos(x₇)*tan(x₈)*x₁₂
    #Part 1: x₁₀
    x7_p1 = :(1*x)
    lb_x7_p1 = lbs[10]
    ub_x7_p1 = ubs[10]

    x7_p1_LB, x7_p1_UB = interpol_nd(bound_univariate(x7_p1, lb_x7_p1, ub_x7_p1, npoint=npoint)...)

    #Part 2: sin(x₇)*tan(x₈)*x₁₁
    #Sub-part 1: sin(x₇)
    x7_p2_sp1 = :(sin(x))
    lb_x7_p2_sp1 = lbs[7]
    ub_x7_p2_sp1 = ubs[7]

    if ub_x7_p2_sp1 - lb_x7_p2_sp1 < 1e-5
        lb_x7_p2_sp1 = lb_x7_p2_sp1 - 1e-5
        ub_x7_p2_sp1 = ub_x7_p2_sp1 + 1e-5
        lbs[7] = lb_x7_p2_sp1
        ubs[7] = ub_x7_p2_sp1
    end

    x7_p2_sp1_LB, x7_p2_sp1_UB = interpol_nd(bound_univariate(x7_p2_sp1, lb_x7_p2_sp1, ub_x7_p2_sp1, npoint=npoint)...)

    #Sub-part 2: tan(x₈)
    x7_p2_sp2 = :(tan(x))
    lb_x7_p2_sp2 = lbs[8]
    ub_x7_p2_sp2 = ubs[8]

    if ub_x7_p2_sp2 - lb_x7_p2_sp2 < 1e-5
        lb_x7_p2_sp2 = lb_x7_p2_sp2 - 1e-5
        ub_x7_p2_sp2 = ub_x7_p2_sp2 + 1e-5
        lbs[8] = lb_x7_p2_sp2
        ubs[8] = ub_x7_p2_sp2
    end

    x7_p2_sp2_LB, x7_p2_sp2_UB = interpol_nd(bound_univariate(x7_p2_sp2, lb_x7_p2_sp2, ub_x7_p2_sp2, npoint=npoint)...)

    #Sub-part 3: x₁₁
    x7_p2_sp3 = :(1*x)
    lb_x7_p2_sp3 = lbs[11]
    ub_x7_p2_sp3 = ubs[11]

    x7_p2_sp3_LB, x7_p2_sp3_UB = interpol_nd(bound_univariate(x7_p2_sp3, lb_x7_p2_sp3, ub_x7_p2_sp3, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2,3] #Sub part 1 missing x₈ and x₁₁ 
    currList = [1]
    lbList = [lbs[7], lbs[8], lbs[11]]
    ubList = [ubs[7], ubs[8], ubs[11]]

    l_x7_p2_sp1_LB, l_x7_p2_sp1_UB = lift_OA(emptyList, currList, x7_p2_sp1_LB, x7_p2_sp1_UB, lbList, ubList)

    #Lift sub part 2 to the same space
    emptyList = [1,3] #Sub part 2 missing x₇ and x₁₁
    currList = [2]

    l_x7_p2_sp2_LB, l_x7_p2_sp2_UB = lift_OA(emptyList, currList, x7_p2_sp2_LB, x7_p2_sp2_UB, lbList, ubList)

    #Lift sub part 3 to the same space
    emptyList = [1,2] #Sub part 3 missing x₇ and x₈
    currList = [3]

    l_x7_p2_sp3_LB, l_x7_p2_sp3_UB = lift_OA(emptyList, currList, x7_p2_sp3_LB, x7_p2_sp3_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x7_p2_LB_i, x7_p2_UB_i = prodBounds(l_x7_p2_sp1_LB, l_x7_p2_sp1_UB, l_x7_p2_sp2_LB, l_x7_p2_sp2_UB)
    x7_p2_LB, x7_p2_UB = prodBounds(x7_p2_LB_i, x7_p2_UB_i, l_x7_p2_sp3_LB, l_x7_p2_sp3_UB)

    # if sanityFlag
    #     validBounds(:((x7)*tan(x8)*x11), [:x7, :x8, :x11], x7_p2_LB, x7_p2_UB)
    # end

    #Part 3: cos(x₇)*tan(x₈)*x₁₂
    #Sub-part 1: cos(x₇)
    x7_p3_sp1 = :(cos(x))
    lb_x7_p3_sp1 = lbs[7]
    ub_x7_p3_sp1 = ubs[7]

    if ub_x7_p3_sp1 - lb_x7_p3_sp1 < 1e-5
        lb_x7_p3_sp1 = lb_x7_p3_sp1 - 1e-5
        ub_x7_p3_sp1 = ub_x7_p3_sp1 + 1e-5
        lbs[7] = lb_x7_p3_sp1
        ubs[7] = ub_x7_p3_sp1
    end

    x7_p3_sp1_LB, x7_p3_sp1_UB = interpol_nd(bound_univariate(x7_p3_sp1, lb_x7_p3_sp1, ub_x7_p3_sp1, npoint=npoint)...)

    #Sub-part 2: tan(x₈)
    x7_p3_sp2 = :(tan(x))
    lb_x7_p3_sp2 = lbs[8]
    ub_x7_p3_sp2 = ubs[8]

    if ub_x7_p3_sp2 - lb_x7_p3_sp2 < 1e-5
        lb_x7_p3_sp2 = lb_x7_p3_sp2 - 1e-5
        ub_x7_p3_sp2 = ub_x7_p3_sp2 + 1e-5
        lbs[8] = lb_x7_p3_sp2
        ubs[8] = ub_x7_p3_sp2
    end

    x7_p3_sp2_LB, x7_p3_sp2_UB = interpol_nd(bound_univariate(x7_p3_sp2, lb_x7_p3_sp2, ub_x7_p3_sp2, npoint=npoint)...)

    #Sub-part 3: x₁₂
    x7_p3_sp3 = :(1*x)
    lb_x7_p3_sp3 = lbs[12]
    ub_x7_p3_sp3 = ubs[12]

    x7_p3_sp3_LB, x7_p3_sp3_UB = interpol_nd(bound_univariate(x7_p3_sp3, lb_x7_p3_sp3, ub_x7_p3_sp3, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2,3] #Sub part 1 missing x₈ and x₁₂
    currList = [1]
    lbList = [lbs[7], lbs[8], lbs[12]]
    ubList = [ubs[7], ubs[8], ubs[12]]

    l_x7_p3_sp1_LB, l_x7_p3_sp1_UB = lift_OA(emptyList, currList, x7_p3_sp1_LB, x7_p3_sp1_UB, lbList, ubList)

    #Lift sub part 2 to the same space
    emptyList = [1,3] #Sub part 2 missing x₇ and x₁₂
    currList = [2]

    l_x7_p3_sp2_LB, l_x7_p3_sp2_UB = lift_OA(emptyList, currList, x7_p3_sp2_LB, x7_p3_sp2_UB, lbList, ubList)

    #Lift sub part 3 to the same space
    emptyList = [1,2] #Sub part 3 missing x₇ and x₈
    currList = [3]

    l_x7_p3_sp3_LB, l_x7_p3_sp3_UB = lift_OA(emptyList, currList, x7_p3_sp3_LB, x7_p3_sp3_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x7_p3_LB_i, x7_p3_UB_i = prodBounds(l_x7_p3_sp1_LB, l_x7_p3_sp1_UB, l_x7_p3_sp2_LB, l_x7_p3_sp2_UB)
    x7_p3_LB, x7_p3_UB = prodBounds(x7_p3_LB_i, x7_p3_UB_i, l_x7_p3_sp3_LB, l_x7_p3_sp3_UB)

    # if sanityFlag
    #     validBounds(:(cos(x7)*tan(x8)*x12), [:x7, :x8, :x12], x7_p3_LB, x7_p3_UB)
    # end

    #Now add the bounds to obtain f(x₇,x₈,x₁₀,x₁₁,x₁₂) = x₁₀ + sin(x₇)*tan(x₈)*x₁₁ + cos(x₇)*tan(x₈)*x₁₂
    #Lift the bounds to the same space
    #First lift part 1 to the space of (x₇,x₈,x₁₀,x₁₁,x₁₂)
    emptyList = [1,2,4,5] #Missing x₇, x₈, x₁₁, and x₁₂
    currList = [3]
    lbList = [lbs[7], lbs[8], lbs[10], lbs[11], lbs[12]]
    ubList = [ubs[7], ubs[8], ubs[10], ubs[11], ubs[12]]

    l_x7_p1_LB, l_x7_p1_UB = lift_OA(emptyList, currList, x7_p1_LB, x7_p1_UB, lbList, ubList)

    #Lift part 2 to the space of (x₇,x₈,x₁₀,x₁₁,x₁₂)
    emptyList = [3,5] #Missing x₁₀ and x₁₂
    currList = [1,2,4]

    l_x7_p2_LB, l_x7_p2_UB = lift_OA(emptyList, currList, x7_p2_LB, x7_p2_UB, lbList, ubList)

    #Lift part 3 to the space of (x₇,x₈,x₁₀,x₁₁,x₁₂)
    emptyList = [3,4] #Missing x₁₀ and x₁₁
    currList = [1,2,5]

    l_x7_p3_LB, l_x7_p3_UB = lift_OA(emptyList, currList, x7_p3_LB, x7_p3_UB, lbList, ubList)

    #Now add the bounds
    x7_LB_i, x7_UB_i = sumBounds(l_x7_p1_LB, l_x7_p1_UB, l_x7_p2_LB, l_x7_p2_UB,false)
    x7_LB, x7_UB = sumBounds(x7_LB_i, x7_UB_i, l_x7_p3_LB, l_x7_p3_UB,false)

    if sanityFlag
        @assert validBounds(:((x10) + sin(x7)*tan(x8)*x11 + cos(x7)*tan(x8)*x12), [:x7, :x8, :x10, :x11, :x12], x7_LB, x7_UB) "Invalid bounds for x7"
    end

    #bounds for x7 include x7. We are done
    return x7_LB, x7_UB
end

function bound_quadx8(Quad, plotFlag, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Bounding cos(x₇)*x₁₁ - sin(x₇)*x₁₂
    #Part 1: cos(x₇)*x₁₁
    #Sub-part 1: cos(x₇)
    x8_p1_sp1 = :(cos(x))
    lb_x8_p1_sp1 = lbs[7]
    ub_x8_p1_sp1 = ubs[7]

    if ub_x8_p1_sp1 - lb_x8_p1_sp1 < 1e-5
        lb_x8_p1_sp1 = lb_x8_p1_sp1 - 1e-5
        ub_x8_p1_sp1 = ub_x8_p1_sp1 + 1e-5
        lbs[7] = lb_x8_p1_sp1
        ubs[7] = ub_x8_p1_sp1
    end

    x8_p1_sp1_LB, x8_p1_sp1_UB = interpol_nd(bound_univariate(x8_p1_sp1, lb_x8_p1_sp1, ub_x8_p1_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₁
    x8_p1_sp2 = :(1*x)
    lb_x8_p1_sp2 = lbs[11]
    ub_x8_p1_sp2 = ubs[11]

    x8_p1_sp2_LB, x8_p1_sp2_UB = interpol_nd(bound_univariate(x8_p1_sp2, lb_x8_p1_sp2, ub_x8_p1_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Sub part 1 missing x₁₁
    currList = [1]
    lbList = [lbs[7], lbs[11]]
    ubList = [ubs[7], ubs[11]]

    l_x8_p1_sp1_LB, l_x8_p1_sp1_UB = lift_OA(emptyList, currList, x8_p1_sp1_LB, x8_p1_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₇, x₁₁)
    emptyList = [1] #Sub part 2 missing x₇
    currList = [2]

    l_x8_p1_sp2_LB, l_x8_p1_sp2_UB = lift_OA(emptyList, currList, x8_p1_sp2_LB, x8_p1_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x8_p1_LB, x8_p1_UB = prodBounds(l_x8_p1_sp1_LB, l_x8_p1_sp1_UB, l_x8_p1_sp2_LB, l_x8_p1_sp2_UB)

    # if sanityFlag
    #     validBounds(:(cos(x7)*x11), [:x7, :x11], x8_p1_LB, x8_p1_UB)
    # end

    #Part 2: sin(x₇)*x₁₂
    #Sub-part 1: sin(x₇)
    x8_p2_sp1 = :(sin(x))
    lb_x8_p2_sp1 = lbs[7]
    ub_x8_p2_sp1 = ubs[7]

    if ub_x8_p2_sp1 - lb_x8_p2_sp1 < 1e-5
        lb_x8_p2_sp1 = lb_x8_p2_sp1 - 1e-5
        ub_x8_p2_sp1 = ub_x8_p2_sp1 + 1e-5
        lbs[7] = lb_x8_p2_sp1
        ubs[7] = ub_x8_p2_sp1
    end

    x8_p2_sp1_LB, x8_p2_sp1_UB = interpol_nd(bound_univariate(x8_p2_sp1, lb_x8_p2_sp1, ub_x8_p2_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₂
    x8_p2_sp2 = :(1*x)
    lb_x8_p2_sp2 = lbs[12]
    ub_x8_p2_sp2 = ubs[12]

    x8_p2_sp2_LB, x8_p2_sp2_UB = interpol_nd(bound_univariate(x8_p2_sp2, lb_x8_p2_sp2, ub_x8_p2_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Sub part 1 missing x₁₂
    currList = [1]
    lbList = [lbs[7], lbs[12]]
    ubList = [ubs[7], ubs[12]]

    l_x8_p2_sp1_LB, l_x8_p2_sp1_UB = lift_OA(emptyList, currList, x8_p2_sp1_LB, x8_p2_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₇, x₁₂)
    emptyList = [1] #Sub part 2 missing x₇
    currList = [2]

    l_x8_p2_sp2_LB, l_x8_p2_sp2_UB = lift_OA(emptyList, currList, x8_p2_sp2_LB, x8_p2_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x8_p2_LB, x8_p2_UB = prodBounds(l_x8_p2_sp1_LB, l_x8_p2_sp1_UB, l_x8_p2_sp2_LB, l_x8_p2_sp2_UB)

    # if sanityFlag
    #     validBounds(:(sin(x7)*x12), [:x7, :x12], x8_p2_LB, x8_p2_UB)
    # end

    #Now add the bounds to obtain f(x₇,x₁₁,x₁₂) = cos(x₇)*x₁₁ - sin(x₇)*x₁₂
    #Lift the bounds to the same space
    #First lift part 1 to the space of (x₇,x₁₁,x₁₂)
    emptyList = [3] #Missing x₁₂
    currList = [1,2]
    lbList = [lbs[7], lbs[11], lbs[12]]
    ubList = [ubs[7], ubs[11], ubs[12]]

    l_x8_p1_LB, l_x8_p1_UB = lift_OA(emptyList, currList, x8_p1_LB, x8_p1_UB, lbList, ubList)

    #Now lift part 2 to the space of (x₇,x₁₁,x₁₂)
    emptyList = [2] #Missing x₁₁
    currList = [1,3]

    l_x8_p2_LB, l_x8_p2_UB = lift_OA(emptyList, currList, x8_p2_LB, x8_p2_UB, lbList, ubList)

    #Now add the lifted bounds
    x8_LB, x8_UB = sumBounds(l_x8_p1_LB, l_x8_p1_UB, l_x8_p2_LB, l_x8_p2_UB,true)

    if sanityFlag
        @assert validBounds(:(cos(x7)*x11 - sin(x7)*x12), [:x7, :x11, :x12], x8_LB, x8_UB) "Invalid bounds for x8"
    end

    #Finally, bounds for x8 have to be a function of x8. Lift bounds to space of (x₇,x₈,x₁₁,x₁₂)
    emptyList = [2] #Since x8 comes after x7 but before x11 and x12
    currList = [1,3,4]
    lbList = [lbs[7], lbs[8], lbs[11], lbs[12]]
    ubList = [ubs[7], ubs[8], ubs[11], ubs[12]]

    x8_LB_l, x8_UB_l = lift_OA(emptyList, currList, x8_LB, x8_UB, lbList, ubList)

    return x8_LB_l, x8_UB_l
end

function bound_quadx9(Quad, plotFlag, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Bounding (sin(x₇)/cos(x₈))*x₁₁ - (cos(x₇)/cos(x₈))*x₁₂
    #Part 1: (sin(x₇)/cos(x₈))*x₁₁
    #Use Minkowski sum approach for this to avoid interval division
    #NOTE: Can't avoid division babes 
    #Sub-part 1: sin(x₇)
    x9_p1_sp1 = :(sin(x7))
    lb_x9_p1_sp1 = lbs[7]
    ub_x9_p1_sp1 = ubs[7]

    if ub_x9_p1_sp1 - lb_x9_p1_sp1 < 1e-5
        lb_x9_p1_sp1 = lb_x9_p1_sp1 - 1e-5
        ub_x9_p1_sp1 = ub_x9_p1_sp1 + 1e-5
        lbs[7] = lb_x9_p1_sp1
        ubs[7] = ub_x9_p1_sp1
    end

    x9_p1_sp1_LB, x9_p1_sp1_UB = interpol_nd(bound_univariate(x9_p1_sp1, lb_x9_p1_sp1, ub_x9_p1_sp1, npoint=npoint)...)

    #Sub-part 2: cos(x₈)
    x9_p1_sp2 = :(cos(x8))
    lb_x9_p1_sp2 = lbs[8]
    ub_x9_p1_sp2 = ubs[8]

    if ub_x9_p1_sp2 - lb_x9_p1_sp2 < 1e-5
        lb_x9_p1_sp2 = lb_x9_p1_sp2 - 1e-5
        ub_x9_p1_sp2 = ub_x9_p1_sp2 + 1e-5
        lbs[8] = lb_x9_p1_sp2
        ubs[8] = ub_x9_p1_sp2
    end

    x9_p1_sp2_LB, x9_p1_sp2_UB = interpol_nd(bound_univariate(x9_p1_sp2, lb_x9_p1_sp2, ub_x9_p1_sp2, npoint=npoint)...)

    #Sub-part 3: x₁₁
    x9_p1_sp3 = :(1*x)
    lb_x9_p1_sp3 = lbs[11]
    ub_x9_p1_sp3 = ubs[11]

    x9_p1_sp3_LB, x9_p1_sp3_UB = interpol_nd(bound_univariate(x9_p1_sp3, lb_x9_p1_sp3, ub_x9_p1_sp3, npoint=npoint)...)

    #Lift bounds of sp1 and sp2 to the same space of (x₇, x₈)
    emptyList = [2] #Sub part 1 missing x₈
    currList = [1]
    lbList = [lbs[7], lbs[8]]
    ubList = [ubs[7], ubs[8]]

    l_x9_p1_sp1_LB, l_x9_p1_sp1_UB = lift_OA(emptyList, currList, x9_p1_sp1_LB,x9_p1_sp1_UB, lbList, ubList)

    #Lift sub part 2 to space of (x₇, x₈, x₁₁)
    emptyList = [1] #Sub part 2 missing x₇ and x₁₁
    currList = [2]

    l_x9_p1_sp2_LB, l_x9_p1_sp2_UB = lift_OA(emptyList, currList, x9_p1_sp2_LB,x9_p1_sp2_UB, lbList, ubList)

    #Compute the division of the bounds to obtain sin(x₇)/cos(x₈)
    x9_p1_sp4_LB, x9_p1_sp4_UB = divBounds(l_x9_p1_sp1_LB, l_x9_p1_sp1_UB, l_x9_p1_sp2_LB, l_x9_p1_sp2_UB)
    
    # if sanityFlag
    #     validBounds(:(sin(x7)/cos(x8)), [:x7, :x8], x9_p1_sp4_LB, x9_p1_sp4_UB)
    # end

    #Lift the bounds of x₁₁ to the space of (x₇, x₈, x₁₁)
    emptyList = [1, 2] #x₁₁ missing x₇ and x₈
    currList = [3]
    lbList = [lbs[7], lbs[8], lbs[11]]
    ubList = [ubs[7], ubs[8], ubs[11]]

    l_x9_p1_sp3_LB, l_x9_p1_sp3_UB = lift_OA(emptyList, currList,x9_p1_sp3_LB, x9_p1_sp3_UB, lbList, ubList)

    #Lift the bounds of sin(x₇)/cos(x₈) to the space of (x₇, x₈, x₁₁)
    emptyList = [3] #sin(x₇)/cos(x₈) missing x₁₁
    currList = [1, 2]

    l_x9_p1_sp4_LB, l_x9_p1_sp4_UB = lift_OA(emptyList, currList, x9_p1_sp4_LB, x9_p1_sp4_UB, lbList, ubList)
    #Multiply the bounds of sin(x₇)/cos(x₈) and x₁₁
    x9_p1_LB, x9_p1_UB = prodBounds(l_x9_p1_sp3_LB, l_x9_p1_sp3_UB, l_x9_p1_sp4_LB, l_x9_p1_sp4_UB)

    # if sanityFlag
    #     validBounds(:((sin(x7)/cos(x8))*x11), [:x7, :x8, :x11], x9_p1_LB, x9_p1_UB)
    # end

    #Part 2: (cos(x₇)/cos(x₈))*x₁₂
    #Sub-part 1: cos(x₇)
    x9_p2_sp1 = :(cos(x7))
    lb_x9_p2_sp1 = lbs[7]
    ub_x9_p2_sp1 = ubs[7]

    if ub_x9_p2_sp1 - lb_x9_p2_sp1 < 1e-5
        lb_x9_p2_sp1 = lb_x9_p2_sp1 - 1e-5
        ub_x9_p2_sp1 = ub_x9_p2_sp1 + 1e-5
        lbs[7] = lb_x9_p2_sp1
        ubs[7] = ub_x9_p2_sp1
    end

    x9_p2_sp1_LB, x9_p2_sp1_UB = interpol_nd(bound_univariate(x9_p2_sp1, lb_x9_p2_sp1, ub_x9_p2_sp1, npoint=npoint)...)

    #Sub-part 2: cos(x₈)
    x9_p2_sp2 = :(cos(x8))
    lb_x9_p2_sp2 = lbs[8]
    ub_x9_p2_sp2 = ubs[8]

    if ub_x9_p2_sp2 - lb_x9_p2_sp2 < 1e-5
        lb_x9_p2_sp2 = lb_x9_p2_sp2 - 1e-5
        ub_x9_p2_sp2 = ub_x9_p2_sp2 + 1e-5
        lbs[8] = lb_x9_p2_sp2
        ubs[8] = ub_x9_p2_sp2
    end

    x9_p2_sp2_LB, x9_p2_sp2_UB = interpol_nd(bound_univariate(x9_p2_sp2, lb_x9_p2_sp2, ub_x9_p2_sp2, npoint=npoint)...)

    #Sub-part 3: x₁₂
    x9_p2_sp3 = :(1*x)
    lb_x9_p2_sp3 = lbs[12]
    ub_x9_p2_sp3 = ubs[12]

    x9_p2_sp3_LB, x9_p2_sp3_UB = interpol_nd(bound_univariate(x9_p2_sp3, lb_x9_p2_sp3, ub_x9_p2_sp3, npoint=npoint)...)

    #Lift bounds of sp1 and sp2 to the same space of (x₇, x₈)
    emptyList = [2] #Sub part 1 missing x₈
    currList = [1]
    lbList = [lbs[7], lbs[8]]
    ubList = [ubs[7], ubs[8]]

    l_x9_p2_sp1_LB, l_x9_p2_sp1_UB = lift_OA(emptyList, currList, x9_p2_sp1_LB,x9_p2_sp1_UB, lbList, ubList)

    #Lift sub part 2 to space of (x₇, x₈)
    emptyList = [1] #Sub part 2 missing x₇
    currList = [2]

    l_x9_p2_sp2_LB, l_x9_p2_sp2_UB = lift_OA(emptyList, currList, x9_p2_sp2_LB,x9_p2_sp2_UB, lbList, ubList)

    #Compute the division of the bounds to obtain cos(x₇)/cos(x₈)
    x9_p2_sp4_LB, x9_p2_sp4_UB = divBounds(l_x9_p2_sp1_LB, l_x9_p2_sp1_UB, l_x9_p2_sp2_LB, l_x9_p2_sp2_UB)

    # if sanityFlag
    #     validBounds(:(cos(x7)/cos(x8)), [:x7, :x8], x9_p2_sp4_LB, x9_p2_sp4_UB)
    # end

    #Lift the bounds of x₁₂ to the space of (x₇, x₈, x₁₂)
    emptyList = [1, 2] #x₁₂ missing x₇ and x₈
    currList = [3]
    lbList = [lbs[7], lbs[8], lbs[12]]
    ubList = [ubs[7], ubs[8], ubs[12]]

    l_x9_p2_sp3_LB, l_x9_p2_sp3_UB = lift_OA(emptyList, currList,x9_p2_sp3_LB, x9_p2_sp3_UB, lbList, ubList)

    #Lift the bounds of cos(x₇)/cos(x₈) to the space of (x₇, x₈, x₁₂)
    emptyList = [3] #cos(x₇)/cos(x₈) missing x₁₂
    currList = [1, 2]

    l_x9_p2_sp4_LB, l_x9_p2_sp4_UB = lift_OA(emptyList, currList, x9_p2_sp4_LB, x9_p2_sp4_UB, lbList, ubList)

    #Multiply the bounds of cos(x₇)/cos(x₈) and x₁₂
    x9_p2_LB, x9_p2_UB = prodBounds(l_x9_p2_sp3_LB, l_x9_p2_sp3_UB, l_x9_p2_sp4_LB, l_x9_p2_sp4_UB)

    # if sanityFlag
    #     validBounds(:((cos(x7)/cos(x8))*x12), [:x7, :x8, :x12], x9_p2_LB, x9_p2_UB)
    # end

    #Subtract the bounds of part 1 and part 2
    #First lift the bounds of part 1 to the space of (x₇, x₈,x₁₁, x₁₂)
    emptyList = [4] #Part 1 missing x₁₂
    currList = [1, 2, 3]
    lbList = [lbs[7], lbs[8], lbs[11], lbs[12]]
    ubList = [ubs[7], ubs[8], ubs[11], ubs[12]]

    l_x9_p1_LB, l_x9_p1_UB = lift_OA(emptyList, currList, x9_p1_LB, x9_p1_UB, lbList, ubList)
    
    #Lift the bounds of part 2 to the space of (x₇, x₈,x₁₁, x₁₂)
    emptyList = [3] #Part 2 missing x₁₁
    currList = [1, 2, 4]

    l_x9_p2_LB, l_x9_p2_UB = lift_OA(emptyList, currList, x9_p2_LB, x9_p2_UB, lbList, ubList)

    x9_LB, x9_UB = sumBounds(l_x9_p1_LB, l_x9_p1_UB, l_x9_p2_LB, l_x9_p2_UB,true)

    if sanityFlag
        @assert validBounds(:((sin(x7)/cos(x8))*x11 - (cos(x7)/cos(x8))*x12), [:x7, :x8, :x11, :x12], x9_LB, x9_UB) "Invalid bounds for x9"
    end

    #Finally, bounds for x9 have to be a function of x9. Lift bounds to space of (x₇,x₈,x₉,x₁₁,x₁₂)
    emptyList = [3] #Since x9 comes after x7 and x8 but before x11 and x12
    currList = [1,2,4,5]
    lbList = [lbs[7], lbs[8], lbs[9], lbs[11], lbs[12]]
    ubList = [ubs[7], ubs[8], ubs[9], ubs[11], ubs[12]]

    x9_LB_l, x9_UB_l = lift_OA(emptyList, currList, x9_LB, x9_UB, lbList, ubList)
    
    return x9_LB_l, x9_UB_l
end

function bound_quadx10(Quad, plotFlag, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Bounding ((Jy - Jz)/Jx)*x₁₁*x₁₂
    #Sub-part 1: (Jy - Jz)/Jx * x₁₁
    x10_p1_sp1 = :($((Jy - Jz)/Jx)*x) 
    lb_x10_p1_sp1 = lbs[11]
    ub_x10_p1_sp1 = ubs[11]

    x10_p1_sp1_LB, x10_p1_sp1_UB = interpol_nd(bound_univariate(x10_p1_sp1, lb_x10_p1_sp1, ub_x10_p1_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₂
    x10_p1_sp2 = :(1*x)
    lb_x10_p1_sp2 = lbs[12]
    ub_x10_p1_sp2 = ubs[12]

    x10_p1_sp2_LB, x10_p1_sp2_UB = interpol_nd(bound_univariate(x10_p1_sp2, lb_x10_p1_sp2, ub_x10_p1_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Sub part 1 missing x₁₂
    currList = [1]
    lbList = [lbs[11], lbs[12]]
    ubList = [ubs[11], ubs[12]]

    l_x10_p1_sp1_LB, l_x10_p1_sp1_UB = lift_OA(emptyList, currList, x10_p1_sp1_LB, x10_p1_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₁₁, x₁₂)
    emptyList = [1] #Sub part 2 missing x₁₁
    currList = [2]

    l_x10_p1_sp2_LB, l_x10_p1_sp2_UB = lift_OA(emptyList, currList, x10_p1_sp2_LB, x10_p1_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x10_p1_LB, x10_p1_UB = prodBounds(l_x10_p1_sp1_LB, l_x10_p1_sp1_UB, l_x10_p1_sp2_LB, l_x10_p1_sp2_UB)

    if sanityFlag
        @assert validBounds(:($((Jy - Jz)/Jx)*x11*x12), [:x11, :x12], x10_p1_LB, x10_p1_UB) "Invalid bounds for x10"
    end

    #Finally, bounds for x10 have to be a function of x10. Lift bounds to space of (x₁₀, x₁₁, x₁₂)
    emptyList = [1] #Since x10 comes before x11 and x12
    currList = [2,3]
    lbList = [lbs[10], lbs[11], lbs[12]]
    ubList = [ubs[10], ubs[11], ubs[12]]

    x10_LB, x10_UB = lift_OA(emptyList, currList, x10_p1_LB, x10_p1_UB, lbList, ubList)

    return x10_LB, x10_UB
end

function bound_quadx11(Quad, plotFlag, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Bounding ((Jz - Jx)/Jy)*x₁₀*x₁₂
    #Sub-part 1: (Jz - Jx)/Jy * x₁₀
    x11_p1_sp1 = :($((Jz - Jx)/Jy)*x)
    lb_x11_p1_sp1 = lbs[10]
    ub_x11_p1_sp1 = ubs[10]

    x11_p1_sp1_LB, x11_p1_sp1_UB = interpol_nd(bound_univariate(x11_p1_sp1, lb_x11_p1_sp1, ub_x11_p1_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₂
    x11_p1_sp2 = :(1*x)
    lb_x11_p1_sp2 = lbs[12]
    ub_x11_p1_sp2 = ubs[12]

    x11_p1_sp2_LB, x11_p1_sp2_UB = interpol_nd(bound_univariate(x11_p1_sp2, lb_x11_p1_sp2, ub_x11_p1_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Sub part 1 missing x₁₂
    currList = [1]
    lbList = [lbs[10], lbs[12]]
    ubList = [ubs[10], ubs[12]]

    l_x11_p1_sp1_LB, l_x11_p1_sp1_UB = lift_OA(emptyList, currList, x11_p1_sp1_LB, x11_p1_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₁₀, x₁₂)
    emptyList = [1] #Sub part 2 missing x₁₀
    currList = [2]

    l_x11_p1_sp2_LB, l_x11_p1_sp2_UB = lift_OA(emptyList, currList, x11_p1_sp2_LB, x11_p1_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x11_p1_LB, x11_p1_UB = prodBounds(l_x11_p1_sp1_LB, l_x11_p1_sp1_UB, l_x11_p1_sp2_LB, l_x11_p1_sp2_UB)

    if sanityFlag
        @assert validBounds(:($((Jz - Jx)/Jy)*x10*x12), [:x10, :x12], x11_p1_LB, x11_p1_UB) "Invalid bounds for x11"
    end

    #Finally, bounds for x11 have to be a function of x11. Lift bounds to space of (x₁₀, x₁₁, x₁₂)
    emptyList = [2] #Since x11 comes after x10 and before x12
    currList = [1,3]
    lbList = [lbs[10], lbs[11], lbs[12]]
    ubList = [ubs[10], ubs[11], ubs[12]]

    x11_LB, x11_UB = lift_OA(emptyList, currList, x11_p1_LB, x11_p1_UB, lbList, ubList)
    return x11_LB, x11_UB
end

function bound_quadx12(Quad, plotFlag, sanityFlag = true, npoint=1)
    lbs, ubs = extrema(Quad.domain)

    #Bounding ((Jx - Jy)/Jz)*x₁₀*x₁₁
    #Sub-part 1: (Jx - Jy)/Jz * x₁₀
    x12_p1_sp1 = :($((Jx - Jy)/Jz)*x)
    lb_x12_p1_sp1 = lbs[10]
    ub_x12_p1_sp1 = ubs[10]

    x12_p1_sp1_LB, x12_p1_sp1_UB = interpol_nd(bound_univariate(x12_p1_sp1, lb_x12_p1_sp1, ub_x12_p1_sp1, npoint=npoint)...)

    #Sub-part 2: x₁₁
    x12_p1_sp2 = :(1*x)
    lb_x12_p1_sp2 = lbs[11]
    ub_x12_p1_sp2 = ubs[11]

    x12_p1_sp2_LB, x12_p1_sp2_UB = interpol_nd(bound_univariate(x12_p1_sp2, lb_x12_p1_sp2, ub_x12_p1_sp2, npoint=npoint)...)

    #Lift the bounds to the same space
    emptyList = [2] #Sub part 1 missing x₁₁
    currList = [1]
    lbList = [lbs[10], lbs[11]]
    ubList = [ubs[10], ubs[11]]

    l_x12_p1_sp1_LB, l_x12_p1_sp1_UB = lift_OA(emptyList, currList, x12_p1_sp1_LB, x12_p1_sp1_UB, lbList, ubList)

    #Now lift sub part 2 to space of (x₁₀, x₁₁)
    emptyList = [1] #Sub part 2 missing x₁₀
    currList = [2]

    l_x12_p1_sp2_LB, l_x12_p1_sp2_UB = lift_OA(emptyList, currList, x12_p1_sp2_LB, x12_p1_sp2_UB, lbList, ubList)

    #Now multiply the lifted bounds
    x12_p1_LB, x12_p1_UB = prodBounds(l_x12_p1_sp1_LB, l_x12_p1_sp1_UB, l_x12_p1_sp2_LB, l_x12_p1_sp2_UB)

    if sanityFlag
        @assert validBounds(:($((Jx - Jy)/Jz)*x10*x11), [:x10, :x11], x12_p1_LB, x12_p1_UB) "Invalid bounds for x12"
    end

    #Finally, bounds for x12 have to be a function of x12. Lift bounds to space of (x₁₀, x₁₁, x₁₂)
    emptyList = [3] #Since x12 comes after x10 and x11
    currList = [1,2]
    lbList = [lbs[10], lbs[11], lbs[12]]
    ubList = [ubs[10], ubs[11], ubs[12]]

    x12_LB, x12_UB = lift_OA(emptyList, currList, x12_p1_LB, x12_p1_UB, lbList, ubList)

    return x12_LB, x12_UB
end
