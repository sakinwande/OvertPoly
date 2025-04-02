include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

# domain  = Hyperrectangle(low=[-0.75, 0.85, -0.65, -0.45, -0.55, 0.65], high=[-0.74, 0.86, -0.64, -0.44, -0.54, 0.66])
# domain  = Hyperrectangle(low=[-0.25,-0.25,-0.25,-0.25,-0.25,-0.25], high=[0.25,0.25,0.25,0.25,0.25,0.25])
lowVec = -1*ones(6)
highVec = 1*ones(6)
domain  = Hyperrectangle(low=lowVec, high=highVec)
npoint = 1

function bound_att1(plotFlag = false, sanityFlag = true, npoint=1)
    """
        Function to bound ̇x₁ = 0.5 * (x₅*(x₁² + x₂² + x₃² - x₃) + x₆*(x₁² + x₂² + x₂ + x₃²) + x₄*(x₁² + x₂² + x₃² +1))

        Domain is a subset of (x₁, x₂, x₃, x₄, x₅, x₆)
    """
    #Divide and conquer
    lbs, ubs = extrema(domain)
    #Part 1: Bound the x₅*(x₁² + x₂² + x₃² - x₃) term
    #Sub-part 1: Bound x₁²

    p1_sp1 = :(x^2)
    lb_1 = lbs[1]
    ub_1 = ubs[1]
    p1_sp1_LB, p1_sp1_UB = interpol_nd(bound_univariate(p1_sp1, lb_1, ub_1, npoint=npoint)...)

    # #Sub-part 2: Bound x₂²
    p1_sp2 = :(x^2)
    lb_2 = lbs[2]
    ub_2 = ubs[2]
    p1_sp2_LB, p1_sp2_UB = interpol_nd(bound_univariate(p1_sp2, lb_2, ub_2, npoint=npoint)...)

    # #Sub-part 3: Bound x₃²
    p1_sp3 = :(x^2 - x)
    lb_3 = lbs[3]
    ub_3 = ubs[3]    
    p1_sp3_LB, p1_sp3_UB = interpol_nd(bound_univariate(p1_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₅
    p1_sp4 = :(0.0009587*0.5*x)
    lb_5 = lbs[5]
    ub_5 = ubs[5]
    p1_sp4_LB, p1_sp4_UB = interpol_nd(bound_univariate(p1_sp4, lb_5, ub_5, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₅*(x₁² + x₂² + x₃² - x₃)
    #First, lift each bound to space of (x₁, x₂, x₃, x₅)

    #For part 1
    lbs_l = [lbs[1:3]..., lbs[5]]
    ubs_l = [ubs[1:3]..., ubs[5]]
    p1_sp1_LB_lifted, p1_sp1_UB_lifted = lift_OA([2,3,4], [1], p1_sp1_LB, p1_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p1_sp2_LB_lifted, p1_sp2_UB_lifted = lift_OA([1,3,4], [2], p1_sp2_LB, p1_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p1_sp3_LB_lifted, p1_sp3_UB_lifted = lift_OA([1,2,4], [3], p1_sp3_LB, p1_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p1_sp4_LB_lifted, p1_sp4_UB_lifted = lift_OA([1,2,3], [4], p1_sp4_LB, p1_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p1_sp5_LB, p1_sp5_UB = sumBounds(p1_sp1_LB_lifted, p1_sp1_UB_lifted, p1_sp2_LB_lifted, p1_sp2_UB_lifted, false)

    p1_sp6_LB, p1_sp6_UB = sumBounds(p1_sp5_LB, p1_sp5_UB, p1_sp3_LB_lifted, p1_sp3_UB_lifted, false)


    #Then multiply bounds
    p1_LB, p1_UB = prodBounds(p1_sp6_LB, p1_sp6_UB, p1_sp4_LB_lifted, p1_sp4_UB_lifted)

    #Part 2: Bound the x₆*(x₁² + x₂² + x₂ + x₃²) term
    #Sub-part 1: Bound x₁²
    p2_sp1 = :(x^2)
    p2_sp1_LB, p2_sp1_UB = interpol_nd(bound_univariate(p2_sp1, lb_1, ub_1, npoint=npoint)...)

    #Sub-part 2: Bound x₂²
    p2_sp2 = :(x^2 + x)
    p2_sp2_LB, p2_sp2_UB = interpol_nd(bound_univariate(p2_sp2, lb_2, ub_2, npoint=npoint)...)

    #Sub-part 3: Bound x₃²
    p2_sp3 = :(x^2)
    p2_sp3_LB, p2_sp3_UB = interpol_nd(bound_univariate(p2_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₆
    p2_sp4 = :(0.5*x)
    lb_6 = lbs[6]
    ub_6 = ubs[6]
    p2_sp4_LB, p2_sp4_UB = interpol_nd(bound_univariate(p2_sp4, lb_6, ub_6, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₆*(x₁² + x₂² + x₂ + x₃²)
    #First, lift each bound to space of (x₁, x₂, x₃, x₆)
    lbs_l = [lbs[1:3]..., lbs[6]]
    ubs_l = [ubs[1:3]..., ubs[6]]

    #For part 1
    p2_sp1_LB_lifted, p2_sp1_UB_lifted = lift_OA([2,3,4], [1], p2_sp1_LB, p2_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p2_sp2_LB_lifted, p2_sp2_UB_lifted = lift_OA([1,3,4], [2], p2_sp2_LB, p2_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p2_sp3_LB_lifted, p2_sp3_UB_lifted = lift_OA([1,2,4], [3], p2_sp3_LB, p2_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p2_sp4_LB_lifted, p2_sp4_UB_lifted = lift_OA([1,2,3], [4], p2_sp4_LB, p2_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p2_sp5_LB, p2_sp5_UB = sumBounds(p2_sp1_LB_lifted, p2_sp1_UB_lifted, p2_sp2_LB_lifted, p2_sp2_UB_lifted, false)

    p2_sp6_LB, p2_sp6_UB = sumBounds(p2_sp5_LB, p2_sp5_UB, p2_sp3_LB_lifted, p2_sp3_UB_lifted, false)

    #Then multiply bounds
    p2_LB, p2_UB = prodBounds(p2_sp6_LB, p2_sp6_UB, p2_sp4_LB_lifted, p2_sp4_UB_lifted)

    #Part 3: Bound the x₄*(x₁² + x₂² + x₃² +1) term
    #Sub-part 1: Bound x₁²
    p3_sp1 = :(x^2 + 1)
    p3_sp1_LB, p3_sp1_UB = interpol_nd(bound_univariate(p3_sp1, lb_1, ub_1, npoint=npoint)...)

    #Sub-part 2: Bound x₂²
    p3_sp2 = :(x^2)
    p3_sp2_LB, p3_sp2_UB = interpol_nd(bound_univariate(p3_sp2, lb_2, ub_2, npoint=npoint)...)

    #Sub-part 3: Bound x₃²
    p3_sp3 = :(x^2)
    p3_sp3_LB, p3_sp3_UB = interpol_nd(bound_univariate(p3_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₄
    p3_sp4 = :(0.5*x)
    lb_4 = lbs[4]
    ub_4 = ubs[4]
    p3_sp4_LB, p3_sp4_UB = interpol_nd(bound_univariate(p3_sp4, lb_4, ub_4, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₄*(x₁² + x₂² + x₃² +1)
    #First, lift each bound to space of (x₁, x₂, x₃, x₄)
    lbs_l = [lbs[1:3]..., lbs[4]]
    ubs_l = [ubs[1:3]..., ubs[4]]

    #For part 1
    p3_sp1_LB_lifted, p3_sp1_UB_lifted = lift_OA([2,3,4], [1], p3_sp1_LB, p3_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p3_sp2_LB_lifted, p3_sp2_UB_lifted = lift_OA([1,3,4], [2], p3_sp2_LB, p3_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p3_sp3_LB_lifted, p3_sp3_UB_lifted = lift_OA([1,2,4], [3], p3_sp3_LB, p3_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p3_sp4_LB_lifted, p3_sp4_UB_lifted = lift_OA([1,2,3], [4], p3_sp4_LB, p3_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p3_sp5_LB, p3_sp5_UB = sumBounds(p3_sp1_LB_lifted, p3_sp1_UB_lifted, p3_sp2_LB_lifted, p3_sp2_UB_lifted, false)

    p3_sp6_LB, p3_sp6_UB = sumBounds(p3_sp5_LB, p3_sp5_UB, p3_sp3_LB_lifted, p3_sp3_UB_lifted, false)

    #Then multiply bounds
    p3_LB, p3_UB = prodBounds(p3_sp6_LB, p3_sp6_UB, p3_sp4_LB_lifted, p3_sp4_UB_lifted)

    #Combine bounds using addition
    #First, lift each bound to space of (x₁, x₂, x₃, x₄, x₅, x₆)

    p1_LB_lifted, p1_UB_lifted = lift_OA([4,6], [1,2,3,5], p1_LB, p1_UB, lbs, ubs)

    p2_LB_lifted, p2_UB_lifted = lift_OA([4,5], [1,2,3,6], p2_LB, p2_UB, lbs, ubs)

    p3_LB_lifted, p3_UB_lifted = lift_OA([5,6], [1,2,3,4], p3_LB, p3_UB, lbs, ubs)

    p4_LB, p4_UB = sumBounds(p1_LB_lifted, p1_UB_lifted, p2_LB_lifted, p2_UB_lifted, false)

    p5_LB, p5_UB = sumBounds(p4_LB, p4_UB, p3_LB_lifted, p3_UB_lifted, false)

    return p5_LB, p5_UB
end

function bound_att2(plotFlag = false, santityFlag = true, npoint=1)
    """
        Function to bound ̇x₂ = 0.5 * (x₄*(x₁² + x₂² + x₃² + x₃) + x₆*(x₁² - x₁+ x₂² + x₃²) + x₅*(x₁² + x₂² + x₃² +1))

        Domain is a subset of (x₁, x₂, x₃, x₄, x₅, x₆)
    """

    #Divide and conquer
    lbs, ubs = extrema(domain)

    #Part 1: Bound the x₄*(x₁² + x₂² + x₃² + x₃) term

    #Sub-part 1: Bound x₁²
    p1_sp1 = :(x^2)
    lb_1 = lbs[1]
    ub_1 = ubs[1]
    p1_sp1_LB, p1_sp1_UB = interpol_nd(bound_univariate(p1_sp1, lb_1, ub_1, npoint=npoint)...)

    #Sub-part 2: Bound x₂²
    p1_sp2 = :(x^2)
    lb_2 = lbs[2]
    ub_2 = ubs[2]
    p1_sp2_LB, p1_sp2_UB = interpol_nd(bound_univariate(p1_sp2, lb_2, ub_2, npoint=npoint)...)

    #Sub-part 3: Bound x₃²+ x₃
    p1_sp3 = :(x^2 + x)
    lb_3 = lbs[3]
    ub_3 = ubs[3]
    p1_sp3_LB, p1_sp3_UB = interpol_nd(bound_univariate(p1_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₄
    p1_sp4 = :(0.0003308*0.5*x)
    lb_4 = lbs[4]
    ub_4 = ubs[4]
    p1_sp4_LB, p1_sp4_UB = interpol_nd(bound_univariate(p1_sp4, lb_4, ub_4, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₄*(x₁² + x₂² + x₃² + x₃)
    #First, lift each bound to space of (x₁, x₂, x₃, x₄)

    lbs_l = [lbs[1:3]..., lbs[4]]
    ubs_l = [ubs[1:3]..., ubs[4]]

    #For part 1
    p1_sp1_LB_lifted, p1_sp1_UB_lifted = lift_OA([2,3,4], [1], p1_sp1_LB, p1_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p1_sp2_LB_lifted, p1_sp2_UB_lifted = lift_OA([1,3,4], [2], p1_sp2_LB, p1_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p1_sp3_LB_lifted, p1_sp3_UB_lifted = lift_OA([1,2,4], [3], p1_sp3_LB, p1_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p1_sp4_LB_lifted, p1_sp4_UB_lifted = lift_OA([1,2,3], [4], p1_sp4_LB, p1_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p1_sp5_LB, p1_sp5_UB = sumBounds(p1_sp1_LB_lifted, p1_sp1_UB_lifted, p1_sp2_LB_lifted, p1_sp2_UB_lifted, false)

    p1_sp6_LB, p1_sp6_UB = sumBounds(p1_sp5_LB, p1_sp5_UB, p1_sp3_LB_lifted, p1_sp3_UB_lifted, false)

    #Then multiply bounds
    p1_LB, p1_UB = prodBounds(p1_sp6_LB, p1_sp6_UB, p1_sp4_LB_lifted, p1_sp4_UB_lifted)

    #Part 2: Bound the x₆*(x₁² - x₁+ x₂² + x₃²) term

    #Sub-part 1: Bound x₁²
    p2_sp1 = :(x^2 - x)
    p2_sp1_LB, p2_sp1_UB = interpol_nd(bound_univariate(p2_sp1, lb_1, ub_1, npoint=npoint)...)

    #Sub-part 2: Bound x₂²
    p2_sp2 = :(x^2)
    p2_sp2_LB, p2_sp2_UB = interpol_nd(bound_univariate(p2_sp2, lb_2, ub_2, npoint=npoint)...)

    #Sub-part 3: Bound x₃²
    p2_sp3 = :(x^2)
    p2_sp3_LB, p2_sp3_UB = interpol_nd(bound_univariate(p2_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₆
    p2_sp4 = :(0.5*x)
    lb_6 = lbs[6]
    ub_6 = ubs[6]
    p2_sp4_LB, p2_sp4_UB = interpol_nd(bound_univariate(p2_sp4, lb_6, ub_6, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₆*(x₁² - x₁+ x₂² + x₃²)
    #First, lift each bound to space of (x₁, x₂, x₃, x₆)

    lbs_l = [lbs[1:3]..., lbs[6]]
    ubs_l = [ubs[1:3]..., ubs[6]]

    #For part 1
    p2_sp1_LB_lifted, p2_sp1_UB_lifted = lift_OA([2,3,4], [1], p2_sp1_LB, p2_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p2_sp2_LB_lifted, p2_sp2_UB_lifted = lift_OA([1,3,4], [2], p2_sp2_LB, p2_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p2_sp3_LB_lifted, p2_sp3_UB_lifted = lift_OA([1,2,4], [3], p2_sp3_LB, p2_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p2_sp4_LB_lifted, p2_sp4_UB_lifted = lift_OA([1,2,3], [4], p2_sp4_LB, p2_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p2_sp5_LB, p2_sp5_UB = sumBounds(p2_sp1_LB_lifted, p2_sp1_UB_lifted, p2_sp2_LB_lifted, p2_sp2_UB_lifted, false)

    p2_sp6_LB, p2_sp6_UB = sumBounds(p2_sp5_LB, p2_sp5_UB, p2_sp3_LB_lifted, p2_sp3_UB_lifted, false)

    #Then multiply bounds
    p2_LB, p2_UB = prodBounds(p2_sp6_LB, p2_sp6_UB, p2_sp4_LB_lifted, p2_sp4_UB_lifted)

    #Part 3: Bound the x₅*(x₁² + x₂² + x₃² +1) term

    #Sub-part 1: Bound x₁²
    p3_sp1 = :(x^2 + 1)
    p3_sp1_LB, p3_sp1_UB = interpol_nd(bound_univariate(p3_sp1, lb_1, ub_1, npoint=npoint)...)

    #Sub-part 2: Bound x₂²
    p3_sp2 = :(x^2)
    p3_sp2_LB, p3_sp2_UB = interpol_nd(bound_univariate(p3_sp2, lb_2, ub_2, npoint=npoint)...)

    #Sub-part 3: Bound x₃²
    p3_sp3 = :(x^2)
    p3_sp3_LB, p3_sp3_UB = interpol_nd(bound_univariate(p3_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₅
    p3_sp4 = :(0.5*x)
    lb_5 = lbs[5]
    ub_5 = ubs[5]
    p3_sp4_LB, p3_sp4_UB = interpol_nd(bound_univariate(p3_sp4, lb_5, ub_5, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₅*(x₁² + x₂² + x₃² +1)
    #First, lift each bound to space of (x₁, x₂, x₃, x₅)

    lbs_l = [lbs[1:3]..., lbs[5]]
    ubs_l = [ubs[1:3]..., ubs[5]]

    #For part 1
    p3_sp1_LB_lifted, p3_sp1_UB_lifted = lift_OA([2,3,4], [1], p3_sp1_LB, p3_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p3_sp2_LB_lifted, p3_sp2_UB_lifted = lift_OA([1,3,4], [2], p3_sp2_LB, p3_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p3_sp3_LB_lifted, p3_sp3_UB_lifted = lift_OA([1,2,4], [3], p3_sp3_LB, p3_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p3_sp4_LB_lifted, p3_sp4_UB_lifted = lift_OA([1,2,3], [5], p3_sp4_LB, p3_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p3_sp5_LB, p3_sp5_UB = sumBounds(p3_sp1_LB_lifted, p3_sp1_UB_lifted, p3_sp2_LB_lifted, p3_sp2_UB_lifted, false)

    p3_sp6_LB, p3_sp6_UB = sumBounds(p3_sp5_LB, p3_sp5_UB, p3_sp3_LB_lifted, p3_sp3_UB_lifted, false)

    #Then multiply bounds
    p3_LB, p3_UB = prodBounds(p3_sp6_LB, p3_sp6_UB, p3_sp4_LB_lifted, p3_sp4_UB_lifted)

    #Combine bounds using addition
    #First, lift each bound to space of (x₁, x₂, x₃, x₄, x₅, x₆)

    p1_LB_lifted, p1_UB_lifted = lift_OA([5,6], [1,2,3,4], p1_LB, p1_UB, lbs, ubs)

    p2_LB_lifted, p2_UB_lifted = lift_OA([4,5], [1,2,3,6], p2_LB, p2_UB, lbs, ubs)

    p3_LB_lifted, p3_UB_lifted = lift_OA([4,6], [1,2,3,5], p3_LB, p3_UB, lbs, ubs)

    p4_LB, p4_UB = sumBounds(p1_LB_lifted, p1_UB_lifted, p2_LB_lifted, p2_UB_lifted, false)

    p5_LB, p5_UB = sumBounds(p4_LB, p4_UB, p3_LB_lifted, p3_UB_lifted, false)

    return p5_LB, p5_UB
end

function bound_att3(plotFlag = false, sanityFlag = true, npoint=1)
    """
        Function to bound ̇x₃ = 0.5 * (x₄*(x₁² + x₂² - x₂ + x₃²) + x₅*(x₁² + x₁ + x₂² + x₃²) + x₆*(x₁² + x₂² + x₃² +1))

        Domain is a subset of (x₁, x₂, x₃, x₄, x₅, x₆)
    """

    #Divide and conquer
    lbs, ubs = extrema(domain)

    #Part 1: Bound the x₄*(x₁² + x₂² - x₂ + x₃²) term

    #Sub-part 1: Bound x₁²
    p1_sp1 = :(x^2)
    lb_1 = lbs[1]
    ub_1 = ubs[1]
    p1_sp1_LB, p1_sp1_UB = interpol_nd(bound_univariate(p1_sp1, lb_1, ub_1, npoint=npoint)...)

    #Sub-part 2: Bound x₂² - x₂
    p1_sp2 = :(x^2 - x)
    lb_2 = lbs[2]
    ub_2 = ubs[2]
    p1_sp2_LB, p1_sp2_UB = interpol_nd(bound_univariate(p1_sp2, lb_2, ub_2, npoint=npoint)...)

    #Sub-part 3: Bound x₃²
    p1_sp3 = :(x^2)
    lb_3 = lbs[3]
    ub_3 = ubs[3]
    p1_sp3_LB, p1_sp3_UB = interpol_nd(bound_univariate(p1_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₄
    p1_sp4 = :(0.00002176*0.5*x)
    lb_4 = lbs[4]
    ub_4 = ubs[4]
    p1_sp4_LB, p1_sp4_UB = interpol_nd(bound_univariate(p1_sp4, lb_4, ub_4, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₄*(x₁² + x₂² - x₂ + x₃²)
    #First, lift each bound to space of (x₁, x₂, x₃, x₄)

    lbs_l = [lbs[1:3]..., lbs[4]]
    ubs_l = [ubs[1:3]..., ubs[4]]

    #For part 1
    p1_sp1_LB_lifted, p1_sp1_UB_lifted = lift_OA([2,3,4], [1], p1_sp1_LB, p1_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p1_sp2_LB_lifted, p1_sp2_UB_lifted = lift_OA([1,3,4], [2], p1_sp2_LB, p1_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p1_sp3_LB_lifted, p1_sp3_UB_lifted = lift_OA([1,2,4], [3], p1_sp3_LB, p1_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p1_sp4_LB_lifted, p1_sp4_UB_lifted = lift_OA([1,2,3], [4], p1_sp4_LB, p1_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p1_sp5_LB, p1_sp5_UB = sumBounds(p1_sp1_LB_lifted, p1_sp1_UB_lifted, p1_sp2_LB_lifted, p1_sp2_UB_lifted, false)

    p1_sp6_LB, p1_sp6_UB = sumBounds(p1_sp5_LB, p1_sp5_UB, p1_sp3_LB_lifted, p1_sp3_UB_lifted, false)

    #Then multiply bounds
    p1_LB, p1_UB = prodBounds(p1_sp6_LB, p1_sp6_UB, p1_sp4_LB_lifted, p1_sp4_UB_lifted)

    #Part 2: Bound the x₅*(x₁² + x₁ + x₂² + x₃²) term

    #Sub-part 1: Bound x₁² + x₁
    p2_sp1 = :(x^2 + x)
    p2_sp1_LB, p2_sp1_UB = interpol_nd(bound_univariate(p2_sp1, lb_1, ub_1, npoint=npoint)...)

    #Sub-part 2: Bound x₂²
    p2_sp2 = :(x^2)
    p2_sp2_LB, p2_sp2_UB = interpol_nd(bound_univariate(p2_sp2, lb_2, ub_2, npoint=npoint)...)

    #Sub-part 3: Bound x₃²
    p2_sp3 = :(x^2)
    p2_sp3_LB, p2_sp3_UB = interpol_nd(bound_univariate(p2_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₅
    p2_sp4 = :(0.5*x)
    lb_5 = lbs[5]
    ub_5 = ubs[5]
    p2_sp4_LB, p2_sp4_UB = interpol_nd(bound_univariate(p2_sp4, lb_5, ub_5, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₅*(x₁² + x₁ + x₂² + x₃²)
    #First, lift each bound to space of (x₁, x₂, x₃, x₅)

    lbs_l = [lbs[1:3]..., lbs[5]]
    ubs_l = [ubs[1:3]..., ubs[5]]

    #For part 1
    p2_sp1_LB_lifted, p2_sp1_UB_lifted = lift_OA([2,3,4], [1], p2_sp1_LB, p2_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p2_sp2_LB_lifted, p2_sp2_UB_lifted = lift_OA([1,3,4], [2], p2_sp2_LB, p2_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p2_sp3_LB_lifted, p2_sp3_UB_lifted = lift_OA([1,2,4], [3], p2_sp3_LB, p2_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p2_sp4_LB_lifted, p2_sp4_UB_lifted = lift_OA([1,2,3], [5], p2_sp4_LB, p2_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p2_sp5_LB, p2_sp5_UB = sumBounds(p2_sp1_LB_lifted, p2_sp1_UB_lifted, p2_sp2_LB_lifted, p2_sp2_UB_lifted, false)

    p2_sp6_LB, p2_sp6_UB = sumBounds(p2_sp5_LB, p2_sp5_UB, p2_sp3_LB_lifted, p2_sp3_UB_lifted, false)

    #Then multiply bounds
    p2_LB, p2_UB = prodBounds(p2_sp6_LB, p2_sp6_UB, p2_sp4_LB_lifted, p2_sp4_UB_lifted)

    #Part 3: Bound the x₆*(x₁² + x₂² + x₃² +1) term

    #Sub-part 1: Bound x₁²
    p3_sp1 = :(x^2)
    p3_sp1_LB, p3_sp1_UB = interpol_nd(bound_univariate(p3_sp1, lb_1, ub_1, npoint=npoint)...)

    #Sub-part 2: Bound x₂²
    p3_sp2 = :(x^2)
    p3_sp2_LB, p3_sp2_UB = interpol_nd(bound_univariate(p3_sp2, lb_2, ub_2, npoint=npoint)...)

    #Sub-part 3: Bound x₃² + 1
    p3_sp3 = :(x^2 + 1)
    p3_sp3_LB, p3_sp3_UB = interpol_nd(bound_univariate(p3_sp3, lb_3, ub_3, npoint=npoint)...)

    #Sub-part 4: Bound x₆
    p3_sp4 = :(0.5*x)
    lb_6 = lbs[6]
    ub_6 = ubs[6]
    p3_sp4_LB, p3_sp4_UB = interpol_nd(bound_univariate(p3_sp4, lb_6, ub_6, npoint=npoint)...)

    #Sub-part 5: Combine bounds to yield x₆*(x₁² + x₂² + x₃² +1)
    #First, lift each bound to space of (x₁, x₂, x₃, x₆)

    lbs_l = [lbs[1:3]..., lbs[6]]
    ubs_l = [ubs[1:3]..., ubs[6]]

    #For part 1
    p3_sp1_LB_lifted, p3_sp1_UB_lifted = lift_OA([2,3,4], [1], p3_sp1_LB, p3_sp1_UB, lbs_l, ubs_l)

    #For part 2
    p3_sp2_LB_lifted, p3_sp2_UB_lifted = lift_OA([1,3,4], [2], p3_sp2_LB, p3_sp2_UB, lbs_l, ubs_l)

    #For part 3
    p3_sp3_LB_lifted, p3_sp3_UB_lifted = lift_OA([1,2,4], [3], p3_sp3_LB, p3_sp3_UB, lbs_l, ubs_l)

    #For part 4
    p3_sp4_LB_lifted, p3_sp4_UB_lifted = lift_OA([1,2,3], [6], p3_sp4_LB, p3_sp4_UB, lbs_l, ubs_l)

    #Combine bounds using addition
    p3_sp5_LB, p3_sp5_UB = sumBounds(p3_sp1_LB_lifted, p3_sp1_UB_lifted, p3_sp2_LB_lifted, p3_sp2_UB_lifted, false)

    p3_sp6_LB, p3_sp6_UB = sumBounds(p3_sp5_LB, p3_sp5_UB, p3_sp3_LB_lifted, p3_sp3_UB_lifted, false)

    #Then multiply bounds
    p3_LB, p3_UB = prodBounds(p3_sp6_LB, p3_sp6_UB, p3_sp4_LB_lifted, p3_sp4_UB_lifted)

    #Combine bounds using addition
    #First, lift each bound to space of (x₁, x₂, x₃, x₄, x₅, x₆)

    p1_LB_lifted, p1_UB_lifted = lift_OA([5,6], [1,2,3,4], p1_LB, p1_UB, lbs, ubs)

    p2_LB_lifted, p2_UB_lifted = lift_OA([4,6], [1,2,3,5], p2_LB, p2_UB, lbs, ubs)

    p3_LB_lifted, p3_UB_lifted = lift_OA([4,5], [1,2,3,6], p3_LB, p3_UB, lbs, ubs)

    p4_LB, p4_UB = sumBounds(p1_LB_lifted, p1_UB_lifted, p2_LB_lifted, p2_UB_lifted, false)

    p5_LB, p5_UB = sumBounds(p4_LB, p4_UB, p3_LB_lifted, p3_UB_lifted, false)

    return p5_LB, p5_UB
end

function bound_att4(plotFlag = false, sanityFlag = true, npoint=1)
    """
        Function to bound  ̇x₄ = 0.25*x₅*x₆

        Domain is a subset of (x₄, x₅, x₆)
    """

    #Divide and conquer
    lbs, ubs = extrema(domain)

    #Part 1: Bound x₅
    p1 = :(0.25*x)
    lb_5 = lbs[5]
    ub_5 = ubs[5]
    p1_LB, p1_UB = interpol_nd(bound_univariate(p1, lb_5, ub_5, npoint=npoint)...)

    #Part 2: Bound x₆
    p2 = :(-0.9451*x)
    lb_6 = lbs[6]
    ub_6 = ubs[6]
    p2_LB, p2_UB = interpol_nd(bound_univariate(p2, lb_6, ub_6, npoint=npoint)...)

    #Combine bounds using multiplication
    #First, lift each bound to space of (x₅, x₆)

    lbs_l = [lbs[4:6]...]
    ubs_l = [ubs[4:6]...]

    p1_LB_lifted, p1_UB_lifted = lift_OA([1,3], [2], p1_LB, p1_UB, lbs_l, ubs_l)

    p2_LB_lifted, p2_UB_lifted = lift_OA([1,2], [3], p2_LB, p2_UB, lbs_l, ubs_l)

    p3_LB, p3_UB = prodBounds(p1_LB_lifted, p1_UB_lifted, p2_LB_lifted, p2_UB_lifted)

    #Lift to space of (x₁, x₂, x₃, x₄, x₅, x₆)
    p3_LB_lifted, p3_UB_lifted = lift_OA([1,2,3], [4,5,6], p3_LB, p3_UB, lbs, ubs)

    return p3_LB_lifted, p3_UB_lifted
end

function bound_att5(plotFlag = false, sanityFlag = true, npoint=1)
    """
        Function to bound ̇x₅ = -1.5*x₄*x₆

        Domain is a subset of (x₄, x₅, x₆)
    """

    #Divide and conquer
    lbs, ubs = extrema(domain)

    #Part 1: Bound x₄
    p1 = :(-1.5*x)
    lb_4 = lbs[4]
    ub_4 = ubs[4]
    p1_LB, p1_UB = interpol_nd(bound_univariate(p1, lb_4, ub_4, npoint=npoint)...)

    #Part 2: Bound x₆
    p2 = :(-0.3261*x) 
    lb_6 = lbs[6]
    ub_6 = ubs[6]
    p2_LB, p2_UB = interpol_nd(bound_univariate(p2, lb_6, ub_6, npoint=npoint)...)

    #Combine bounds using multiplication
    #First, lift each bound to space of (x₄, x₆)

    lbs_l = [lbs[4:6]...]
    ubs_l = [ubs[4:6]...]

    p1_LB_lifted, p1_UB_lifted = lift_OA([2,3], [1], p1_LB, p1_UB, lbs_l, ubs_l)

    p2_LB_lifted, p2_UB_lifted = lift_OA([1,2], [3], p2_LB, p2_UB, lbs_l, ubs_l)

    p3_LB, p3_UB = prodBounds(p1_LB_lifted, p1_UB_lifted, p2_LB_lifted, p2_UB_lifted)

    #Lift to space of (x₁, x₂, x₃, x₄, x₅, x₆)
    p3_LB_lifted, p3_UB_lifted = lift_OA([1,2,3], [4,5,6], p3_LB, p3_UB, lbs, ubs)

    return p3_LB_lifted, p3_UB_lifted

end

function bound_att6(plotFlag = false, sanityFlag = true, npoint=1)
    """
        Function to bound ̇x₆ = 2*x₄*x₅

        Domain is a subset of (x₄, x₅, x₆)
    """

    #Divide and conquer
    lbs, ubs = extrema(domain)

    #Part 1: Bound x₄
    p1 = :(2*x)
    lb_4 = lbs[4]
    ub_4 = ubs[4]
    p1_LB, p1_UB = interpol_nd(bound_univariate(p1, lb_4, ub_4, npoint=npoint)...)

    #Part 2: Bound x₅
    p2 = :(-0.0214*x)
    lb_5 = lbs[5]
    ub_5 = ubs[5]
    p2_LB, p2_UB = interpol_nd(bound_univariate(p2, lb_5, ub_5, npoint=npoint)...)

    #Combine bounds using multiplication
    #First, lift each bound to space of (x₄, x₅)

    lbs_l = [lbs[4:6]...]
    ubs_l = [ubs[4:6]...]

    p1_LB_lifted, p1_UB_lifted = lift_OA([2,3], [1], p1_LB, p1_UB, lbs_l, ubs_l)

    p2_LB_lifted, p2_UB_lifted = lift_OA([1,3], [2], p2_LB, p2_UB, lbs_l, ubs_l)

    p3_LB, p3_UB = prodBounds(p1_LB_lifted, p1_UB_lifted, p2_LB_lifted, p2_UB_lifted)

    #Lift to space of (x₁, x₂, x₃, x₄, x₅, x₆)
    p3_LB_lifted, p3_UB_lifted = lift_OA([1,2,3], [4,5,6], p3_LB, p3_UB, lbs, ubs)

    return p3_LB_lifted, p3_UB_lifted
end

function bound_attitude(npoints)
    x1LB, x1UB = bound_att1(false, true, npoints)
    x2LB, x2UB = bound_att2(false, true, npoints)
    x3LB, x3UB = bound_att3(false, true, npoints)
    x4LB, x4UB = bound_att4(false, true, npoints)
    x5LB, x5UB = bound_att5(false, true, npoints)
    x6LB, x6UB = bound_att6(false, true, npoints)

    LB_int1, UB_int1 = sumBounds(x1LB, x1UB, x2LB, x2UB, false)
    LB_int2, UB_int2 = sumBounds(LB_int1, UB_int1, x3LB, x3UB, false)
    LB_int3, UB_int3 = sumBounds(LB_int2, UB_int2, x4LB, x4UB, false)
    LB_int4, UB_int4 = sumBounds(LB_int3, UB_int3, x5LB, x5UB, false)
    LB_int5, UB_int5 = sumBounds(LB_int4, UB_int4, x6LB, x6UB, false)

    LB_inputs = [tup[1:end-1] for tup in LB_int5]

    #Write to file 
    open("nfunc4_n$(npoints)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], ",", tup[3],  ",", tup[4], ",", tup[5], ",", tup[6], "\n"))
        end
    end
end

bound_attitude(1)
bound_attitude(2)
bound_attitude(3)
bound_attitude(4)
bound_attitude(5)
bound_attitude(10)
