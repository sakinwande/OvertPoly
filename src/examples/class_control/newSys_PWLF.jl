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
domain = Hyperrectangle(low=[-0.6, -0.6], high=[0.6, 0.6])

npoint = 2

J1_1= -0.6360
J1_2 = 0.7717

J2_1 = -0.8337
J2_2 = 0.5523

J3_1 = -0.363
J3_2 = -0.9311
J3_3 = -0.035

Jp_1 = -0.8957
Jp_2 = -0.4447

function bound_func(npoint)
    domain = Hyperrectangle(low=[-1, -1], high=[1, 1])
    lbs, ubs = extrema(domain)
    #Function to bound: -sin(x1) + 0.5(x1)^3 - 0.5x2
    #Part 1: Bound the -sin(x1) + 0.5(x1)^3 term
    p1 = :($(J1_1)*sin(x) + $(J1_2) * 0.5*(x)^3)
    
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[1], ubs[1], npoint=npoint)

    #Part 2: Bound the -0.5x2 term
    p2 = :($(J1_1) * 0.5*x)
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([2], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([1], [2], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_inputs = [tup[1:end-1] for tup in LB_1]

    open("nfunc_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end

    
end

function bound_func2(npoint)
    domain = Hyperrectangle(low=[-0.6, -0.6], high=[0.6, 0.6])
    lbs, ubs = extrema(domain)
    #Function to bound: -sin(x1) + 0.5(x1)^3 - x1 + (x2)^2 - sin(x2)
    #Part 1: Bound the -sin(x1) - 0.5(x1)^3 term
    p1 = :($(J2_1)*-sin(x) + $(J2_2) - 0.5*(x)^3 + $(J2_2) -x)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[1], ubs[1], npoint=npoint)

    #Part 2: Bound the (x2)^2 - sin(x2) term
    p2 = :($(J2_1) * x^2 + $(J2_2) - sin(x))
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([2], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([1], [2], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_inputs = [tup[1:end-1] for tup in LB_1]

    open("nfunc2_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end

end

function bound_pend(npoint)
    #Function to bound: -0.5x2 - 0.5sin(x1)
    lbs = [-π/2, -π/2]
    ubs = [π/2, π/2]

    #Part 1: Bound the -0.5x2 term
    p1 = :($(Jp_1)*-0.5*x)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[2], ubs[2], npoint=npoint)

    #Part 2: Bound the -0.5sin(x1) term
    p2 = :($(Jp_1)*0.5*sin(x))
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[1], ubs[1], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([1], [2], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([2], [1], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_inputs = [tup[1:end-1] for tup in LB_1]

    open("pend_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end
end

function bound_func3(npoint)
    #want to bound x3x2^2 + x2x3^2 + x1x2^2
    domain = Hyperrectangle(low=[-1, -1, -1], high=[1, 1, 1])
    lbs, ubs = extrema(domain)

    #Part 1: Bound the x3x2^2 term
    p1_sp1 = :(1*x)
    p1_sp1_LB, p1_sp1_UB = bound_univariate(p1_sp1, lbs[3], ubs[3], npoint=npoint)

    p1_sp2 = :($(J3_1)*x^2)
    p1_sp2_LB, p1_sp2_UB = bound_univariate(p1_sp2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₂, x₃)
    l_p1_LB_1, l_p1_UB_1 = lift_OA([2], [1], p1_sp1_LB, p1_sp1_UB, lbs, ubs)
    l_p1_LB_2, l_p1_UB_2 = lift_OA([1], [2], p1_sp2_LB, p1_sp2_UB, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_1, l_p1_UB_1, l_p1_LB_2, l_p1_UB_2, false)

    #Part 2: Bound the x2x3^2 term
    p2_sp1 = :(-1*x)
    p2_sp1_LB, p2_sp1_UB = bound_univariate(p2_sp1, lbs[2], ubs[2], npoint=npoint)

    p2_sp2 = :($(J3_1)*x^2)
    p2_sp2_LB, p2_sp2_UB = bound_univariate(p2_sp2, lbs[3], ubs[3], npoint=npoint)

    #Lift bounds to space of (x₂, x₃)
    l_p2_LB_1, l_p2_UB_1 = lift_OA([1], [2], p2_sp1_LB, p2_sp1_UB, lbs, ubs)
    l_p2_LB_2, l_p2_UB_2 = lift_OA([2], [1], p2_sp2_LB, p2_sp2_UB, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_2, UB_2 = sumBounds(l_p2_LB_1, l_p2_UB_1, l_p2_LB_2, l_p2_UB_2, false)

    #Part 3: Bound the x1x2^2 term
    p3_sp1 = :(1*x)
    p3_sp1_LB, p3_sp1_UB = bound_univariate(p3_sp1, lbs[1], ubs[1], npoint=npoint)

    p3_sp2 = :($(J3_3)*x^2)
    p3_sp2_LB, p3_sp2_UB = bound_univariate(p3_sp2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    l_p3_LB_1, l_p3_UB_1 = lift_OA([1], [2], p3_sp1_LB, p3_sp1_UB, lbs, ubs)
    l_p3_LB_2, l_p3_UB_2 = lift_OA([2], [1], p3_sp2_LB, p3_sp2_UB, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_3, UB_3 = sumBounds(l_p3_LB_1, l_p3_UB_1, l_p3_LB_2, l_p3_UB_2, false)

    #Lift bounds to space of (x₁, x₂, x₃)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([1], [2,3], LB_1, UB_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([1], [2,3], LB_2, UB_2, lbs, ubs)

    #Lift part 3
    l_p3_LB_13, l_p3_UB_13 = lift_OA([3], [1,2], LB_3, UB_3, lbs, ubs)

    #Sum the bounds to recover 3d bounds
    LB_4, UB_4 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_5, UB_5 = sumBounds(LB_4, UB_4, l_p3_LB_13, l_p3_UB_13, false)

    LB_inputs = [tup[1:end-1] for tup in LB_5]

    open("nfunc3_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], ",", tup[3], "\n"))
        end
    end
end

function bound_func4(npoint)
    """
    x_dot = -3x + 4y + x² - y²
    y_dot = sin(5x) - y³

    We want to bound it as one function so we can use 
        func = -y³ - y² + x² + sin(5x)
    the domain is x,y in [-1, 1] x [-1, 1]
    """
    domain = Hyperrectangle(low=[-1, -1], high=[1, 1])
    lbs, ubs = extrema(domain)
    #Part 1: Bound the -y³ - y² term
    p1 = :(-1*x^3 - 1*x^2)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[2], ubs[2], npoint=npoint)

    #Part 2: Bound the x² + sin(5x) term
    p2 = :(1*x^2 + 1*sin(5*x))
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[1], ubs[1], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([1], [2], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([2], [1], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_inputs = [tup[1:end-1] for tup in LB_1]

    open("nfunc4_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end

    return LB_1, UB_1
end

function bound_func5(npoint)
    """
    x_dot = x^2 - y^2 - x
    y_dot = 2xy - 0.5x^2

    Use something like x^2 + y^2 since that's what primarily drives the convexity here
    """

    domain = Hyperrectangle(low=[-1, -1], high=[1, 1])
    lbs, ubs = extrema(domain)

    #Part 1, bound the x^2 term 
    p1 = :(1*x^2)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[1], ubs[1], npoint=npoint)

    #Part 2, bound the y^2 term
    p2 = :( -1*x^2)
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([2], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([1], [2], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_inputs = [tup[1:end-1] for tup in LB_1]
    open("nfunc5_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end
end

function bound_func6(npoint)
    """
    x_dot = 0.25y - 0.5x
    y_dot = 0.25x^3 -2.5x - 0.5y^3

    want to bound as one function, so focus on bounding 0.5x^3 + 0.5y^3 since that's what really drives the curvature here
    """

    domain = Hyperrectangle(low=[-1, -1], high=[1, 1])
    lbs, ubs = extrema(domain)

    #Part 1, bound the 0.5x^3 term
    p1 = :(0.5*x^3)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[1], ubs[1], npoint=npoint)

    #Part 2, bound the 0.5y^3 term
    p2 = :(0.5*x^3)
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([2], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([1], [2], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_inputs = [tup[1:end-1] for tup in LB_1]
    open("nfunc6_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end
end

function bound_func7(npoint)
    """
    x_dot = z
    y_dot = w
    z_dot = -x^3 - 0.5z  - x + y -0.25z + 0.25w
    w_dot = -y + x - 0.25w + 0.25z

    want to bound as one function, so focus on bounding -x^3 + y + z + w since that's what really drives the curvature here
    """

    domain = Hyperrectangle(low=[-0.25, -0.25, -0.25, -0.25], high=[0.25, 0.25, 0.25, 0.25])
    lbs, ubs = extrema(domain)

    #Part 1, bound the -x^3 term
    p1 = :(-1*x^3)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[1], ubs[1], npoint=npoint)

    #Part 2, bound the y term
    p2 = :(1*x)
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[2], ubs[2], npoint=npoint)

    #Part 3, bound the z term
    p3 = :(1*x)
    p3_LB_1_3, p3_UB_1_3 = bound_univariate(p3, lbs[3], ubs[3], npoint=npoint)

    #Part 4, bound the w term
    p4 = :(1*x)
    p4_LB_1_4, p4_UB_1_4 = bound_univariate(p4, lbs[4], ubs[4], npoint=npoint)

    #Lift bounds to space of (x₁, x₂, x₃, x₄)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([2,3,4], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([1,3,4], [2], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Lift part 3
    l_p3_LB_13, l_p3_UB_13 = lift_OA([1,2,4], [3], p3_LB_1_3, p3_UB_1_3, lbs, ubs)

    #Lift part 4
    l_p4_LB_14, l_p4_UB_14 = lift_OA([1,2,3], [4], p4_LB_1_4, p4_UB_1_4, lbs, ubs)

    #Sum the bounds to recover 4d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_2, UB_2 = sumBounds(LB_1, UB_1, l_p3_LB_13, l_p3_UB_13, false)

    LB_3, UB_3 = sumBounds(LB_2, UB_2, l_p4_LB_14, l_p4_UB_14, false)

    LB_inputs = [tup[1:end-1] for tup in LB_3]
    open("nfunc7_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], ",", tup[3], ",", tup[4], "\n"))
        end
    end
end

function bound_func8(npoint)
    """
    x_dot = z
    y_dot = w
    z_dot = -x -0.5z - x^3 + 43x^2 - 3xy^2 + y^3 - 0.25z + 0.25w
    w_dot = x^3 + 3x^2y + 3xy^2 - y^3 - y - 0.25w + 0.25z

    want to bound as one function. So focus on bounding x^3 + y^3 + 3x^2y + 3xy^2 + z + w since that's what really drives the curvature here
    """

    domain = Hyperrectangle(low=[-0.25, -0.25, -0.25, -0.25], high=[0.25, 0.25, 0.25, 0.25])
    lbs, ubs = extrema(domain)

    #Part 1, bound the x^3 term
    p1 = :(1*x^3)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[1], ubs[1], npoint=npoint)  

    #Part 2, bound the y^3 term
    p2 = :(1*x^3)
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[2], ubs[2], npoint=npoint)

    #Part 3, bound the 3x^2y term
    #Split into two parts
    p3_sp1 = :(3*x^2)
    p3_sp1_LB, p3_sp1_UB = bound_univariate(p3_sp1, lbs[1], ubs[1], npoint=npoint)

    p3_sp2 = :(1*x)
    p3_sp2_LB, p3_sp2_UB = bound_univariate(p3_sp2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    l_p3_LB_1, l_p3_UB_1 = lift_OA([1], [2], p3_sp1_LB, p3_sp1_UB, lbs, ubs)
    l_p3_LB_2, l_p3_UB_2 = lift_OA([2], [1], p3_sp2_LB, p3_sp2_UB, lbs, ubs)

    #Combine to get bounds on 3x^2y]
    p3_LB, p3_UB = prodBounds(l_p3_LB_1, l_p3_UB_1, l_p3_LB_2, l_p3_UB_2)

    #Part 4, bound the 3xy^2 term
    #Split into two parts
    p4_sp1 = :(3*x)
    p4_sp1_LB, p4_sp1_UB = bound_univariate(p4_sp1, lbs[1], ubs[1], npoint=npoint)

    p4_sp2 = :(x^2)
    p4_sp2_LB, p4_sp2_UB = bound_univariate(p4_sp2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    l_p4_LB_1, l_p4_UB_1 = lift_OA([1], [2], p4_sp1_LB, p4_sp1_UB, lbs, ubs)
    l_p4_LB_2, l_p4_UB_2 = lift_OA([2], [1], p4_sp2_LB, p4_sp2_UB, lbs, ubs)

    #Combine to get bounds on 3xy^2
    p4_LB, p4_UB = prodBounds(l_p4_LB_1, l_p4_UB_1, l_p4_LB_2, l_p4_UB_2)

    #Part 5, bound the z term
    p5 = :(1*x)
    p5_LB_1_1, p5_UB_1_1 = bound_univariate(p5, lbs[3], ubs[3], npoint=npoint)
    
    #Part 6, bound the w term
    p6 = :(1*x)
    p6_LB_1_2, p6_UB_1_2 = bound_univariate(p6, lbs[4], ubs[4], npoint=npoint)

    #Lift bounds to space of (x₁, x₂, x₃, x₄)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([2,3,4], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([1,3,4], [2], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Lift part 3
    l_p3_LB_13, l_p3_UB_13 = lift_OA([3,4], [1,2], p3_LB, p3_UB, lbs, ubs)

    #Lift part 4
    l_p4_LB_14, l_p4_UB_14 = lift_OA([3,4], [1,2], p4_LB, p4_UB, lbs, ubs)

    #Lift part 5
    l_p5_LB_15, l_p5_UB_15 = lift_OA([1,2,4], [3], p5_LB_1_1, p5_UB_1_1, lbs, ubs)  

    #Lift part 6
    l_p6_LB_16, l_p6_UB_16 = lift_OA([1,2,3], [4], p6_LB_1_2, p6_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 4d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    LB_2, UB_2 = sumBounds(LB_1, UB_1, l_p3_LB_13, l_p3_UB_13, false)
    LB_3, UB_3 = sumBounds(LB_2, UB_2, l_p4_LB_14, l_p4_UB_14, false)
    LB_4, UB_4 = sumBounds(LB_3, UB_3, l_p5_LB_15, l_p5_UB_15, false)
    LB_5, UB_5 = sumBounds(LB_4, UB_4, l_p6_LB_16, l_p6_UB_16, false)   

    LB_inputs = [tup[1:end-1] for tup in LB_5]
    open("nfunc8_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_inputs
            write(file, string(tup[1], ",", tup[2], ",", tup[3], ",", tup[4], "\n"))
        end
        write(file, "dx_2\n")
    end


end
###########################################

bound_func4(2)
bound_func4(5)
bound_func4(10)
bound_func4(20)
bound_func4(50)
bound_func4(100)
bound_func4(200)


bound_func5(2)
bound_func5(5)
bound_func5(10)
bound_func5(20)
bound_func5(50)
bound_func5(100)
bound_func5(200)

bound_func6(2)
bound_func6(5)
bound_func6(10)
bound_func6(20)
bound_func6(50)
bound_func6(100)
bound_func6(200)

bound_func7(2)
bound_func7(5)
bound_func7(10)
bound_func7(20)
bound_func7(50)
bound_func7(100)
bound_func7(200)

bound_func8(2)
bound_func8(5)
bound_func8(10)
bound_func8(20)
bound_func8(50)
bound_func8(100)
bound_func8(200)

####################################
# end
# bound_func(1)
# bound_func(2)
# bound_func(3)
# bound_func(4)
# bound_func(5)

# bound_func2(1)
# bound_func2(2)
# bound_func2(3)
# bound_func2(4)
# bound_func2(5)

# bound_pend(1)
# bound_pend(2)
# bound_pend(3)
# bound_pend(4)
# bound_pend(5)

# bound_func3(1)
# bound_func3(2)
# bound_func3(3)
# bound_func3(4)


bound_func3(5)

# bound_func(2)
# bound_func(5)
# bound_func(10)
# bound_func(20)
# bound_func(50)
# bound_func(100)
# bound_func(200)
# bound_func(500)
# bound_func(1000)

# bound_func2(2)
# bound_func2(5)
# bound_func2(10)
# bound_func2(20)
# bound_func2(50)
# bound_func2(100)
# bound_func2(200)
# bound_func2(500)
# bound_func2(1000) 

# bound_pend(2)
# bound_pend(5)
# bound_pend(10)
# bound_pend(20)
# bound_pend(50)
# bound_pend(100)
# bound_pend(200)
# bound_pend(500)
# bound_pend(1000)

# bound_func3(2)
# bound_func3(5)
# bound_func3(10)
# bound_func3(20)
# bound_func3(50)
# bound_func3(100)
# bound_func3(200)
# bound_func3(500)
# bound_func3(1000)

function bound_func_old(npoint)
    lbs, ubs = extrema(domain)
    #Part 2: Bound the -x2 - 0.3*x1 - (0.5^x1)^3 term
    p1 = :(-1*x)
    p2_LB_1_1, p2_UB_1_1 = bound_univariate(p1, lbs[2], ubs[2], npoint=npoint)

    p2 = :(- 0.3*x - 0.5*(x)^3)
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[1], ubs[1], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p2_LB_11, l_p2_UB_11 = lift_OA([1], [2], p2_LB_1_1, p2_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([2], [1], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p2_LB_11, l_p2_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    #Part 1: Bound the x2 - sin(x1) term
    p1 = :(1*x)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[2], ubs[2], npoint=npoint)

    p2 = :(sin(x))
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[1], ubs[1], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([1], [2], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p1_LB_12, l_p1_UB_12 = lift_OA([2], [1], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_2, UB_2 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p1_LB_12, l_p1_UB_12, true)

    LB_1_inputs = [tup[1:end-1] for tup in LB_1]
    LB_2_inputs = [tup[1:end-1] for tup in LB_2]

    open("nfunc1_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_1_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
        write(file, "dx_2\n")
        for tup in LB_2_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end
end

function bound_func2_old(npoint)
    lbs, ubs = extrema(domain)
    #Part 2: Bound the -sin(x2) - 0.3*x1 - (0.5^x1)^3 term
    p1 = :(sin(x))
    p2_LB_1_1, p2_UB_1_1 = bound_univariate(p1, lbs[2], ubs[2], npoint=npoint)

    p2 = :(-x - 0.5*(x)^3)
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[1], ubs[1], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p2_LB_11, l_p2_UB_11 = lift_OA([1], [2], p2_LB_1_1, p2_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p2_LB_12, l_p2_UB_12 = lift_OA([2], [1], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p2_LB_11, l_p2_UB_11, l_p2_LB_12, l_p2_UB_12, false)

    #Part 1: Bound the (x2)^2 - sin(x1) term
    p1 = :(x^2)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[2], ubs[2], npoint=npoint)

    p2 = :(sin(x))
    p2_LB_1_2, p2_UB_1_2 = bound_univariate(p2, lbs[1], ubs[1], npoint=npoint)

    #Lift bounds to space of (x₁, x₂)
    #Lift part 1 first
    l_p1_LB_11, l_p1_UB_11 = lift_OA([1], [2], p1_LB_1_1, p1_UB_1_1, lbs, ubs)

    #Lift part 2
    l_p1_LB_12, l_p1_UB_12 = lift_OA([2], [1], p2_LB_1_2, p2_UB_1_2, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_2, UB_2 = sumBounds(l_p1_LB_11, l_p1_UB_11, l_p1_LB_12, l_p1_UB_12, true)

    LB_1_inputs = [tup[1:end-1] for tup in LB_1]
    LB_2_inputs = [tup[1:end-1] for tup in LB_2]

    open("nfunc2_n$(npoint)_bounds.txt", "w") do file 
        write(file, "dx_1\n")
        for tup in LB_1_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
        write(file, "dx_2\n")
        for tup in LB_2_inputs
            write(file, string(tup[1], ",", tup[2], "\n"))
        end
    end
end