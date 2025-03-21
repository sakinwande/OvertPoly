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

function bound_func(npoint)
    domain = Hyperrectangle(low=[-1, -1], high=[1, 1])
    lbs, ubs = extrema(domain)
    #Function to bound: -sin(x1) + 0.5(x1)^3 - 0.5x2
    #Part 1: Bound the -sin(x1) + 0.5(x1)^3 term
    p1 = :(-sin(x) - 0.5*(x)^3)
    
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[1], ubs[1], npoint=npoint)

    #Part 2: Bound the -0.5x2 term
    p2 = :(-0.5*x)
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
    p1 = :(-sin(x) - 0.5*(x)^3 - x)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[1], ubs[1], npoint=npoint)

    #Part 2: Bound the (x2)^2 - sin(x2) term
    p2 = :(x^2 - sin(x))
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
    p1 = :(-0.5*x)
    p1_LB_1_1, p1_UB_1_1 = bound_univariate(p1, lbs[2], ubs[2], npoint=npoint)

    #Part 2: Bound the -0.5sin(x1) term
    p2 = :(0.5*sin(x))
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

    p1_sp2 = :(x^2)
    p1_sp2_LB, p1_sp2_UB = bound_univariate(p1_sp2, lbs[2], ubs[2], npoint=npoint)

    #Lift bounds to space of (x₂, x₃)
    l_p1_LB_1, l_p1_UB_1 = lift_OA([2], [1], p1_sp1_LB, p1_sp1_UB, lbs, ubs)
    l_p1_LB_2, l_p1_UB_2 = lift_OA([1], [2], p1_sp2_LB, p1_sp2_UB, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_1, UB_1 = sumBounds(l_p1_LB_1, l_p1_UB_1, l_p1_LB_2, l_p1_UB_2, false)

    #Part 2: Bound the x2x3^2 term
    p2_sp1 = :(1*x)
    p2_sp1_LB, p2_sp1_UB = bound_univariate(p2_sp1, lbs[2], ubs[2], npoint=npoint)

    p2_sp2 = :(x^2)
    p2_sp2_LB, p2_sp2_UB = bound_univariate(p2_sp2, lbs[3], ubs[3], npoint=npoint)

    #Lift bounds to space of (x₂, x₃)
    l_p2_LB_1, l_p2_UB_1 = lift_OA([1], [2], p2_sp1_LB, p2_sp1_UB, lbs, ubs)
    l_p2_LB_2, l_p2_UB_2 = lift_OA([2], [1], p2_sp2_LB, p2_sp2_UB, lbs, ubs)

    #Sum the bounds to recover 2d bounds
    LB_2, UB_2 = sumBounds(l_p2_LB_1, l_p2_UB_1, l_p2_LB_2, l_p2_UB_2, false)

    #Part 3: Bound the x1x2^2 term
    p3_sp1 = :(1*x)
    p3_sp1_LB, p3_sp1_UB = bound_univariate(p3_sp1, lbs[1], ubs[1], npoint=npoint)

    p3_sp2 = :(x^2)
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


bound_func(1)
bound_func(2)
bound_func(3)
bound_func(4)
bound_func(5)

bound_func2(1)
bound_func2(2)
bound_func2(3)
bound_func2(4)
bound_func2(5)
bound_func2(6)
bound_func2(7)
bound_func2(8)
bound_func2(9)


bound_pend(1)
bound_pend(2)
bound_pend(3)
bound_pend(4)
bound_pend(5)

bound_func3(1)
bound_func3(2)
bound_func3(3)
bound_func3(4)
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