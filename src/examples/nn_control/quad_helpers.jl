include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

function bound_quadx1(QUAD, plotFlag = false, sanityFlag = false)
    """
    Function to bound ̇x₁ = cos(x₈)*cos(x₉)*x₄ + (sin(x₇)*sin(x₈)*cos(x₉) - cos(x₇)*sin(x₉))*x₅ + (cos(x₇)*sin(x₈)*cos(x₉) + sin(x₇)*sin(x₉))*x₆

    The need for a separate bounding function is self evident 

    Args:
        QUAD: Quadrotor dynamics
        plotFlag: Flag to plot the bounds
        sanityFlag: Flag to check the validity of the bounds
    """
    lbs, ubs = extrema(QUAD.domain)
    #bound first state variable of quadcopter (inertial north position)
    #function of 6 variables f(x4, x5, x6, x7, x8, x9)
    #Strategy: break into 7 parts and bound each part separately

    #Part 1: f(x4, x8, x9) = cos(x8)*cos(x9)*x4
    #K-A decomposition is exp(log(cos(x8)) + log(cos(x9)) + log(x4))
    #Sub-part 1 = cos(x8)
    x1_p1_sp1 = :(cos(x))
    lb_x1_p1_sp1 = lbs[8]
    ub_x1_p1_sp1 = ubs[8]

    #Bounding trig functions over very thin intervals is tricky. Widen for now
    if ub_x1_p1_sp1 - lb_x1_p1_sp1 < 1e-5
        lb_x1_p1_sp1 = lb_x1_p1_sp1 - 1e-5
        ub_x1_p1_sp1 = ub_x1_p1_sp1 + 1e-5
        lbs[8] = lb_x1_p1_sp1
        ubs[8] = ub_x1_p1_sp1
    end
    x1_p1_sp1_LB, x1_p1_sp1_UB = interpol_nd(bound_univariate(x1_p1_sp1, lb_x1_p1_sp1, ub_x1_p1_sp1)...)
    
    if sanityFlag
        validBounds(x1_p1_sp1, [:x], x1_p1_sp1_LB, x1_p1_sp1_UB, true)
    end

    #Sub-part 2 = cos(x9)
    x1_p1_sp2 = :(cos(x))
    lb_x1_p1_sp2 = lbs[9]
    ub_x1_p1_sp2 = ubs[9]
    if ub_x1_p1_sp2 - lb_x1_p1_sp2 < 1e-5
        lb_x1_p1_sp2 = lb_x1_p1_sp2 - 1e-5
        ub_x1_p1_sp2 = ub_x1_p1_sp2 + 1e-5
        lbs[9] = lb_x1_p1_sp2
        ubs[9] = ub_x1_p1_sp2
    end
    x1_p1_sp2_LB, x1_p1_sp2_UB = interpol_nd(bound_univariate(x1_p1_sp2, lb_x1_p1_sp2, ub_x1_p1_sp2)...)

    if sanityFlag
        validBounds(x1_p1_sp2, [:x], x1_p1_sp2_LB, x1_p1_sp2_UB, true)
    end
    
    #Sub-part 3 = x4
    x1_p1_sp3 = :(1*x)
    lb_x1_p1_sp3 = lbs[4]
    ub_x1_p1_sp3 = ubs[4]
    x1_p1_sp3_LB, x1_p1_sp3_UB = interpol_nd(bound_univariate(x1_p1_sp3, lb_x1_p1_sp3, ub_x1_p1_sp3)...)

    if sanityFlag
        validBounds(x1_p1_sp3, [:x], x1_p1_sp3_LB, x1_p1_sp3_UB, true)
    end

    #Find how much to shift each pair of bounds by for valid log 
    s_x1p1sp1 = inpShiftLog(lb_x1_p1_sp1, ub_x1_p1_sp1, bounds=x1_p1_sp1_LB)
    s_x1p1sp2 = inpShiftLog(lb_x1_p1_sp2, ub_x1_p1_sp2, bounds=x1_p1_sp2_LB)
    s_x1p1sp3 = inpShiftLog(lb_x1_p1_sp3, ub_x1_p1_sp3, bounds=x1_p1_sp3_LB)

    #Apply log
    x1_p1_sp1_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p1sp1)) for tup in x1_p1_sp1_LB]
    x1_p1_sp1_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p1sp1)) for tup in x1_p1_sp1_UB]

    x1_p1_sp2_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p1sp2)) for tup in x1_p1_sp2_LB]
    x1_p1_sp2_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p1sp2)) for tup in x1_p1_sp2_UB]

    x1_p1_sp3_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p1sp3)) for tup in x1_p1_sp3_LB]
    x1_p1_sp3_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p1sp3)) for tup in x1_p1_sp3_UB]

    #Add dimensions to prepare for Minkowski sum 
    #p1 is a function of x4, x8, x9. For sp1, add dimension for x4 and x9
    #Set zeroval to be zero
    x1_p1_sp1_LB_ll = addDim(x1_p1_sp1_LB_l, 1, 0.0) #for x4, index 1
    x1_p1_sp1_LB_ll = addDim(x1_p1_sp1_LB_ll, 3, 0.0) #for x9 index 3

    x1_p1_sp1_UB_ll = addDim(x1_p1_sp1_UB_l, 1, 0.0) #for x4, index 1
    x1_p1_sp1_UB_ll = addDim(x1_p1_sp1_UB_ll, 3, 0.0) #for x9 index 3

    if sanityFlag
        validBounds(:(log((cos(x8) + $s_x1p1sp1))), [:x4, :x8, :x9], x1_p1_sp1_LB_ll, x1_p1_sp1_UB_ll, true)
    end

    #For sp2, add dimension for x4 and x8
    x1_p1_sp2_LB_ll = addDim(x1_p1_sp2_LB_l, 1, 0.0) #for x4, index 1
    x1_p1_sp2_LB_ll = addDim(x1_p1_sp2_LB_ll, 2, 0.0) #for x8 index 2

    x1_p1_sp2_UB_ll = addDim(x1_p1_sp2_UB_l, 1, 0.0) #for x4, index 1
    x1_p1_sp2_UB_ll = addDim(x1_p1_sp2_UB_ll, 2, 0.0) #for x8 index 2

    if sanityFlag
        validBounds(:(log((cos(x9) + $s_x1p1sp2))), [:x4, :x8, :x9], x1_p1_sp2_LB_ll, x1_p1_sp2_UB_ll, true)
    end

    #For sp3, add dimension for x8 and x9
    x1_p1_sp3_LB_ll = addDim(x1_p1_sp3_LB_l, 2, 0.0) #for x8, index 2
    x1_p1_sp3_LB_ll = addDim(x1_p1_sp3_LB_ll, 3, 0.0) #for x9 index 3

    x1_p1_sp3_UB_ll = addDim(x1_p1_sp3_UB_l, 2, 0.0) #for x8, index 2
    x1_p1_sp3_UB_ll = addDim(x1_p1_sp3_UB_ll, 3, 0.0) #for x9 index 3

    if sanityFlag 
        validBounds(:(log((x4 + $s_x1p1sp3))), [:x4, :x8, :x9], x1_p1_sp3_LB_ll, x1_p1_sp3_UB_ll, true)
    end

    #Combine sp1 and sp2 first to get log(cos(x8)*cos(x9))
    x1_p1_sp4_LB_l = MinkSum(x1_p1_sp1_LB_ll, x1_p1_sp2_LB_ll)
    x1_p1_sp4_UB_l = MinkSum(x1_p1_sp1_UB_ll, x1_p1_sp2_UB_ll)

    if sanityFlag
        validBounds(:((0*log(x4 + $s_x1p1sp3)+ log(cos(x8) + $s_x1p1sp1) + log(cos(x9) + $s_x1p1sp2))), [:x4, :x8, :x9], x1_p1_sp4_LB_l, x1_p1_sp4_UB_l, true)
    end

    #Combine sp4 with sp3 to get log(cos(x8)*cos(x9)*x4)
    x1_p1_LB_l = MinkSum(x1_p1_sp4_LB_l, x1_p1_sp3_LB_ll)
    x1_p1_UB_l = MinkSum(x1_p1_sp4_UB_l, x1_p1_sp3_UB_ll)

    if sanityFlag
        validBounds(:((log(x4 + $s_x1p1sp3))+log(cos(x8)+$s_x1p1sp1)+log(cos(x9) + $s_x1p1sp2)), [:x4, :x8, :x9], x1_p1_LB_l, x1_p1_UB_l, true)
    end
    
    #Compute exp to get bounds for cos(x8)*cos(x9)*x4
    x1_p1_LB_s = [(tup[1:end-1]..., floor_n(exp(tup[end]))) for tup in x1_p1_LB_l]
    x1_p1_UB_s = [(tup[1:end-1]..., ceil_n(exp(tup[end]))) for tup in x1_p1_UB_l]

    if sanityFlag
        validBounds(:((cos(x8) + $s_x1p1sp1)*(cos(x9) + $s_x1p1sp2)*(x4 + $s_x1p1sp3)), [:x4, :x8, :x9], x1_p1_LB_s, x1_p1_UB_s, true)
    end

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first 
    #NOTE: Modify index name. Index 1 is not the variable order index (which would be x4), instead index 1 is the index for the variable for x1_p1_sp1
    
    x1_p1_LB = []
    x1_p1_UB = []
    #Shift down using interval subtraction 
    #f1f2f3 = f_hat- f1f3b - f2f3a - f3ab - f1f2c - f1bc - f2ac - abc
    #a is s_x1p1sp1, b is s_x1p1sp2, c is s_x1p1sp3
    #f1 is cos(x8)[x1_p1_sp1], f2 is cos(x9)[x1_p1_sp2], f3 is x4[x1_p1_sp3]
    #NOTE: Edited to actually do interval subtraction
    for (i,tup) in enumerate(x1_p1_LB_s)
        tupUB = x1_p1_UB_s[i]
        #First find corresponding indices 
        #Index 1 is x4, its bounds are in x1_p1_sp3
        ind3 = findall(x->x[1] == tup[1], x1_p1_sp3_LB)[1]
        #Index 2 is x8, its bounds are in x1_p1_sp1
        ind1 = findall(x->x[1] == tup[2], x1_p1_sp1_LB)[1]
        #Index 3 is x9, its bounds are in x1_p1_sp2
        ind2 = findall(x->x[1] == tup[3], x1_p1_sp2_LB)[1]
        f1f2f3_UB = tupUB[end] - x1_p1_sp1_LB[ind1][end]*x1_p1_sp3_LB[ind3][end]*s_x1p1sp2 - x1_p1_sp2_LB[ind2][end]*x1_p1_sp3_LB[ind3][end]*s_x1p1sp1 - x1_p1_sp3_LB[ind3][end]*s_x1p1sp1*s_x1p1sp2 - x1_p1_sp1_LB[ind1][end]*x1_p1_sp2_LB[ind2][end]*s_x1p1sp3 - x1_p1_sp1_LB[ind1][end]*s_x1p1sp2*s_x1p1sp3 - x1_p1_sp2_LB[ind2][end]*s_x1p1sp1*s_x1p1sp3 - s_x1p1sp1*s_x1p1sp2*s_x1p1sp3

        push!(x1_p1_UB, (tup[1:end-1]..., f1f2f3_UB))
    end

    for (i,tup) in enumerate(x1_p1_UB_s)
        tupLB = x1_p1_LB_s[i]
        #Index 1 is x4, its bounds are in x1_p1_sp3
        ind3 = findall(x->x[1] == tup[1], x1_p1_sp3_UB)[1]
        #Index 2 is x8, its bounds are in x1_p1_sp1
        ind1 = findall(x->x[1] == tup[2], x1_p1_sp1_UB)[1]
        #Index 3 is x9, its bounds are in x1_p1_sp2
        ind2 = findall(x->x[1] == tup[3], x1_p1_sp2_UB)[1]
        f1f2f3_LB = tupLB[end] - x1_p1_sp1_UB[ind1][end]*x1_p1_sp3_UB[ind3][end]*s_x1p1sp2 - x1_p1_sp2_UB[ind2][end]*x1_p1_sp3_UB[ind3][end]*s_x1p1sp1 - x1_p1_sp3_UB[ind3][end]*s_x1p1sp1*s_x1p1sp2 - x1_p1_sp1_UB[ind1][end]*x1_p1_sp2_UB[ind2][end]*s_x1p1sp3 - x1_p1_sp1_UB[ind1][end]*s_x1p1sp2*s_x1p1sp3 - x1_p1_sp2_UB[ind2][end]*s_x1p1sp1*s_x1p1sp3 - s_x1p1sp1*s_x1p1sp2*s_x1p1sp3

        push!(x1_p1_LB, (tup[1:end-1]..., f1f2f3_LB))
    end

    #Sanity check, validity
    if sanityFlag
        validBounds(:(x4*cos(x8)*cos(x9)), [:x4, :x8, :x9], x1_p1_LB, x1_p1_UB, true)
    end

    #Part 2: f(x8, x9) = sin(x8)*cos(x9)
    #K-A decomposition is exp(log(sin(x8)) + log(cos(x9)))

    #Sub-part 1 = sin(x8)
    #Bounding sine around zero is tricky :|
    #NOTE: Use wider bounds for now 
    x1_p2_sp1 = :(sin(x))
    lb_x1_p2_sp1 = lbs[8]
    ub_x1_p2_sp1 = ubs[8]
    if ub_x1_p2_sp1 - lb_x1_p2_sp1 < 1e-5
        lb_x1_p2_sp1 = lb_x1_p2_sp1 - 1e-5
        ub_x1_p2_sp1 = ub_x1_p2_sp1 + 1e-5
        lbs[8] = lb_x1_p2_sp1
        ubs[8] = ub_x1_p2_sp1
    end
    x1_p2_sp1_LB, x1_p2_sp1_UB = interpol_nd(bound_univariate(x1_p2_sp1, lb_x1_p2_sp1, ub_x1_p2_sp1)...)

    if sanityFlag
        validBounds(x1_p2_sp1, [:x],x1_p2_sp1_LB, x1_p2_sp1_UB, true)
    end
    
    #Sub-part 2 = cos(x9)
    x1_p2_sp2 = :(cos(x))
    lb_x1_p2_sp2 = lbs[9]
    ub_x1_p2_sp2 = ubs[9]
    if ub_x1_p2_sp2 - lb_x1_p2_sp2 < 1e-5
        lb_x1_p2_sp2 = lb_x1_p2_sp2 - 1e-5
        ub_x1_p2_sp2 = ub_x1_p2_sp2 + 1e-5
        lbs[9] = lb_x1_p2_sp2
        ubs[9] = ub_x1_p2_sp2
    end
    x1_p2_sp2_LB, x1_p2_sp2_UB = interpol_nd(bound_univariate(x1_p2_sp2, lb_x1_p2_sp2, ub_x1_p2_sp2)...)

    if sanityFlag
        validBounds(x1_p2_sp2, [:x],x1_p2_sp2_LB, x1_p2_sp2_UB, true)
    end
    
    #Find how much to shift each pair of bounds by for valid log
    s_x1p2sp1 = inpShiftLog(lb_x1_p2_sp1, ub_x1_p2_sp1, bounds=x1_p2_sp1_LB)
    s_x1p2sp2 = inpShiftLog(lb_x1_p2_sp2, ub_x1_p2_sp2, bounds=x1_p2_sp2_LB)

    #Apply log
    x1_p2_sp1_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p2sp1)) for tup in x1_p2_sp1_LB]
    x1_p2_sp1_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p2sp1)) for tup in x1_p2_sp1_UB]

    x1_p2_sp2_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p2sp2)) for tup in x1_p2_sp2_LB]
    x1_p2_sp2_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p2sp2)) for tup in x1_p2_sp2_UB]

    if sanityFlag
        validBounds(:(log(sin(x8) + $s_x1p2sp1)), [:x8], x1_p2_sp1_LB_l, x1_p2_sp1_UB_l, true)
    end

    if sanityFlag
        validBounds(:(log(cos(x9) + $s_x1p2sp2)), [:x9], x1_p2_sp2_LB_l, x1_p2_sp2_UB_l, true)
    end
    #Add dimensions to prepare for Minkowski sum
    #p2 is a function of x8, x9. For sp1, add dimension for x9
    #Set zeroval to be zero
    x1_p2_sp1_LB_ll = addDim(x1_p2_sp1_LB_l, 2, 0.0) #for x9, index 2
    x1_p2_sp1_UB_ll = addDim(x1_p2_sp1_UB_l, 2, 0.0) #for x9, index 2

    #For sp2, add dimension for x8
    x1_p2_sp2_LB_ll = addDim(x1_p2_sp2_LB_l, 1, 0.0) #for x8, index 1
    x1_p2_sp2_UB_ll = addDim(x1_p2_sp2_UB_l, 1, 0.0) #for x8, index 1

     
    #Combine sp1 and sp2 to get log(sin(x8)*cos(x9))
    x1_p2_LB_l = MinkSum(x1_p2_sp1_LB_ll, x1_p2_sp2_LB_ll)
    x1_p2_UB_l = MinkSum(x1_p2_sp1_UB_ll, x1_p2_sp2_UB_ll)

    if sanityFlag
        validBounds(:(log(sin(x8) + $s_x1p2sp1) + log(cos(x9) + $s_x1p2sp2)), [:x8, :x9], x1_p2_LB_l, x1_p2_UB_l, true)
    end
    
    if sanityFlag
        validBounds(:(log((sin(x8) + $s_x1p2sp1)*(cos(x9) + $s_x1p2sp2))), [:x8, :x9], x1_p2_LB_l, x1_p2_UB_l, true)
    end
    
    #Compute exp to get bounds for sin(x8)*cos(x9)
    x1_p2_LB_s = [(tup[1:end-1]..., floor_n(exp(tup[end]))) for tup in x1_p2_LB_l]
    x1_p2_UB_s = [(tup[1:end-1]..., ceil_n(exp(tup[end]))) for tup in x1_p2_UB_l]

    if sanityFlag
        validBounds(:((sin(x8)+ $s_x1p2sp1)*(cos(x9) + $s_x1p2sp2)), [:x8, :x9], x1_p2_LB_s, x1_p2_UB_s, true)
    end
    
    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p2_LB = []
    x1_p2_UB = []
    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p2sp1, b is s_x1p2sp2
    #f1 is sin(x8)[x1_p2_sp1], f2 is cos(x9)[x1_p2_sp2]

    for (i,tup) in enumerate(x1_p2_LB_s)
        tupUB = x1_p2_UB_s[i]
        #Index 1 is x8, its bounds are in x1_p2_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p2_sp1_LB)[1]
        #Index 2 is x9, its bounds are in x1_p2_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p2_sp2_LB)[1]
        f1f2_UB = tupUB[end] - x1_p2_sp1_LB[ind1][end]*s_x1p2sp2 - x1_p2_sp2_LB[ind2][end]*s_x1p2sp1 - s_x1p2sp1*s_x1p2sp2

        push!(x1_p2_UB, (tup[1:end-1]..., f1f2_UB))
    end

    
    for (i,tup) in enumerate(x1_p2_UB_s)
        tupLB = x1_p2_LB_s[i]
        #Index 1 is x8, its bounds are in x1_p2_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p2_sp1_UB)[1]
        #Index 2 is x9, its bounds are in x1_p2_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p2_sp2_UB)[1]
        f1f2_LB = tupLB[end] - x1_p2_sp1_UB[ind1][end]*s_x1p2sp2 - x1_p2_sp2_UB[ind2][end]*s_x1p2sp1 - s_x1p2sp1*s_x1p2sp2

        push!(x1_p2_LB, (tup[1:end-1]..., f1f2_LB))
    end

    if sanityFlag
        validBounds(:(sin(x8)*cos(x9)), [:x8, :x9], x1_p2_LB, x1_p2_UB, true)
    end

    #Part 3: f(x5, x7) = sin(x7)*x5
    #K-A decomposition is exp(log(sin(x7)) + log(x5))
    #Sub-part 1 = sin(x7)
    x1_p3_sp1 = :(sin(x))
    lb_x1_p3_sp1 = lbs[7]
    ub_x1_p3_sp1 = ubs[7]
    if ub_x1_p3_sp1 - lb_x1_p3_sp1 < 1e-5
        lb_x1_p3_sp1 = lb_x1_p3_sp1 - 1e-5
        ub_x1_p3_sp1 = ub_x1_p3_sp1 + 1e-5
        lbs[7] = lb_x1_p3_sp1
        ubs[7] = ub_x1_p3_sp1
    end
    x1_p3_sp1_LB, x1_p3_sp1_UB = interpol_nd(bound_univariate(x1_p3_sp1, lb_x1_p3_sp1, ub_x1_p3_sp1)...)

    if sanityFlag
        validBounds(x1_p3_sp1, [:x], x1_p3_sp1_LB, x1_p3_sp1_UB, true)
    end
    #Sub-part 2 = x5
    x1_p3_sp2 = :(1*x)
    lb_x1_p3_sp2 = lbs[5]
    ub_x1_p3_sp2 = ubs[5]
    #Specify digits for interpolation
    x1_p3_sp2_LB, x1_p3_sp2_UB = interpol_nd(bound_univariate(x1_p3_sp2, lb_x1_p3_sp2, ub_x1_p3_sp2)...)

    if sanityFlag
        validBounds(x1_p3_sp2, [:x], x1_p3_sp2_LB, x1_p3_sp2_UB, true)
    end
    #Find how much to shift each pair of bounds by for valid log
    s_x1p3sp1 = inpShiftLog(lb_x1_p3_sp1, ub_x1_p3_sp1, bounds=x1_p3_sp1_LB)
    s_x1p3sp2 = inpShiftLog(lb_x1_p3_sp2, ub_x1_p3_sp2, bounds=x1_p3_sp2_LB)

    #Apply log
    x1_p3_sp1_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p3sp1)) for tup in x1_p3_sp1_LB]
    x1_p3_sp1_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p3sp1)) for tup in x1_p3_sp1_UB]

    x1_p3_sp2_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p3sp2)) for tup in x1_p3_sp2_LB]
    x1_p3_sp2_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p3sp2)) for tup in x1_p3_sp2_UB]

    #Add dimensions to prepare for Minkowski sum
    #p3 is a function of x5, x7. For sp1, add dimension for x5
    #Set zeroval to be zero
    x1_p3_sp1_LB_ll = addDim(x1_p3_sp1_LB_l, 1, 0.0) #for x5, index 1
    x1_p3_sp1_UB_ll = addDim(x1_p3_sp1_UB_l, 1, 0.0) #for x5, index 1

    #For sp2, add dimension for x7
    x1_p3_sp2_LB_ll = addDim(x1_p3_sp2_LB_l, 2, 0.0) #for x7, index 2
    x1_p3_sp2_UB_ll = addDim(x1_p3_sp2_UB_l, 2, 0.0) #for x7, index 2

    #Combine sp1 and sp2 to get log(sin(x7)*x5)
    x1_p3_LB_l = MinkSum(x1_p3_sp1_LB_ll, x1_p3_sp2_LB_ll)
    x1_p3_UB_l = MinkSum(x1_p3_sp1_UB_ll, x1_p3_sp2_UB_ll)

    #Compute exp to get bounds for sin(x7)*x5
    x1_p3_LB_s = [(tup[1:end-1]..., floor_n(exp(tup[end]))) for tup in x1_p3_LB_l]
    x1_p3_UB_s = [(tup[1:end-1]..., ceil_n(exp(tup[end]))) for tup in x1_p3_UB_l]

    if sanityFlag
        validBounds(:((x5 + $s_x1p3sp2)*(sin(x7) + $s_x1p3sp1)), [:x5, :x7], x1_p3_LB_s, x1_p3_UB_s, true)
    end
    
    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p3_LB = []
    x1_p3_UB = []

    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p3sp1, b is s_x1p3sp2
    #f1 is sin(x7)[x1_p3_sp1], f2 is x5[x1_p3_sp2]
    for tup in x1_p3_UB_s
        #Index 1 is x5, its bounds are in x1_p3_sp2
        ind2 = findall(x->x[1] == tup[1], x1_p3_sp2_LB)[1]
        #Index 2 is x7, its bounds are in x1_p3_sp1
        ind1 = findall(x->x[1] == tup[2], x1_p3_sp1_LB)[1]
        f1f2_UB = tup[end] - x1_p3_sp1_LB[ind1][end]*s_x1p3sp2 - x1_p3_sp2_LB[ind2][end]*s_x1p3sp1 - s_x1p3sp1*s_x1p3sp2

        push!(x1_p3_UB, (tup[1:end-1]..., f1f2_UB))
    end

    for tup in x1_p3_LB_s
        #Index 1 is x5, its bounds are in x1_p3_sp2
        ind2 = findall(x->x[1] == tup[1], x1_p3_sp2_UB)[1]
        #Index 2 is x7, its bounds are in x1_p3_sp1
        ind1 = findall(x->x[1] == tup[2], x1_p3_sp1_UB)[1]
        f1f2_LB = tup[end] - x1_p3_sp1_UB[ind1][end]*s_x1p3sp2 - x1_p3_sp2_UB[ind2][end]*s_x1p3sp1 - s_x1p3sp1*s_x1p3sp2

        push!(x1_p3_LB, (tup[1:end-1]..., f1f2_LB))
    end

    if sanityFlag
        validBounds(:(x5*sin(x7)), [:x5, :x7], x1_p3_LB, x1_p3_UB, true)
    end
    
    #Part 4: f(x6, x7) = x6cos(x7)
    #K-A decomposition is exp(log(x6) + log(cos(x7)))
    #Sub-part 1 = x6
    x1_p4_sp1 = :(1*x)
    lb_x1_p4_sp1 = lbs[6]
    ub_x1_p4_sp1 = ubs[6]
    x1_p4_sp1_LB, x1_p4_sp1_UB = interpol_nd(bound_univariate(x1_p4_sp1, lb_x1_p4_sp1, ub_x1_p4_sp1)...)

    if sanityFlag
        validBounds(x1_p4_sp1, [:x], x1_p4_sp1_LB, x1_p4_sp1_UB, true)
    end
    #Sub-part 2 = cos(x7)
    x1_p4_sp2 = :(cos(x))
    lb_x1_p4_sp2 = lbs[7]
    ub_x1_p4_sp2 = ubs[7]
    if ub_x1_p4_sp2 - lb_x1_p4_sp2 < 1e-5
        lb_x1_p4_sp2 = lb_x1_p4_sp2 - 1e-5
        ub_x1_p4_sp2 = ub_x1_p4_sp2 + 1e-5
        lbs[7] = lb_x1_p4_sp2
        ubs[7] = ub_x1_p4_sp2
    end
    #Specify digits for interpolation
    x1_p4_sp2_LB, x1_p4_sp2_UB = interpol_nd(bound_univariate(x1_p4_sp2, lb_x1_p4_sp2, ub_x1_p4_sp2)...)

    if sanityFlag
        validBounds(x1_p4_sp2, [:x], x1_p4_sp2_LB, x1_p4_sp2_UB, true)
    end
    #Find how much to shift each pair of bounds by for valid log
    s_x1p4sp1 = inpShiftLog(lb_x1_p4_sp1, ub_x1_p4_sp1, bounds=x1_p4_sp1_LB)
    s_x1p4sp2 = inpShiftLog(lb_x1_p4_sp2, ub_x1_p4_sp2, bounds=x1_p4_sp2_LB)

    #Apply log
    x1_p4_sp1_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p4sp1)) for tup in x1_p4_sp1_LB]
    x1_p4_sp1_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p4sp1)) for tup in x1_p4_sp1_UB]

    x1_p4_sp2_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p4sp2)) for tup in x1_p4_sp2_LB]
    x1_p4_sp2_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p4sp2)) for tup in x1_p4_sp2_UB]

    #Add dimensions to prepare for Minkowski sum
    #p4 is a function of x6, x7. For sp1, add dimension for x7
    #Set zeroval to be zero
    x1_p4_sp1_LB_ll = addDim(x1_p4_sp1_LB_l, 2, 0.0) #for x7, index 2
    x1_p4_sp1_UB_ll = addDim(x1_p4_sp1_UB_l, 2, 0.0) #for x7, index 2

    #For sp2, add dimension for x6
    x1_p4_sp2_LB_ll = addDim(x1_p4_sp2_LB_l, 1, 0.0) #for x6, index 1
    x1_p4_sp2_UB_ll = addDim(x1_p4_sp2_UB_l, 1, 0.0) #for x6, index 1

    #Combine sp1 and sp2 to get log(x6*cos(x7))
    x1_p4_LB_l = MinkSum(x1_p4_sp1_LB_ll, x1_p4_sp2_LB_ll)
    x1_p4_UB_l = MinkSum(x1_p4_sp1_UB_ll, x1_p4_sp2_UB_ll)

    #Compute exp to get bounds for x6*cos(x7)
    x1_p4_LB_s = [(tup[1:end-1]..., floor_n(exp(tup[end]))) for tup in x1_p4_LB_l]
    x1_p4_UB_s = [(tup[1:end-1]..., ceil_n(exp(tup[end]))) for tup in x1_p4_UB_l]

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p4_LB = []
    x1_p4_UB = []

    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p4sp1, b is s_x1p4sp2
    #f1 is x6[x1_p4_sp1], f2 is cos(x7)[x1_p4_sp2]

    for tup in x1_p4_UB_s
        #Index 1 is x6, its bounds are in x1_p4_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p4_sp1_LB)[1]
        #Index 2 is x7, its bounds are in x1_p4_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p4_sp2_LB)[1]
        f1f2_UB = tup[end] - x1_p4_sp1_LB[ind1][end]*s_x1p4sp2 - x1_p4_sp2_LB[ind2][end]*s_x1p4sp1 - s_x1p4sp1*s_x1p4sp2

        push!(x1_p4_UB, (tup[1:end-1]..., f1f2_UB))
    end

    for tup in x1_p4_LB_s
        #Index 1 is x6, its bounds are in x1_p4_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p4_sp1_UB)[1]
        #Index 2 is x7, its bounds are in x1_p4_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p4_sp2_UB)[1]
        f1f2_LB = tup[end] - x1_p4_sp1_UB[ind1][end]*s_x1p4sp2 - x1_p4_sp2_UB[ind2][end]*s_x1p4sp1 - s_x1p4sp1*s_x1p4sp2

        push!(x1_p4_LB, (tup[1:end-1]..., f1f2_LB))
    end

    if sanityFlag
        validBounds(:(x6*cos(x7)), [:x6, :x7], x1_p4_LB, x1_p4_UB, true)
    end
    #Part 5: f(x9) = sin(x9)
    x1_p5 = :(sin(x))
    lb_x1_p5 = lbs[9]
    ub_x1_p5 = ubs[9]
    if ub_x1_p5 - lb_x1_p5 < 1e-5
        lb_x1_p5 = lb_x1_p5 - 1e-5
        ub_x1_p5 = ub_x1_p5 + 1e-5
        lbs[9] = lb_x1_p5
        ubs[9] = ub_x1_p5
    end
    #Specify digits for interpolation
    x1_p5_LB, x1_p5_UB = interpol_nd(bound_univariate(x1_p5, lb_x1_p5, ub_x1_p5)...)

    #Part 6: f(x6, x7) = x6sin(x7)
    #K-A decomposition is exp(log(x6) + log(sin(x7)))
    #Sub-part 1 = x6
    x1_p6_sp1 = :(1*x)
    lb_x1_p6_sp1 = lbs[6]
    ub_x1_p6_sp1 = ubs[6]
    #Specify digits for interpolation
    x1_p6_sp1_LB, x1_p6_sp1_UB = interpol_nd(bound_univariate(x1_p6_sp1, lb_x1_p6_sp1, ub_x1_p6_sp1)...)

    #Sub-part 2 = sin(x7)
    x1_p6_sp2 = :(sin(x))
    lb_x1_p6_sp2 = lbs[7]
    ub_x1_p6_sp2 = ubs[7]
    if ub_x1_p6_sp2 - lb_x1_p6_sp2 < 1e-5
        lb_x1_p6_sp2 = lb_x1_p6_sp2 - 1e-5
        ub_x1_p6_sp2 = ub_x1_p6_sp2 + 1e-5
        lbs[7] = lb_x1_p6_sp2
        ubs[7] = ub_x1_p6_sp2
    end 
    #Specify digits for interpolation
    x1_p6_sp2_LB, x1_p6_sp2_UB = interpol_nd(bound_univariate(x1_p6_sp2, lb_x1_p6_sp2, ub_x1_p6_sp2)...)

    #Find how much to shift each pair of bounds by for valid log
    s_x1p6sp1 = inpShiftLog(lb_x1_p6_sp1, ub_x1_p6_sp1, bounds=x1_p6_sp1_LB)
    s_x1p6sp2 = inpShiftLog(lb_x1_p6_sp2, ub_x1_p6_sp2, bounds=x1_p6_sp2_LB)

    #Apply log
    x1_p6_sp1_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p6sp1)) for tup in x1_p6_sp1_LB]
    x1_p6_sp1_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p6sp1)) for tup in x1_p6_sp1_UB]

    x1_p6_sp2_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p6sp2)) for tup in x1_p6_sp2_LB]
    x1_p6_sp2_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p6sp2)) for tup in x1_p6_sp2_UB]

    #Add dimensions to prepare for Minkowski sum
    #p6 is a function of x6, x7. For sp1, add dimension for x7
    #Set zeroval to be zero
    x1_p6_sp1_LB_ll = addDim(x1_p6_sp1_LB_l, 2, 0.0) #for x7, index 2
    x1_p6_sp1_UB_ll = addDim(x1_p6_sp1_UB_l, 2, 0.0) #for x7, index 2
    
    #For sp2, add dimension for x6
    x1_p6_sp2_LB_ll = addDim(x1_p6_sp2_LB_l, 1, 0.0) #for x6, index 1
    x1_p6_sp2_UB_ll = addDim(x1_p6_sp2_UB_l, 1, 0.0) #for x6, index 1

    #Combine sp1 and sp2 to get log(x6*sin(x7))
    x1_p6_LB_l = MinkSum(x1_p6_sp1_LB_ll, x1_p6_sp2_LB_ll)
    x1_p6_UB_l = MinkSum(x1_p6_sp1_UB_ll, x1_p6_sp2_UB_ll)

    #Compute exp to get bounds for x6*sin(x7)
    x1_p6_LB_s = [(tup[1:end-1]..., floor_n(exp(tup[end]))) for tup in x1_p6_LB_l]
    x1_p6_UB_s = [(tup[1:end-1]..., ceil_n(exp(tup[end]))) for tup in x1_p6_UB_l]

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p6_LB = []
    x1_p6_UB = []

    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p6sp1, b is s_x1p6sp2
    #f1 is x6[x1_p6_sp1], f2 is sin(x7)[x1_p6_sp2]

    for tup in x1_p6_UB_s
        #Index 1 is x6, its bounds are in x1_p6_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p6_sp1_LB)[1]
        #Index 2 is x7, its bounds are in x1_p6_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p6_sp2_LB)[1]
        f1f2_UB = tup[end] - x1_p6_sp1_LB[ind1][end]*s_x1p6sp2 - x1_p6_sp2_LB[ind2][end]*s_x1p6sp1 - s_x1p6sp1*s_x1p6sp2

        push!(x1_p6_UB, (tup[1:end-1]..., f1f2_UB))
    end

    for tup in x1_p6_LB_s
        #Index 1 is x6, its bounds are in x1_p6_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p6_sp1_UB)[1]
        #Index 2 is x7, its bounds are in x1_p6_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p6_sp2_UB)[1]
        f1f2_LB = tup[end] - x1_p6_sp1_UB[ind1][end]*s_x1p6sp2 - x1_p6_sp2_UB[ind2][end]*s_x1p6sp1 - s_x1p6sp1*s_x1p6sp2

        push!(x1_p6_LB, (tup[1:end-1]..., f1f2_LB))
    end

    if sanityFlag
        validBounds(:(x6*sin(x7)), [:x6, :x7], x1_p6_LB, x1_p6_UB, true)
    end
    #Part 7: f(x5, x7) = x5cos(x7)
    #K-A decomposition is exp(log(x5) + log(cos(x7)))
    #Sub-part 1 = x5
    x1_p7_sp1 = :(1*x)
    lb_x1_p7_sp1 = lbs[5]
    ub_x1_p7_sp1 = ubs[5]
    #Specify digits for interpolation
    x1_p7_sp1_LB, x1_p7_sp1_UB = interpol_nd(bound_univariate(x1_p7_sp1, lb_x1_p7_sp1, ub_x1_p7_sp1)...)

    #Sub-part 2 = cos(x7)
    x1_p7_sp2 = :(cos(x))
    lb_x1_p7_sp2 = lbs[7]
    ub_x1_p7_sp2 = ubs[7]
    if ub_x1_p7_sp2 - lb_x1_p7_sp2 < 1e-5
        lb_x1_p7_sp2 = lb_x1_p7_sp2 - 1e-5
        ub_x1_p7_sp2 = ub_x1_p7_sp2 + 1e-5
        lbs[7] = lb_x1_p7_sp2
        ubs[7] = ub_x1_p7_sp2
    end
    #Specify digits for interpolation
    x1_p7_sp2_LB, x1_p7_sp2_UB = interpol_nd(bound_univariate(x1_p7_sp2, lb_x1_p7_sp2, ub_x1_p7_sp2)...)

    #Find how much to shift each pair of bounds by for valid log
    s_x1p7sp1 = inpShiftLog(lb_x1_p7_sp1, ub_x1_p7_sp1, bounds=x1_p7_sp1_LB)
    s_x1p7sp2 = inpShiftLog(lb_x1_p7_sp2, ub_x1_p7_sp2, bounds=x1_p7_sp2_LB)

    #Apply log
    x1_p7_sp1_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p7sp1)) for tup in x1_p7_sp1_LB]
    x1_p7_sp1_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p7sp1)) for tup in x1_p7_sp1_UB]

    x1_p7_sp2_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p7sp2)) for tup in x1_p7_sp2_LB]
    x1_p7_sp2_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p7sp2)) for tup in x1_p7_sp2_UB]

    #Add dimensions to prepare for Minkowski sum
    #p7 is a function of x5, x7. For sp1, add dimension for x7
    #Set zeroval to be zero
    x1_p7_sp1_LB_ll = addDim(x1_p7_sp1_LB_l, 2, 0.0) #for x7, index 2
    x1_p7_sp1_UB_ll = addDim(x1_p7_sp1_UB_l, 2, 0.0) #for x7, index 2

    #For sp2, add dimension for x5
    
    x1_p7_sp2_LB_ll = addDim(x1_p7_sp2_LB_l, 1, 0.0) #for x5, index 1
    x1_p7_sp2_UB_ll = addDim(x1_p7_sp2_UB_l, 1, 0.0) #for x5, index 1

    #Combine sp1 and sp2 to get log(x5*cos(x7))
    x1_p7_LB_l = MinkSum(x1_p7_sp1_LB_ll, x1_p7_sp2_LB_ll)
    x1_p7_UB_l = MinkSum(x1_p7_sp1_UB_ll, x1_p7_sp2_UB_ll)

    #Compute exp to get bounds for x5*cos(x7)
    x1_p7_LB_s = [(tup[1:end-1]..., floor_n(exp(tup[end]))) for tup in x1_p7_LB_l]
    x1_p7_UB_s = [(tup[1:end-1]..., ceil_n(exp(tup[end]))) for tup in x1_p7_UB_l]

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p7_LB = []
    x1_p7_UB = []

    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p7sp1, b is s_x1p7sp2
    #f1 is x5[x1_p7_sp1], f2 is cos(x7)[x1_p7_sp2]

    for tup in x1_p7_UB_s
        #Index 1 is x5, its bounds are in x1_p7_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p7_sp1_LB)[1]
        #Index 2 is x7, its bounds are in x1_p7_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p7_sp2_LB)[1]
        f1f2_UB = tup[end] - x1_p7_sp1_LB[ind1][end]*s_x1p7sp2 - x1_p7_sp2_LB[ind2][end]*s_x1p7sp1 - s_x1p7sp1*s_x1p7sp2

        push!(x1_p7_UB, (tup[1:end-1]..., f1f2_UB))
    end

    for tup in x1_p7_LB_s
        #Index 1 is x5, its bounds are in x1_p7_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p7_sp1_UB)[1]
        #Index 2 is x7, its bounds are in x1_p7_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p7_sp2_UB)[1]
        f1f2_LB = tup[end] - x1_p7_sp1_UB[ind1][end]*s_x1p7sp2 - x1_p7_sp2_UB[ind2][end]*s_x1p7sp1 - s_x1p7sp1*s_x1p7sp2

        push!(x1_p7_LB, (tup[1:end-1]..., f1f2_LB))
    end

    if sanityFlag
        validBounds(:(x5*cos(x7)), [:x5, :x7], x1_p7_LB, x1_p7_UB, true)
    end
    
    #Now each chunk is bounded. Next, we need to combine chunks 
    #Combine part 6 and part 7 to get f₈(x₅, x₆, x₇)
    #First, lift f6 to be a function of x₅ as well 
    emptyList = [1] #Since x₅ comes before 6 and 7
    currList = [2,3]
    lbList = lbs[5:7]
    ubList = ubs[5:7]

    l_x1_p6_LB, l_x1_p6_UB = lift_OA(emptyList, currList, x1_p6_LB, x1_p6_UB, lbList, ubList)

    #Similarly lift f7 to be a function of x₆ as well
    emptyList = [2] #Since x₆ comes before 7 and after 5
    currList = [1,3]
    
    l_x1_p7_LB, l_x1_p7_UB = lift_OA(emptyList, currList, x1_p7_LB, x1_p7_UB, lbList, ubList)

    #Now, combine the two lifted chunks to get f₈(x₅, x₆, x₇)
    x1_p8_LB, x1_p8_UB = sumBounds(l_x1_p6_LB, l_x1_p6_UB, l_x1_p7_LB, l_x1_p7_UB, true)

    if sanityFlag
        validBounds(:(x6*sin(x7) - x5*cos(x7)), [:x5, :x6, :x7], x1_p8_LB, x1_p8_UB, true)
    end

    #Next, combine f₅ with f₈ to get f₉(x₅, x₆, x₇, x₉)
    #Lift f5 to be a function of x₅, x₆, x₇
    emptyList = [1,2,3] #Since x₅, x₆, x₇ come before 9
    currList = [4]
    lbList = lbs[5:7]
    push!(lbList, lbs[9])
    ubList = ubs[5:7]
    push!(ubList, ubs[9])

    l_x1_p5_LB, l_x1_p5_UB = lift_OA(emptyList, currList, x1_p5_LB, x1_p5_UB, lbList, ubList)

    #Then lift f8 to be a function of x₉
    emptyList = [4] #Since x₉ comes after 5, 6, 7
    currList = [1,2,3]

    l_x1_p8_LB, l_x1_p8_UB = lift_OA(emptyList, currList, x1_p8_LB, x1_p8_UB, lbList, ubList)
    
    #Define a function to compute the product of two sets of bounds 
    x1_p9_LB, x1_p9_UB = prodBounds(l_x1_p5_LB, l_x1_p5_UB, l_x1_p8_LB, l_x1_p8_UB)

    if sanityFlag
        validBounds(:(sin(x9)*((x6*sin(x7) - x5*cos(x7)))), [:x5, :x6, :x7, :x9], x1_p9_LB, x1_p9_UB, true)
    end

    #Next, define f₁₀(x₅, x₆, x₇) as f₃(x₅, x₇) + f₄(x₆, x₇)
    #First, lift f₃(x₅, x₇) = x5*sin(x7) to be a function of x6 as well
    emptyList = [2] #Since x₆ comes after 5 and before 7
    currList = [1,3]
    lbList = lbs[5:7]
    ubList = ubs[5:7]

    l_x1_p3_LB, l_x1_p3_UB = lift_OA(emptyList, currList, x1_p3_LB, x1_p3_UB, lbList, ubList)

    #Then lift f₄(x₆, x₇) = x6*cos(x7) to be a function of x₅ as well
    emptyList = [1] #Since x₅ comes before 6 and 7
    currList = [2,3]

    l_x1_p4_LB, l_x1_p4_UB = lift_OA(emptyList, currList, x1_p4_LB, x1_p4_UB, lbList, ubList)

    #Combine the two lifted chunks to get f₁₀(x₅, x₆, x₇)
    x1_p10_LB, x1_p10_UB = sumBounds(l_x1_p3_LB, l_x1_p3_UB, l_x1_p4_LB, l_x1_p4_UB, false)

    if sanityFlag
        validBounds(:(x5*sin(x7) + x6*cos(x7)), [:x5, :x6, :x7], x1_p10_LB, x1_p10_UB, true)
    end

    #Next define f₁₁(x₅, x₆, x₇, x₈, x₉) as f₂(x₈, x₉) * f₁₀(x₅, x₆, x₇)
    #First, lift f₂(x₈, x₉) = sin(x8)cos(x9) to be a function of x₅, x₆, x₇
    emptyList = [1,2,3] #Since x₅, x₆, x₇ come before 8 and 9
    currList = [4,5]
    lbList = lbs[5:9]
    ubList = ubs[5:9]

    l_x1_p2_LB, l_x1_p2_UB = lift_OA(emptyList, currList, x1_p2_LB, x1_p2_UB, lbList, ubList)
    
    #Then lift f₁₀(x₅, x₆, x₇) to be a function of x₈, x₉
    emptyList = [4,5] #Since x₈, x₉ come after 5, 6, 7
    currList = [1,2,3]

    l_x1_p10_LB, l_x1_p10_UB = lift_OA(emptyList, currList, x1_p10_LB, x1_p10_UB, lbList, ubList)

    #Combine the two lifted chunks to get f₁₁(x₅, x₆, x₇, x₈, x₉)
    x1_p11_LB, x1_p11_UB = prodBounds(l_x1_p2_LB, l_x1_p2_UB, l_x1_p10_LB, l_x1_p10_UB)

    if sanityFlag
        validBounds(:(sin(x8)*cos(x9)*(x5*sin(x7) + x6*cos(x7))), [:x5, :x6, :x7, :x8, :x9], x1_p11_LB, x1_p11_UB, true)
    end
    
    #Combine 3 distinct chunks to get f₁₂(x₄, x₅, x₆, x₇, x₈, x₉) = f₁(x₄, x₈, x₉) + f₁₁(x₅, x₆, x₇, x₈, x₉) + f₉(x₅, x₆, x₇, x₉)
    
    #First lift f₁(x₄, x₈, x₉) = x4*cos(x8)*cos(x9) to be a function of x₅, x₆, x₇
    emptyList = [2,3,4] #Since x₅, x₆, x₇ come before 4 and 8, 9
    currList = [1,5,6]
    lbList = lbs[4:9]
    ubList = ubs[4:9]

    l_x1_p1_LB, l_x1_p1_UB = lift_OA(emptyList, currList, x1_p1_LB, x1_p1_UB, lbList, ubList)
    
    #Then lift f₁₁(x₅, x₆, x₇, x₈, x₉) to be a function of x₄
    emptyList = [1] #Since x₄ comes before 5, 6, 7, 8, 9
    currList = [2,3,4,5,6]

    l_x1_p11_LB, l_x1_p11_UB = lift_OA(emptyList, currList, x1_p11_LB, x1_p11_UB, lbList, ubList)
    
    #Finally, lift f₉(x₅, x₆, x₇, x₉) to be a function of x₄ and x₈
    emptyList = [1,5] #Since x₄ comes before 5, 6, 7, 9 and x₈ comes after 7
    currList = [2,3,4,6]

    l_x1_p9_LB, l_x1_p9_UB = lift_OA(emptyList, currList, x1_p9_LB, x1_p9_UB, lbList, ubList)

    #Combine the three lifted chunks to get f₁₂(x₄, x₅, x₆, x₇, x₈, x₉)
    x1_p12_LB_i, x1_p12_UB_i = sumBounds(l_x1_p1_LB, l_x1_p1_UB, l_x1_p11_LB, l_x1_p11_UB, false)
    x1_p12_LB, x1_p12_UB = sumBounds(x1_p12_LB_i, x1_p12_UB_i, l_x1_p9_LB, l_x1_p9_UB, false)
    
    if sanityFlag
        validBounds(:(x4*cos(x8)*cos(x9) + sin(x8)*cos(x9)*(x5*sin(x7) + x6*cos(x7)) + sin(x9)*((x6*sin(x7) - x5*cos(x7)))), [:x4, :x5, :x6, :x7, :x8, :x9], x1_p12_LB, x1_p12_UB, true)
    end
    return x1_p12_LB, x1_p12_UB
end

function bound_quadx2(Quad, plotFlag = false, sanityFlag = true)
    lbs, ubs = extrema(Quad.domain)

    #Follow a similar strategy to the x1 bounds. Break the initial function into 7 parts then combine parts to regain full bounds 

    #Part 1: f₁(x₄, x₈, x₉) = x₄*cos(x₈)*cos(x₉)
    #K-A decomposition is exp(log(x₄) + log(cos(x₈)) + log(cos(x₉)))

    #Sub-part 1 = x₄
    x2_p1_sp1 = :(1*x)
    lb_x2_p1_sp1 = lbs[4]
    ub_x2_p1_sp1 = ubs[4]

    x2_p1_sp1_LB, x2_p1_sp1_UB = interpol_nd(bound_univariate(x2_p1_sp1, lb_x2_p1_sp1, ub_x2_p1_sp1)...)

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

    x2_p1_sp2_LB, x2_p1_sp2_UB = interpol_nd(bound_univariate(x2_p1_sp2, lb_x2_p1_sp2, ub_x2_p1_sp2)...)

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

    x2_p1_sp3_LB, x2_p1_sp3_UB = interpol_nd(bound_univariate(x2_p1_sp3, lb_x2_p1_sp3, ub_x2_p1_sp3)...)

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

    if sanityFlag
        validBounds(:(x4*cos(x8)*cos(x9)), [:x4, :x8, :x9], x2_p1_LB, x2_p1_UB)
    end


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

    x2_p2_sp1_LB, x2_p2_sp1_UB = interpol_nd(bound_univariate(x2_p2_sp1, lb_x2_p2_sp1, ub_x2_p2_sp1)...)

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

    x2_p2_sp2_LB, x2_p2_sp2_UB = interpol_nd(bound_univariate(x2_p2_sp2, lb_x2_p2_sp2, ub_x2_p2_sp2)...)

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

    if sanityFlag
        validBounds(:(sin(x8)*sin(x9)), [:x8, :x9], x2_p2_LB, x2_p2_UB)
    end

   #Part 3: f₃(x₅,x₇) = x₅*sin(x₇) 
   #Sub part 1: x₅
    x2_p3_sp1 = :(1*x)
    lb_x2_p3_sp1 = lbs[5]
    ub_x2_p3_sp1 = ubs[5]

    x2_p3_sp1_LB, x2_p3_sp1_UB = interpol_nd(bound_univariate(x2_p3_sp1, lb_x2_p3_sp1, ub_x2_p3_sp1)...)

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

    x2_p3_sp2_LB, x2_p3_sp2_UB = interpol_nd(bound_univariate(x2_p3_sp2, lb_x2_p3_sp2, ub_x2_p3_sp2)...)

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

    if sanityFlag
        validBounds(:(x5*sin(x7)), [:x5, :x7], x2_p3_LB, x2_p3_UB)
    end

    #Part 4: f₄(x₅,x₇) = x₆*cos(x₇)
    #Sub part 1: x₆
    x2_p4_sp1 = :(1*x)
    lb_x2_p4_sp1 = lbs[6]
    ub_x2_p4_sp1 = ubs[6]

    x2_p4_sp1_LB, x2_p4_sp1_UB = interpol_nd(bound_univariate(x2_p4_sp1, lb_x2_p4_sp1, ub_x2_p4_sp1)...)

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

    x2_p4_sp2_LB, x2_p4_sp2_UB = interpol_nd(bound_univariate(x2_p4_sp2, lb_x2_p4_sp2, ub_x2_p4_sp2)...)

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

    if sanityFlag
        validBounds(:(x6*cos(x7)), [:x6, :x7], x2_p4_LB, x2_p4_UB)
    end

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

    x2_p5_LB, x2_p5_UB = interpol_nd(bound_univariate(x2_p5, lb_x2_p5, ub_x2_p5)...)

    if sanityFlag
        validBounds(:(cos(x9)), [:x9], x2_p5_LB, x2_p5_UB)
    end

    #Part 6: f₆(x₅, x₇) = x₅*cos(x₇)
    #Sub part 1: x₅
    x2_p6_sp1 = :(1*x)
    lb_x2_p6_sp1 = lbs[5]
    ub_x2_p6_sp1 = ubs[5]

    x2_p6_sp1_LB, x2_p6_sp1_UB = interpol_nd(bound_univariate(x2_p6_sp1, lb_x2_p6_sp1, ub_x2_p6_sp1)...)

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

    x2_p6_sp2_LB, x2_p6_sp2_UB = interpol_nd(bound_univariate(x2_p6_sp2, lb_x2_p6_sp2, ub_x2_p6_sp2)...)

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

    if sanityFlag
        validBounds(:(x5*cos(x7)), [:x5, :x7], x2_p6_LB, x2_p6_UB)
    end
   
    #Part 7: f₇(x₅, x₇) = x₆*sin(x₇)
    #Sub part 1: x₆
    x2_p7_sp1 = :(1*x)
    lb_x2_p7_sp1 = lbs[6]
    ub_x2_p7_sp1 = ubs[6]

    x2_p7_sp1_LB, x2_p7_sp1_UB = interpol_nd(bound_univariate(x2_p7_sp1, lb_x2_p7_sp1, ub_x2_p7_sp1)...)

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

    x2_p7_sp2_LB, x2_p7_sp2_UB = interpol_nd(bound_univariate(x2_p7_sp2, lb_x2_p7_sp2, ub_x2_p7_sp2)...)

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
   
    if sanityFlag
        validBounds(:(x6*sin(x7)), [:x6, :x7], x2_p7_LB, x2_p7_UB)
    end
    
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

    if sanityFlag
        validBounds(:(x5*cos(x7) - x6*sin(x7)), [:x5, :x6, :x7], x2_p8_LB, x2_p8_UB)
    end

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

    if sanityFlag
        validBounds(:(cos(x9)*(x5*cos(x7) - x6*sin(x7))), [:x5, :x6, :x7, :x9], x2_p9_LB, x2_p9_UB)
    end

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

    if sanityFlag
        validBounds(:(x5*sin(x7) + x6*cos(x7)), [:x5, :x6, :x7], x2_p10_LB, x2_p10_UB)
    end
    
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

    if sanityFlag
        validBounds(:(sin(x8)*sin(x9)*(x5*sin(x7) + x6*cos(x7))), [:x5, :x6, :x7, :x8, :x9], x2_p11_LB, x2_p11_UB)
    end
   
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
        validBounds(:(x4*cos(x8)*cos(x9) + sin(x8)*sin(x9)*(x5*sin(x7) + x6*cos(x7)) + cos(x9)*(x5*cos(x7) - x6*sin(x7))), [:x4, :x5, :x6, :x7, :x8, :x9], x2_p12_LB, x2_p12_UB)
    end
    return x2_p12_LB, x2_p12_UB
end

function bound_quadx3(Quad, plotFlag = false, sanityFlag = true)

    lbs, ubs = extrema(Quad.domain)

    #Part 1: f₁(x₄, x₈) = x₄*sin(x₈)
    #Sub-part 1: x₄
    x3_p1_sp1 = :(1*x)
    lb_x3_p1_sp1 = lbs[4]
    ub_x3_p1_sp1 = ubs[4]

    x3_p1_sp1_LB, x3_p1_sp1_UB = interpol_nd(bound_univariate(x3_p1_sp1, lb_x3_p1_sp1, ub_x3_p1_sp1)...)

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

    x3_p1_sp2_LB, x3_p1_sp2_UB = interpol_nd(bound_univariate(x3_p1_sp2, lb_x3_p1_sp2, ub_x3_p1_sp2)...)

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

    if sanityFlag
        validBounds(:(x₄*sin(x₈)), [:x₄, :x₈], x3_p1_LB, x3_p1_UB)
    end
    
    #Part 2: f₂(x₅, x₇, x₈) = x₅*sin(x₇)*cos(x₈) 
    #Sub-part 1: x₅
    x3_p2_sp1 = :(1*x)
    lb_x3_p2_sp1 = lbs[5]
    ub_x3_p2_sp1 = ubs[5]

    x3_p2_sp1_LB, x3_p2_sp1_UB = interpol_nd(bound_univariate(x3_p2_sp1, lb_x3_p2_sp1, ub_x3_p2_sp1)...)

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

    x3_p2_sp2_LB, x3_p2_sp2_UB = interpol_nd(bound_univariate(x3_p2_sp2, lb_x3_p2_sp2, ub_x3_p2_sp2)...)
    
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

    x3_p2_sp3_LB, x3_p2_sp3_UB = interpol_nd(bound_univariate(x3_p2_sp3, lb_x3_p2_sp3, ub_x3_p2_sp3)...)

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

    if sanityFlag
        validBounds(:(x₅*sin(x₇)*cos(x₈)), [:x₅, :x₇, :x₈], x3_p2_LB, x3_p2_UB)
    end
    
    #Part 3: f₃(x₆,x₇,x₈) = x₆*cos(x₇)*cos(x₈)
    #Sub-part 1: x₆
    x3_p3_sp1 = :(1*x)
    lb_x3_p3_sp1 = lbs[6]
    ub_x3_p3_sp1 = ubs[6]

    x3_p3_sp1_LB, x3_p3_sp1_UB = interpol_nd(bound_univariate(x3_p3_sp1, lb_x3_p3_sp1, ub_x3_p3_sp1)...)

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

    x3_p3_sp2_LB, x3_p3_sp2_UB = interpol_nd(bound_univariate(x3_p3_sp2, lb_x3_p3_sp2, ub_x3_p3_sp2)...)

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

    x3_p3_sp3_LB, x3_p3_sp3_UB = interpol_nd(bound_univariate(x3_p3_sp3, lb_x3_p3_sp3, ub_x3_p3_sp3)...)
    
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

    if sanityFlag
        validBounds(:(x₆*cos(x₇)*cos(x₈)), [:x₆, :x₇, :x₈], x3_p3_LB, x3_p3_UB)
    end

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

    return x3_p4_LB, x3_p4_UB
end

function bound_quadx4(Quad, plotFlag = false, sanityFlag = true)
    lbs, ubs = extrema(Quad.domain)

    #Bounding x₅*x₁₂ -x₆*x₁₁ - g*sin(x₈)
    #Part 1: f₁(x₅,x₁₂) = x₅*x₁₂
    #Sub-part 1: x₅
    x4_p1_sp1 = :(1*x)
    lb_x4_p1_sp1 = lbs[5]
    ub_x4_p1_sp1 = ubs[5]

    x4_p1_sp1_LB, x4_p1_sp1_UB = interpol_nd(bound_univariate(x4_p1_sp1, lb_x4_p1_sp1, ub_x4_p1_sp1)...)

    #Sub-part 2: x₁₂
    x4_p1_sp2 = :(1*x)
    lb_x4_p1_sp2 = lbs[12]
    ub_x4_p1_sp2 = ubs[12]

    x4_p1_sp2_LB, x4_p1_sp2_UB = interpol_nd(bound_univariate(x4_p1_sp2, lb_x4_p1_sp2, ub_x4_p1_sp2)...)

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

    if sanityFlag
        validBounds(:(x₅*x₁₂),[:x₅,:x₁₂], x4_p1_LB, x4_p1_UB)
    end

    #Part 2: f₂(x₆, x₁₁) = x₆*x₁₁
    #Sub-part 1: x₆
    x4_p2_sp1 = :(1*x)
    lb_x4_p2_sp1 = lbs[6]
    ub_x4_p2_sp1 = ubs[6]

    x4_p2_sp1_LB, x4_p2_sp1_UB = interpol_nd(bound_univariate(x4_p2_sp1, lb_x4_p2_sp1, ub_x4_p2_sp1)...)

    #Sub-part 2: x₁₁
    x4_p2_sp2 = :(1*x)
    lb_x4_p2_sp2 = lbs[11]
    ub_x4_p2_sp2 = ubs[11]

    x4_p2_sp2_LB, x4_p2_sp2_UB = interpol_nd(bound_univariate(x4_p2_sp2, lb_x4_p2_sp2, ub_x4_p2_sp2)...)

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

    if sanityFlag
        validBounds(:(x₆*x₁₁),[:x₆,:x₁₁], x4_p2_LB, x4_p2_UB)
    end

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
        validBounds(:(x₅*x₁₂ - x₆*x₁₁ - $g*sin(x₈)),[:x₅,:x₆,:x₈,:x₁₁,:x₁₂], x4_LB, x4_UB,true)
    end
    return x4_LB, x4_UB
end

function bound_quadx5(Quad, plotFlag = false, sanityFlag = true)
    lbs, ubs = extrema(Quad.domain)

    #Bounding x₆*x₁₀ - x₄*x₁₂ + g*sin(x₇)*cos(x₈)
    #Part 1: x₆*x₁₀
    #Sub-part 1: x₆
    x5_p1_sp1 = :(1*x)
    lb_x5_p1_sp1 = lbs[6]
    ub_x5_p1_sp1 = ubs[6]

    x5_p1_sp1_LB, x5_p1_sp1_UB = interpol_nd(bound_univariate(x5_p1_sp1, lb_x5_p1_sp1, ub_x5_p1_sp1)...)

    #Sub-part 2: x₁₀
    x5_p1_sp2 = :(1*x)
    lb_x5_p1_sp2 = lbs[10]
    ub_x5_p1_sp2 = ubs[10]

    x5_p1_sp2_LB, x5_p1_sp2_UB = interpol_nd(bound_univariate(x5_p1_sp2, lb_x5_p1_sp2, ub_x5_p1_sp2)...)

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

    if sanityFlag
        validBounds(:(x6*x10), [:x6, :x10], x5_p1_LB, x5_p1_UB)
    end


    #Part 2: x₄*x₁₂
    #Sub-part 1: x₄
    x5_p2_sp1 = :(1*x)
    lb_x5_p2_sp1 = lbs[4]
    ub_x5_p2_sp1 = ubs[4]

    x5_p2_sp1_LB, x5_p2_sp1_UB = interpol_nd(bound_univariate(x5_p2_sp1, lb_x5_p2_sp1, ub_x5_p2_sp1)...)

    #Sub-part 2: x₁₂
    x5_p2_sp2 = :(1*x)
    lb_x5_p2_sp2 = lbs[12]
    ub_x5_p2_sp2 = ubs[12]

    x5_p2_sp2_LB, x5_p2_sp2_UB = interpol_nd(bound_univariate(x5_p2_sp2, lb_x5_p2_sp2, ub_x5_p2_sp2)...)

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

    if sanityFlag
        validBounds(:(x4*x12), [:x4, :x12], x5_p2_LB, x5_p2_UB)
    end

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

    x5_p3_sp2_LB, x5_p3_sp2_UB = interpol_nd(bound_univariate(x5_p3_sp2, lb_x5_p3_sp2, ub_x5_p3_sp2)...)

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

    if sanityFlag
        validBounds(:($g*sin(x7)*cos(x8)), [:x7, :x8], x5_p3_LB, x5_p3_UB)
    end

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
        validBounds(:($g*sin(x7)*cos(x8) + x6*x10 - x4*x12), [:x4, :x6, :x7, :x8, :x10, :x12], x5_LB, x5_UB)
    end

    return x5_LB, x5_UB
end

function bound_bound_quadx6(Quad, plotFlag, sanityFlag)
    lbs, ubs = extrema(Quad.domain)

    #Bounding x₄*x₁₁ - x₅*x₁₀ + g*cos(x₇)*cos(x₈) - g
    #Part 1: x₄*x₁₁
    #Sub-part 1: x₄
    x6_p1_sp1 = :(1*x)
    lb_x6_p1_sp1 = lbs[4]
    ub_x6_p1_sp1 = ubs[4]

    x6_p1_sp1_LB, x6_p1_sp1_UB = interpol_nd(bound_univariate(x6_p1_sp1, lb_x6_p1_sp1, ub_x6_p1_sp1)...)

    #Sub-part 2: x₁₁
    x6_p1_sp2 = :(1*x)
    lb_x6_p1_sp2 = lbs[11]
    ub_x6_p1_sp2 = ubs[11]

    x6_p1_sp2_LB, x6_p1_sp2_UB = interpol_nd(bound_univariate(x6_p1_sp2, lb_x6_p1_sp2, ub_x6_p1_sp2)...)

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

    if sanityFlag
        validBounds(:(x4*x11), [:x4, :x11], x6_p1_LB, x6_p1_UB)
    end

    #Part 2: x₅*x₁₀
    #Sub-part 1: x₅
    x6_p2_sp1 = :(1*x)
    lb_x6_p2_sp1 = lbs[5]
    ub_x6_p2_sp1 = ubs[5]

    x6_p2_sp1_LB, x6_p2_sp1_UB = interpol_nd(bound_univariate(x6_p2_sp1, lb_x6_p2_sp1, ub_x6_p2_sp1)...)

    #Sub-part 2: x₁₀
    x6_p2_sp2 = :(1*x)
    lb_x6_p2_sp2 = lbs[10]
    ub_x6_p2_sp2 = ubs[10]

    x6_p2_sp2_LB, x6_p2_sp2_UB = interpol_nd(bound_univariate(x6_p2_sp2, lb_x6_p2_sp2, ub_x6_p2_sp2)...)

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

    if sanityFlag
        validBounds(:(x5*x10), [:x5, :x10], x6_p2_LB, x6_p2_UB)
    end

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

    x6_p3_sp1_LB, x6_p3_sp1_UB = interpol_nd(bound_univariate(x6_p3_sp1, lb_x6_p3_sp1, ub_x6_p3_sp1)...)

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

    x6_p3_sp2_LB, x6_p3_sp2_UB = interpol_nd(bound_univariate(x6_p3_sp2, lb_x6_p3_sp2, ub_x6_p3_sp2)...)

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

    if sanityFlag
        validBounds(:($g*cos(x7)*cos(x8) - $g), [:x7, :x8], x6_p3_LB, x6_p3_UB)
    end

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
        validBounds(:($g*cos(x7)*cos(x8) - $g + x4*x11 - x5*x10), [:x4, :x5, :x7, :x8, :x10, :x11], x6_LB, x6_UB)
    end
    return x6_LB, x6_UB
end