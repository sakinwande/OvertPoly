include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

#Define constants for the quadrotor
g = 9.81
m = 1.4
Jx = 0.054
Jy = 0.054
Jz = 0.104
τ = 0

#Define the control coefficients for each axis
control_coef = [[0], [0], [0],[0], [0], [-1/m],[0], [0], [0],[1/Jx], [1/Jy], [0]]
exprList = [] #Empty bc what's the point? We can't plot anyway
controller = "Networks/ARCH-COMP-2023/nnet/controllerQuad.nnet"

dt = 0.1
ϵ = 1e-8
domain = Hyperrectangle(low=[-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-ϵ,-ϵ,-ϵ,-ϵ,-ϵ,-ϵ], high=[0.4,0.4,0.4,0.4,0.4,0.4,ϵ,ϵ,ϵ,ϵ,ϵ,ϵ])
numsteps = 50
sigFigs = 12

####Define quad dynamics#####
function quad_dynamics(x, u)
    """
    Dynamics of the quad benchmark 

    Args:
        x: 12d state of the system
        u: 3d control input
    """
    dx1 = cos(x[8])*cos(x[9])*x[4] + (sin(x[7])*sin(x[8])*cos(x[9]) - cos(x[7])*sin(x[9]))*x[5] + (cos(x[7])*sin(x[8])*cos(x[9]) + sin(x[7])*sin(x[9]))*x[6]
    dx2 = cos(x[8])*sin(x[9])*x[4] + (sin(x[7])*sin(x[8])*sin(x[9]) + cos(x[7])*cos(x[9]))*x[5] + (cos(x[7])*sin(x[8])*sin(x[9]) - sin(x[7])*cos(x[9]))*x[6]
    dx3 = sin(x[8])*x[4] - sin(x[7])*cos(x[8])*x[5] - cos(x[7])*cos(x[8])*x[6]
    dx4 = x[12]*x[5] - x[11]*x[6] - g*sin(x[8])
    dx5 = x[10]*x[6] - x[12]*x[4] + g*cos(x[8])*sin(x[7])
    dx6 = x[11]*x[4] - x[10]*x[5] + g*cos(x[8])*cos(x[7]) - u[1]/m -g
    dx7 = x[10] + sin(x[7])*tan(x[8])*x[11] + cos(x[7])*tan(x[8])*x[12]
    dx8 = cos(x[7])*x[11] - sin(x[7])*x[12]
    dx9 = (sin(x[7])/cos(x[8]))*x[11] - (cos(x[7])/cos(x[8]))*x[12]
    dx10 = u[2]/Jx - ((Jz - Jy)/Jx)*x[11]*x[12]
    dx11 = u[3]/Jy - ((Jx - Jz)/Jy)*x[10]*x[12]
    dx12 = τ/Jz + ((Jx - Jy)/Jz)*x[10]*x[11]

    xNew = x + [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12].*dt
    return xNew
end

#Define control prep function 
function quad_control(input_set)
    con_inp_set = input_set
    return con_inp_set
end

#Bounding the quadcopter dynamics
sanityFlag = true
lbs, ubs = extrema(domain)
# function bound_quad(QUAD, plotFlag = false)
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
    x1_p1_LB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p1_LB_l]
    x1_p1_UB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p1_UB_l]

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

    #Compute exp to get bounds for sin(x8)*cos(x9)
    x1_p2_LB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p2_LB_l]
    x1_p2_UB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p2_UB_l]

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
    end
    x1_p3_sp1_LB, x1_p3_sp1_UB = interpol_nd(bound_univariate(x1_p3_sp1, lb_x1_p3_sp1, ub_x1_p3_sp1)...)

    #Sub-part 2 = x5
    x1_p3_sp2 = :(1*x)
    lb_x1_p3_sp2 = lbs[5]
    ub_x1_p3_sp2 = ubs[5]
    #Specify digits for interpolation
    x1_p3_sp2_LB, x1_p3_sp2_UB = interpol_nd(bound_univariate(x1_p3_sp2, lb_x1_p3_sp2, ub_x1_p3_sp2)...)

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
    x1_p3_LB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p3_LB_l]
    x1_p3_UB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p3_UB_l]

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p3_LB = []
    x1_p3_UB = []

    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p3sp1, b is s_x1p3sp2
    #f1 is sin(x7)[x1_p3_sp1], f2 is x5[x1_p3_sp2]
    for tup in x1_p3_LB_s
        #Index 1 is x5, its bounds are in x1_p3_sp2
        ind2 = findall(x->x[1] == tup[1], x1_p3_sp2_LB)[1]
        #Index 2 is x7, its bounds are in x1_p3_sp1
        ind1 = findall(x->x[1] == tup[2], x1_p3_sp1_LB)[1]
        f1f2_UB = tup[end] - x1_p3_sp1_LB[ind1][end]*s_x1p3sp2 - x1_p3_sp2_LB[ind2][end]*s_x1p3sp1 - s_x1p3sp1*s_x1p3sp2

        push!(x1_p3_UB, (tup[1:end-1]..., f1f2_UB))
    end

    for tup in x1_p3_UB_s
        #Index 1 is x5, its bounds are in x1_p3_sp2
        ind2 = findall(x->x[1] == tup[1], x1_p3_sp2_UB)[1]
        #Index 2 is x7, its bounds are in x1_p3_sp1
        ind1 = findall(x->x[1] == tup[2], x1_p3_sp1_UB)[1]
        f1f2_LB = tup[end] - x1_p3_sp1_UB[ind1][end]*s_x1p3sp2 - x1_p3_sp2_UB[ind2][end]*s_x1p3sp1 - s_x1p3sp1*s_x1p3sp2

        push!(x1_p3_LB, (tup[1:end-1]..., f1f2_LB))
    end

    #Part 4: f(x6, x7) = x6cos(x7)
    #K-A decomposition is exp(log(x6) + log(cos(x7)))
    #Sub-part 1 = x6
    x1_p4_sp1 = :(1*x)
    lb_x1_p4_sp1 = lbs[6]
    ub_x1_p4_sp1 = ubs[6]
    x1_p4_sp1_LB, x1_p4_sp1_UB = interpol_nd(bound_univariate(x1_p4_sp1, lb_x1_p4_sp1, ub_x1_p4_sp1)...)

    #Sub-part 2 = cos(x7)
    x1_p4_sp2 = :(cos(x))
    lb_x1_p4_sp2 = lbs[7]
    ub_x1_p4_sp2 = ubs[7]
    if ub_x1_p4_sp2 - lb_x1_p4_sp2 < 1e-5
        lb_x1_p4_sp2 = lb_x1_p4_sp2 - 1e-5
        ub_x1_p4_sp2 = ub_x1_p4_sp2 + 1e-5
    end
    #Specify digits for interpolation
    x1_p4_sp2_LB, x1_p4_sp2_UB = interpol_nd(bound_univariate(x1_p4_sp2, lb_x1_p4_sp2, ub_x1_p4_sp2)...)

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
    x1_p4_LB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p4_LB_l]
    x1_p4_UB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p4_UB_l]

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p4_LB = []
    x1_p4_UB = []

    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p4sp1, b is s_x1p4sp2
    #f1 is x6[x1_p4_sp1], f2 is cos(x7)[x1_p4_sp2]

    for tup in x1_p4_LB_s
        #Index 1 is x6, its bounds are in x1_p4_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p4_sp1_LB)[1]
        #Index 2 is x7, its bounds are in x1_p4_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p4_sp2_LB)[1]
        f1f2_UB = tup[end] - x1_p4_sp1_LB[ind1][end]*s_x1p4sp2 - x1_p4_sp2_LB[ind2][end]*s_x1p4sp1 - s_x1p4sp1*s_x1p4sp2

        push!(x1_p4_UB, (tup[1:end-1]..., f1f2_UB))
    end

    for tup in x1_p4_UB_s
        #Index 1 is x6, its bounds are in x1_p4_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p4_sp1_UB)[1]
        #Index 2 is x7, its bounds are in x1_p4_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p4_sp2_UB)[1]
        f1f2_LB = tup[end] - x1_p4_sp1_UB[ind1][end]*s_x1p4sp2 - x1_p4_sp2_UB[ind2][end]*s_x1p4sp1 - s_x1p4sp1*s_x1p4sp2

        push!(x1_p4_LB, (tup[1:end-1]..., f1f2_LB))
    end

    #Part 5: f(x9) = sin(x9)
    x1_p5 = :(sin(x))
    lb_x1_p5 = lbs[9]
    ub_x1_p5 = ubs[9]
    if ub_x1_p5 - lb_x1_p5 < 1e-5
        lb_x1_p5 = lb_x1_p5 - 1e-5
        ub_x1_p5 = ub_x1_p5 + 1e-5
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
    x1_p6_LB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p6_LB_l]
    x1_p6_UB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p6_UB_l]

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p6_LB = []
    x1_p6_UB = []

    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p6sp1, b is s_x1p6sp2
    #f1 is x6[x1_p6_sp1], f2 is sin(x7)[x1_p6_sp2]

    for tup in x1_p6_LB_s
        #Index 1 is x6, its bounds are in x1_p6_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p6_sp1_LB)[1]
        #Index 2 is x7, its bounds are in x1_p6_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p6_sp2_LB)[1]
        f1f2_UB = tup[end] - x1_p6_sp1_LB[ind1][end]*s_x1p6sp2 - x1_p6_sp2_LB[ind2][end]*s_x1p6sp1 - s_x1p6sp1*s_x1p6sp2

        push!(x1_p6_UB, (tup[1:end-1]..., f1f2_UB))
    end

    for tup in x1_p6_UB_s
        #Index 1 is x6, its bounds are in x1_p6_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p6_sp1_UB)[1]
        #Index 2 is x7, its bounds are in x1_p6_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p6_sp2_UB)[1]
        f1f2_LB = tup[end] - x1_p6_sp1_UB[ind1][end]*s_x1p6sp2 - x1_p6_sp2_UB[ind2][end]*s_x1p6sp1 - s_x1p6sp1*s_x1p6sp2

        push!(x1_p6_LB, (tup[1:end-1]..., f1f2_LB))
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
    x1_p7_LB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p7_LB_l]
    x1_p7_UB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p7_UB_l]

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p7_LB = []
    x1_p7_UB = []

    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p7sp1, b is s_x1p7sp2
    #f1 is x5[x1_p7_sp1], f2 is cos(x7)[x1_p7_sp2]

    for tup in x1_p7_LB_s
        #Index 1 is x5, its bounds are in x1_p7_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p7_sp1_LB)[1]
        #Index 2 is x7, its bounds are in x1_p7_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p7_sp2_LB)[1]
        f1f2_UB = tup[end] - x1_p7_sp1_LB[ind1][end]*s_x1p7sp2 - x1_p7_sp2_LB[ind2][end]*s_x1p7sp1 - s_x1p7sp1*s_x1p7sp2

        push!(x1_p7_UB, (tup[1:end-1]..., f1f2_UB))
    end

    for tup in x1_p7_UB_s
        #Index 1 is x5, its bounds are in x1_p7_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p7_sp1_UB)[1]
        #Index 2 is x7, its bounds are in x1_p7_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p7_sp2_UB)[1]
        f1f2_LB = tup[end] - x1_p7_sp1_UB[ind1][end]*s_x1p7sp2 - x1_p7_sp2_UB[ind2][end]*s_x1p7sp1 - s_x1p7sp1*s_x1p7sp2

        push!(x1_p7_LB, (tup[1:end-1]..., f1f2_LB))
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



    include("../../overtPoly_helpers.jl")