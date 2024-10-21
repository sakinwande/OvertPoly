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
    #Specify digits for interpolation 
    x1_p1_sp1_LB, x1_p1_sp1_UB = interpol(bound_univariate(x1_p1_sp1, lb_x1_p1_sp1, ub_x1_p1_sp1)...,sigFigs)

    #Sub-part 2 = cos(x9)
    x1_p1_sp2 = :(cos(x))
    lb_x1_p1_sp2 = lbs[9]
    ub_x1_p1_sp2 = ubs[9]

    if ub_x1_p1_sp2 - lb_x1_p1_sp2 < 1e-5
        lb_x1_p1_sp2 = lb_x1_p1_sp2 - 1e-5
        ub_x1_p1_sp2 = ub_x1_p1_sp2 + 1e-5
    end
    #Specify digits for interpolation
    x1_p1_sp2_LB, x1_p1_sp2_UB = interpol(bound_univariate(x1_p1_sp2, lb_x1_p1_sp2, ub_x1_p1_sp2)...,sigFigs)

    #Sub-part 3 = x4
    x1_p1_sp3 = :(1*x)
    lb_x1_p1_sp3 = lbs[4]
    ub_x1_p1_sp3 = ubs[4]
    #Specify digits for interpolation
    x1_p1_sp3_LB, x1_p1_sp3_UB = interpol(bound_univariate(x1_p1_sp3, lb_x1_p1_sp3, ub_x1_p1_sp3)...,sigFigs)

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

    #For sp2, add dimension for x4 and x8
    x1_p1_sp2_LB_ll = addDim(x1_p1_sp2_LB_l, 1, 0.0) #for x4, index 1
    x1_p1_sp2_LB_ll = addDim(x1_p1_sp2_LB_ll, 2, 0.0) #for x8 index 2

    x1_p1_sp2_UB_ll = addDim(x1_p1_sp2_UB_l, 1, 0.0) #for x4, index 1
    x1_p1_sp2_UB_ll = addDim(x1_p1_sp2_UB_ll, 2, 0.0) #for x8 index 2

    #For sp3, add dimension for x8 and x9
    x1_p1_sp3_LB_ll = addDim(x1_p1_sp3_LB_l, 2, 0.0) #for x8, index 2
    x1_p1_sp3_LB_ll = addDim(x1_p1_sp3_LB_ll, 3, 0.0) #for x9 index 3

    #Combine sp1 and sp2 first to get log(cos(x8)*cos(x9))
    x1_p1_sp4_LB_l = MinkSum(x1_p1_sp1_LB_ll, x1_p1_sp2_LB_ll)
    x1_p1_sp4_UB_l = MinkSum(x1_p1_sp1_UB_ll, x1_p1_sp2_UB_ll)

    #Combine sp4 with sp3 to get log(cos(x8)*cos(x9)*x4)
    x1_p1_LB_l = MinkSum(x1_p1_sp4_LB_l, x1_p1_sp3_LB_ll)
    x1_p1_UB_l = MinkSum(x1_p1_sp4_UB_l, x1_p1_sp3_LB_ll)

    #Compute exp to get bounds for cos(x8)*cos(x9)*x4
    x1_p1_LB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p1_LB_l]
    x1_p1_UB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1_p1_UB_l]

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
    for tup in x1_p1_LB_s
        #First find corresponding indices 
        #Index 1 is x4, its bounds are in x1_p1_sp3
        ind3 = findall(x->x[1] == tup[1], x1_p1_sp3_LB)[1]
        #Index 2 is x8, its bounds are in x1_p1_sp1
        ind1 = findall(x->x[1] == tup[2], x1_p1_sp1_LB)[1]
        #Index 3 is x9, its bounds are in x1_p1_sp2
        ind2 = findall(x->x[1] == tup[3], x1_p1_sp2_LB)[1]
        f1f2f3_UB = tup[end] - x1_p1_sp1_LB[ind1][end]*x1_p1_sp3_LB[ind3][end]*s_x1p1sp2 - x1_p1_sp2_LB[ind2][end]*x1_p1_sp3_LB[ind3][end]*s_x1p1sp1 - x1_p1_sp3_LB[ind3][end]*s_x1p1sp1*s_x1p1sp2 - x1_p1_sp1_LB[ind1][end]*x1_p1_sp2_LB[ind2][end]*s_x1p1sp3 - x1_p1_sp1_LB[ind1][end]*s_x1p1sp2*s_x1p1sp3 - x1_p1_sp2_LB[ind2][end]*s_x1p1sp1*s_x1p1sp3 - s_x1p1sp1*s_x1p1sp2*s_x1p1sp3

        push!(x1_p1_UB, (tup[1:end-1]..., f1f2f3_UB))
    end

    for tup in x1_p1_UB_s
        #Index 1 is x4, its bounds are in x1_p1_sp3
        ind3 = findall(x->x[1] == tup[1], x1_p1_sp3_UB)[1]
        #Index 2 is x8, its bounds are in x1_p1_sp1
        ind1 = findall(x->x[1] == tup[2], x1_p1_sp1_UB)[1]
        #Index 3 is x9, its bounds are in x1_p1_sp2
        ind2 = findall(x->x[1] == tup[3], x1_p1_sp2_UB)[1]
        f1f2f3_LB = tup[end] - x1_p1_sp1_UB[ind1][end]*x1_p1_sp3_UB[ind3][end]*s_x1p1sp2 - x1_p1_sp2_UB[ind2][end]*x1_p1_sp3_UB[ind3][end]*s_x1p1sp1 - x1_p1_sp3_UB[ind3][end]*s_x1p1sp1*s_x1p1sp2 - x1_p1_sp1_UB[ind1][end]*x1_p1_sp2_UB[ind2][end]*s_x1p1sp3 - x1_p1_sp1_UB[ind1][end]*s_x1p1sp2*s_x1p1sp3 - x1_p1_sp2_UB[ind2][end]*s_x1p1sp1*s_x1p1sp3 - s_x1p1sp1*s_x1p1sp2*s_x1p1sp3

        push!(x1_p1_LB, (tup[1:end-1]..., f1f2f3_LB))
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
    #Specify digits for interpolation
    x1_p2_sp1_LB, x1_p2_sp1_UB = interpol(bound_univariate(x1_p2_sp1, lb_x1_p2_sp1, ub_x1_p2_sp1)...,sigFigs)

    #Sub-part 2 = cos(x9)
    x1_p2_sp2 = :(cos(x))
    lb_x1_p2_sp2 = lbs[9]
    ub_x1_p2_sp2 = ubs[9]
    if ub_x1_p2_sp2 - lb_x1_p2_sp2 < 1e-5
        lb_x1_p2_sp2 = lb_x1_p2_sp2 - 1e-5
        ub_x1_p2_sp2 = ub_x1_p2_sp2 + 1e-5
    end
    #Specify digits for interpolation
    x1_p2_sp2_LB, x1_p2_sp2_UB = interpol(bound_univariate(x1_p2_sp2, lb_x1_p2_sp2, ub_x1_p2_sp2)...,sigFigs)

    #Find how much to shift each pair of bounds by for valid log
    s_x1p2sp1 = inpShiftLog(lb_x1_p2_sp1, ub_x1_p2_sp1, bounds=x1_p2_sp1_LB)
    s_x1p2sp2 = inpShiftLog(lb_x1_p2_sp2, ub_x1_p2_sp2, bounds=x1_p2_sp2_LB)

    #Apply log
    x1_p2_sp1_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p2sp1)) for tup in x1_p2_sp1_LB]
    x1_p2_sp1_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p2sp1)) for tup in x1_p2_sp1_UB]

    x1_p2_sp2_LB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p2sp2)) for tup in x1_p2_sp2_LB]
    x1_p2_sp2_UB_l = [(tup[1:end-1]..., log(tup[end] + s_x1p2sp2)) for tup in x1_p2_sp2_UB]

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

    #Account for the shift induced by the log
    #Don't have to round bc we used zeroVal = 0.0
    #This will be done in a loop, but check the logic first

    x1_p2_LB = []
    x1_p2_UB = []
    #Shift down using interval subtraction
    #f1f2 = f_hat - f1b - f2a - ab
    #a is s_x1p2sp1, b is s_x1p2sp2
    #f1 is sin(x8)[x1_p2_sp1], f2 is cos(x9)[x1_p2_sp2]

    for tup in x1_p2_LB_s
        #Index 1 is x8, its bounds are in x1_p2_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p2_sp1_LB)[1]
        #Index 2 is x9, its bounds are in x1_p2_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p2_sp2_LB)[1]
        f1f2_UB = tup[end] - x1_p2_sp1_LB[ind1][end]*s_x1p2sp2 - x1_p2_sp2_LB[ind2][end]*s_x1p2sp1 - s_x1p2sp1*s_x1p2sp2

        push!(x1_p2_UB, (tup[1:end-1]..., f1f2_UB))
    end

    
    for tup in x1_p2_UB_s
        #Index 1 is x8, its bounds are in x1_p2_sp1
        ind1 = findall(x->x[1] == tup[1], x1_p2_sp1_UB)[1]
        #Index 2 is x9, its bounds are in x1_p2_sp2
        ind2 = findall(x->x[1] == tup[2], x1_p2_sp2_UB)[1]
        f1f2_LB = tup[end] - x1_p2_sp1_UB[ind1][end]*s_x1p2sp2 - x1_p2_sp2_UB[ind2][end]*s_x1p2sp1 - s_x1p2sp1*s_x1p2sp2

        push!(x1_p2_LB, (tup[1:end-1]..., f1f2_LB))
    end

    x1_p2_UB