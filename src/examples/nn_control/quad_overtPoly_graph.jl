include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
include("quad_helpers.jl")
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

#TODO: Implement this function
function quad_dyn_con_link!(query)
    boo = 1
end

function bound_quad()
end

Quad = GraphPolyProblem(
    exprList,
    nothing, 
    control_coef,
    domain,
    [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :x10, :x11, :x12],
    nothing, 
    quad_dynamics,
    bound_quad,
    quad_control,
    quad_dyn_con_link!
)

query = GraphPolyQuery(
    Quad, 
    controller,
    Id(),
    "MIP",
    numsteps,
    dt, 
    2,
    nothing,
    nothing,
    2
)

###############
#Bounding the quadcopter dynamics
sanityFlag = true
plotFlag = false
# function bound_quad(Quad, plotFlag = false, sanityFlag = true)
    x1LB, x1UB = bound_quadx1(Quad, plotFlag, sanityFlag)
    x2LB, x2UB = bound_quadx2(Quad, plotFlag, sanityFlag)
    x3LB, x3UB = bound_quadx3(Quad, plotFlag, sanityFlag)
    x4LB, x4UB = bound_quadx4(Quad, plotFlag, sanityFlag)
    


include("../../overtPoly_helpers.jl")
include("quad_helpers.jl")
##############################################
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