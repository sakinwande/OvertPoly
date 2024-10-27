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

function bound_quad(Quad, plotFlag = false, sanityFlag = true)
    x1LB, x1UB = bound_quadx1(Quad, plotFlag, sanityFlag)
    x2LB, x2UB = bound_quadx2(Quad, plotFlag, sanityFlag)
    x3LB, x3UB = bound_quadx3(Quad, plotFlag, sanityFlag)
    x4LB, x4UB = bound_quadx4(Quad, plotFlag, sanityFlag)
    x5LB, x5UB = bound_quadx5(Quad, plotFlag, sanityFlag)
    x6LB, x6UB = bound_quadx6(Quad, plotFlag, sanityFlag)
    x7LB, x7UB = bound_quadx7(Quad, plotFlag, sanityFlag)
    x8LB, x8UB = bound_quadx8(Quad, plotFlag, sanityFlag)
    x9LB, x9UB = bound_quadx9(Quad, plotFlag, sanityFlag)
    x10LB, x10UB = bound_quadx10(Quad, plotFlag, sanityFlag)
    x11LB, x11UB = bound_quadx11(Quad, plotFlag, sanityFlag)
    x12LB, x12UB = bound_quadx12(Quad, plotFlag, sanityFlag)

    return [[x1LB, x1UB], [x2LB, x2UB], [x3LB, x3UB], [x4LB, x4UB], [x5LB, x5UB], [x6LB, x6UB], [x7LB, x7UB], [x8LB, x8UB], [x9LB, x9UB], [x10LB, x10UB], [x11LB, x11UB], [x12LB, x12UB]]
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


###############################
include("../../overtPoly_helpers.jl")
include("quad_helpers.jl")
##############################################
function bound_quadx10(Quad, plotFlag, sanityFlag)
    lbs, ubs = extrema(Quad.domain)

    #Bounding ((Jy - Jz)/Jx)*x₁₁*x₁₂
    #Sub-part 1: (Jy - Jz)/Jx * x₁₁
    x10_p1_sp1 = :($((Jy - Jz)/Jx)*x) 
    lb_x10_p1_sp1 = lbs[11]
    ub_x10_p1_sp1 = ubs[11]

    x10_p1_sp1_LB, x10_p1_sp1_UB = interpol_nd(bound_univariate(x10_p1_sp1, lb_x10_p1_sp1, ub_x10_p1_sp1)...)

    #Sub-part 2: x₁₂
    x10_p1_sp2 = :(1*x)
    lb_x10_p1_sp2 = lbs[12]
    ub_x10_p1_sp2 = ubs[12]

    x10_p1_sp2_LB, x10_p1_sp2_UB = interpol_nd(bound_univariate(x10_p1_sp2, lb_x10_p1_sp2, ub_x10_p1_sp2)...)

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
        validBounds(:($((Jy - Jz)/Jx)*x11*x12), [:x11, :x12], x10_p1_LB, x10_p1_UB)
    end

    return x10_p1_LB, x10_p1_UB
end