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
    x5LB, x5UB = bound_quadx5(Quad, plotFlag, sanityFlag)
    x6LB, x6UB = bound_quadx6(Quad, plotFlag, sanityFlag)


include("../../overtPoly_helpers.jl")
include("quad_helpers.jl")
##############################################
