include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo
control_coef = [[0],[0],[0],[1]]
controller = "Networks/ARCH-COMP-2023/nnet/car_big_controller.nnet"
exprList = [:(x4*cos(x3)), :(x4*sin(x3)), :(0*x3), :(0*x4)]

##Define Unicycle Dynamics#####
function unicycle_dynamics(x, u)
    """
    Dynamics of the unicycle benchmark. NGL I don't know why I use this
    """
    dx1 = x[4]*cos(x[3])
    dx2 = x[4]*sin(x[3])
    dx3 = u[2]
    dx4 = u[1]

    xNew = x + [dx1, dx2, dx3, dx4].*dt

    return xNew
end

function unicycle_control(input_set)
    con_inp_set = input_set 
    return con_inp_set
end

domain = Hyperrectangle(low=[9.5,-4.5,2.1,1.5], high = [9.55,-4.45,2.11,1.51])

dt = 0.2
numSteps = 50

lbs, ubs = extrema(domain)
plotFlag = false
###Define Bound Unicycle########
# function bound_unicycle(Unicycle; plotFlag=false)
    lbs, ubs = extrema(Unicycle.domain)

end