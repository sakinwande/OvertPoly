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

println("Running quad benchmark")
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
controller = "../../../Networks/ARCH-COMP-2023/nnet/controllerQuad.nnet"
#TEST: Controller 
#controller = "Networks/ARCH-COMP-2023/nnet/controllerACC.nnet"

dt = 0.1
ϵ = 1e-5
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
function quad_dyn_con_link!(query, neurons, graph, dynModel, netModel, t_ind=nothing)
    #Define variables that are inputs to the network model 
    @variable(netModel, x1)
    @variable(netModel, x2)
    @variable(netModel, x3)
    @variable(netModel, x4)
    @variable(netModel, x5)
    @variable(netModel, x6)
    @variable(netModel, x7)
    @variable(netModel, x8)
    @variable(netModel, x9)
    @variable(netModel, x10)
    @variable(netModel, x11)
    @variable(netModel, x12)

    #Specify inputs to the network
    @constraint(netModel, neurons[1][1] == x1)
    @constraint(netModel, neurons[1][2] == x2)
    @constraint(netModel, neurons[1][3] == x3)
    @constraint(netModel, neurons[1][4] == x4)
    @constraint(netModel, neurons[1][5] == x5)
    @constraint(netModel, neurons[1][6] == x6)
    @constraint(netModel, neurons[1][7] == x7)
    @constraint(netModel, neurons[1][8] == x8)
    @constraint(netModel, neurons[1][9] == x9)
    @constraint(netModel, neurons[1][10] == x10)
    @constraint(netModel, neurons[1][11] == x11)
    @constraint(netModel, neurons[1][12] == x12)

    #Link network inputs to appropriate dynamics models 
    @linkconstraint(graph, netModel[:x1] == dynModel[1][:x][1])
    @linkconstraint(graph, netModel[:x2] == dynModel[2][:x][1])
    @linkconstraint(graph, netModel[:x3] == dynModel[3][:x][1])
    @linkconstraint(graph, netModel[:x4] == dynModel[4][:x][1])
    @linkconstraint(graph, netModel[:x5] == dynModel[5][:x][2])
    @linkconstraint(graph, netModel[:x6] == dynModel[6][:x][3])
    @linkconstraint(graph, netModel[:x7] == dynModel[7][:x][1])
    @linkconstraint(graph, netModel[:x8] == dynModel[8][:x][2])
    @linkconstraint(graph, netModel[:x9] == dynModel[9][:x][3])
    @linkconstraint(graph, netModel[:x10] == dynModel[10][:x][1])
    @linkconstraint(graph, netModel[:x11] == dynModel[11][:x][2])
    @linkconstraint(graph, netModel[:x12] == dynModel[12][:x][3])

    #Define network outputs 
    @variable(netModel, u1)
    @variable(netModel, u2)
    @variable(netModel, u3)

    #Connect network outputs to neurons 
    @constraint(netModel, neurons[end][1] == u1)
    @constraint(netModel, neurons[end][2] == u2)
    @constraint(netModel, neurons[end][3] == u3)

    #Connect network outputs to dynamics model
    @linkconstraint(graph, netModel[:u1] == dynModel[6][:u])
    @linkconstraint(graph, netModel[:u2] == dynModel[10][:u])
    @linkconstraint(graph, netModel[:u3] == dynModel[11][:u])

    #Finally, identify pertient variable for each model
    for (i, sym) in enumerate(query.problem.varList) 
        if !isnothing(t_ind)
            sym_t = Meta.parse("$(sym)_$(t_ind)")
        else
            sym_t = sym
        end
        if i > 3 && i % 3 == 2
            #Catches 5, 8, 11
            pertVar = dynModel[i][:x][2]
            push!(query.var_dict[sym_t], [pertVar])
        elseif i > 3 && i % 3 == 0
            #Catches 6, 9, 12
            pertVar = dynModel[i][:x][3]
            push!(query.var_dict[sym_t], [pertVar])
        else
            pertVar = dynModel[i][:x][1]
            push!(query.var_dict[sym_t], [pertVar])
        end

    end
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

# #############################
# query.var_dict = Dict{Symbol,Any}()
# query.mod_dict = Dict{Symbol,Any}()
# graph = OptiGraph()
# query.mod_dict[:graph] = graph


# #############################################
# encode_control!(query)
query1 = deepcopy(query)
tstart = Dates.now()
reachSets, boundSets = multi_step_concreach(query1)
tend = Dates.now()
println("##########################################################################################")
println("Time taken to compute concrete reach: ", tend-tstart)
println("########################################################################################")
