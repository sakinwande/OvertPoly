include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
include("attitude_helpers.jl")
using LazySets
using Dates
using Plasmo
control_coef = [[0],[0],[0], [0.25], [0.5],[1]]

controller = "Networks/ARCH-COMP-2023/nnet/controllerAttitude.nnet"
exprList = []
dt = 0.1
domain = domain  = Hyperrectangle(low=[-0.75, 0.85, -0.65, -0.45, -0.55, 0.65], high=[-0.74, 0.86, -0.64, -0.44, -0.54, 0.66])
numSteps = 30

#Define attitude dynamics
function attitude_dynamics(x, u)
    """
    Attitude dynamics 
    """

    dx1 = 0.5*(x[5]*(x[1]^2 + x[2]^2 + x[3]^2 - x[3]) + x[6]*(x[1]^2 + x[2]^2 + x[2] + x[3]^2) + x[4]*(x[1]^2 + x[2]^2 + x[3]^2 + 1))
    dx2 = 0.5*(x[4]*(x[1]^2 + x[2]^2 + x[3]^2 + x[3]) + x[6]*(x[1]^2 - x[1] + x[2]^2 + x[3]^2) + x[5]*(x[1]^2 + x[2]^2 + x[3]^2 + 1))
    dx3 = 0.5*(x[4]*(x[1]^2 + x[2]^2 - x[2] + x[3]^2) + x[5]*(x[1]^2 + x[1] + x[2]^2 + x[3]^2) + x[6]*(x[1]^2 + x[2]^2 + x[3]^2 + 1))
    dx4 = 0.25*(u[1] + x[5]*x[6])
    dx5 = 0.5*(u[2] - 3*x[4]*x[6])
    dx6 = u[3] + 2*x[4]*x[5]

    xNew = x + [dx1, dx2, dx3, dx4, dx5, dx6].*dt
    return xNew
end

function attitude_control(input_set)
    con_inp_set = input_set
    return con_inp_set
end

function bound_attitude(attitude, plotFlag=false, sanityFlag = false)
    x1LB, x1UB = bound_att1(attitude, plotFlag, sanityFlag)
    x2LB, x2UB = bound_att2(attitude, plotFlag, sanityFlag)
    x3LB, x3UB = bound_att3(attitude, plotFlag, sanityFlag)
    x4LB, x4UB = bound_att4(attitude, plotFlag, sanityFlag)
    x5LB, x5UB = bound_att5(attitude, plotFlag, sanityFlag)
    x6LB, x6UB = bound_att6(attitude, plotFlag, sanityFlag)

    return [[x1LB, x1UB], [x2LB, x2UB], [x3LB, x3UB], [x4LB, x4UB], [x5LB, x5UB], [x6LB, x6UB]]
end

function attitude_dyn_con_link!(query, neurons, graph, dynModel, netModel, t_ind=nothing)
    #Define network model inputs 
    @variable(netModel, x1)
    @variable(netModel, x2)
    @variable(netModel, x3)
    @variable(netModel, x4)
    @variable(netModel, x5)
    @variable(netModel, x6)

    #Specify inputs to the network
    @constraint(netModel, neurons[1][1] == x1)
    @constraint(netModel, neurons[1][2] == x2)
    @constraint(netModel, neurons[1][3] == x3)
    @constraint(netModel, neurons[1][4] == x4)
    @constraint(netModel, neurons[1][5] == x5)
    @constraint(netModel, neurons[1][6] == x6)

    #Link network inputs to appropriate dynamics model
    @linkconstraint(graph, netModel[:x1] == dynModel[1][:x][1])
    @linkconstraint(graph, netModel[:x2] == dynModel[2][:x][2])
    @linkconstraint(graph, netModel[:x3] == dynModel[3][:x][3])
    @linkconstraint(graph, netModel[:x4] == dynModel[4][:x][1])
    @linkconstraint(graph, netModel[:x5] == dynModel[5][:x][2])
    @linkconstraint(graph, netModel[:x6] == dynModel[6][:x][3])

    #Define network outputs 
    @variable(netModel, u1)
    @variable(netModel, u2)
    @variable(netModel, u3)

    #Connect network outputs to neurons
    @constraint(netModel, neurons[end][1] == u1)
    @constraint(netModel, neurons[end][2] == u2)
    @constraint(netModel, neurons[end][3] == u3)

    #Connect network outputs to dynamics model
    @linkconstraint(graph, netModel[:u1] == dynModel[4][:u])
    @linkconstraint(graph, netModel[:u2] == dynModel[5][:u])
    @linkconstraint(graph, netModel[:u3] == dynModel[6][:u])

    #Finally, identify persistent input variable for each model
    for (i,sym) in enumerate(query.problem.varList)
        if !isnothing(t_ind)
            sym_t = Meta.parse("$(sym)_$(t_ind)")
        else
            sym_t = sym
        end
        if i <= 3
            pertVar = dynModel[i][:x][i]
            push!(query.var_dict[sym_t], [pertVar])
        else
            pertVar = dynModel[i][:x][i-3]
            push!(query.var_dict[sym_t], [pertVar])
        end
    end
end

Attitude = GraphPolyProblem(
    exprList, 
    nothing,
    control_coef,
    domain,
    [:x1, :x2, :x3, :x4, :x5, :x6],
    nothing,
    attitude_dynamics,
    bound_attitude,
    attitude_control,
    attitude_dyn_con_link!
)

query = GraphPolyQuery(
    Attitude,
    controller,
    Id(),
    "MIP",
    numSteps, 
    dt,
    1,
    nothing,
    nothing,
    2
)

#Test one-step concrete reachability
query1 = deepcopy(query)
@time reachSet, boundSet = concreach!(query1);

#Test multi-step concrete reachability
query2 = deepcopy(query)
query2.ntime = 30
@time reachSets, boundSets = multi_step_concreach(query2);

extrema(reachSets[end])

volume(reachSets[end])

using Plots
plot(reachSets)