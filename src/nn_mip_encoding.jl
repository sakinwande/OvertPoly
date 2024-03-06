using JuMP
using LazySets
using Parameters
include("nv/maxSens.jl")
include("nv/activation.jl")
include("nv/network.jl")
include("nv/constraints.jl")
include("nv/util.jl")

function add_controller_constraints!(model, network_nnet_address, input_set, input_vars, output_vars; last_layer_activation=Id())
    """
    Encode controller as MIP. Directly taken from OVERTVerify
    """
    #Read network file 
    network = read_nnet(network_nnet_address, last_layer_activation=last_layer_activation)
    #Initialize neurons (adds variables)
    neurons = init_neurons(model, network)
    #Initialize deltas (adds binary variables)
    deltas = init_deltas(model, network)
    #Use Taylor Johnson paper (https://arxiv.org/abs/1708.03322) to get bounds  
    bounds = get_bounds(network, input_set)
    #Add NN MIP model to the given model
    #This is defined in the constraints.jl file. Appears to be the Tjeng paper encoding
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())
    #Relate the NN variables to the dynamics variables
    @constraint(model, input_vars .== neurons[1])  # set inputvars
    @constraint(model, output_vars .== neurons[end])  # set outputvars
    return bounds[end]
end


function init_variables(model::Model, layers::Vector{Layer}; binary = false, include_input = false)
    # TODO: only neurons get offset array
    vars = Vector{Vector{VariableRef}}(undef, length(layers))
    all_layers_n = n_nodes.(layers)

    if include_input
        # input to the first layer also gets variables
        # essentially an input constraint
        input_layer_n = size(first(layers).weights, 2)
        prepend!(all_layers_n, input_layer_n)
        push!(vars, Vector{VariableRef}())        # expand vars by one to account
    end

    for (i, n) in enumerate(all_layers_n)
        vars[i] = @variable(model, [1:n], binary = binary, base_name = "z$i")
    end
    return vars
end

init_neurons(model::Model, layers::Vector{Layer})     = init_variables(model, layers, include_input = true)
init_deltas(model::Model, layers::Vector{Layer})      = init_variables(model, layers, binary = true)
init_neurons(m,     network::Network) = init_neurons(m, network.layers)
init_deltas(m,      network::Network) = init_deltas(m,  network.layers)




#######################Testing Area#####################################
#NOTE: Delete or comment out after use 
# using LazySets
# controller_type = "small" # pass from command line, e.g. "small"
# controller = "Networks/nnet/single_pendulum_$(controller_type)_controller.nnet"

# network = read_nnet(controller, last_layer_activation=Id())
# input_set = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2]) # domain

# #This gives element-wise bounds 
# bounds = get_bounds(network, input_set)

# boundsC = [
#     Hyperrectangle(low=[1., 0.], high=[1.20000005, 0.20000000]), 
#     Hyperrectangle(
#         low=[0.00000000, 0.00000000, 0.06779400, 0.00000000, 0.00000000, 0.37062001, 0.72581995, 0.03107001, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.23027001, 0.00000000, 0.00000000, 0.00214000, 0.00000000, 0.66058999, 0.00000000, 0.00000000, 0.00000000, 0.40715000, 0.00000000, 0.00000000], 
#         high=[0.00000000, 0.00000000, 0.16838601, 0.00000000, 0.00000000, 0.53822005, 1.00910604, 0.09147601, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.32687205, 0.00000000, 0.00000000, 0.08687800, 0.00000000, 0.82984000, 0.00000000, 0.00000000, 0.00000000, 0.49659804, 0.00000000, 0.00000000]), 
#     Hyperrectangle(
#         low=[0.33707854, 0.03688293, 0.00000000, 0.00000000, 0.65297151, 0.13813956, 0.00000000, 0.11324575, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.07053523, 0.39192262, 0.09188993, 0.01795765, 0.00000000, 0.00000000, 0.64817131, 0.00000000, 0.02351930, 0.00000000, 0.09244308, 0.51171654, 0.00000000],
#         high=[0.46692976, 0.08296970, 0.04684000, 0.00000000, 0.84860194, 0.18440942, 0.00000000, 0.20420133, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.08734600, 0.55084288, 0.14003707, 0.05558294, 0.00000000, 0.01129147, 0.89140284, 0.00000000, 0.06163520, 0.00000000, 0.13134380, 0.65339273, 0.00000000]),
#     Hyperrectangle(
#         low=[-0.79562807],
#         high=[-0.53514576]
#     )
# ]
# bboundsC = Vector{Hyperrectangle}(undef, 0)
# push!(bboundsC, boundsC[4])



