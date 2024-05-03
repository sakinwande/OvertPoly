include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../reachability.jl")
using LazySets
using Dates

#Define problem parameters
pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.0
controller_type = "small" # pass from command line, e.g. "small"
controller = "Networks/nnet/single_pendulum_$(controller_type)_controller.nnet"
expr = [:($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)]
control_coef = 1/(pend_mass*pend_len^2)


domain = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
domain = Hyperrectangle(low=[0., -0.1], high=[1, 0.1])
numSteps = 20
dt = 0.05

function single_pend_update_rule(input_vars,
    overt_output_vars)
    ddth = overt_output_vars[1][1]
    integration_map = Dict(input_vars[1] => input_vars[2], input_vars[2] => ddth)
    return integration_map
end

function single_pend_dynamics(x, u, dt)
    """
    Dynamics of the single pendulum for a single time step
    """
    dx1 = x[2]
    dx2 = (grav_const/pend_len) * sin(x[1]) + (1/(pend_mass*pend_len^2)) * u - (friction/(pend_mass*pend_len^2)) * x[2]
    
    xNew = [x[1] + dx1*dt, x[2] + dx2*dt]
    return xNew
end

function bound_pend(SinglePendulum; plotFlag=false)
    #Define the true dynamics
    # single_pend_θ_doubledot = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)

    #get input bounds 
    lbs, ubs = extrema(SinglePendulum.domain)

    #Bound f(x1)
    lb1 = lbs[1]
    ub1 = ubs[1]
    bF1sub1 = :($(grav_const/pend_len) * sin(x1))
    bF1s1LB, bF1s1UB = bound_univariate(bF1sub1, lb1, ub1, plotflag = true) 

    #TEST: Compute length of pre interpolation bounds 
    size(bF1s1LB)[1]
    size(bF1s1UB)[1]

    #Bound f(x2)
    lb2 = lbs[2]
    ub2 = ubs[2]
    bF1sub2 = :($((friction)/((pend_mass)*(pend_len)^2)) * x2)
    bF1s2LB, bF1s2UB = bound_univariate(bF1sub2, lb2, ub2, plotflag = true)

    #TEST: Compute length of pre interpolation bounds
    size(bF1s2LB)[1]
    size(bF1s2UB)[1]

    #For future use, interpolate to ensure UB and LB for each is over the same set of points 
    bF1s1LB, bF1s1UB = interpol(bF1s1LB, bF1s1UB)
    bF1s2LB, bF1s2UB = interpol(bF1s2LB, bF1s2UB)

    #TEST: Compute length of post interpolation bounds
    size(bF1s1LB)[1]
    size(bF1s1UB)[1]

    #Add a dimension to prepare for Minkowski sum
    bF1s1LB_l = addDim(bF1s1LB, 2)
    bF1s1UB_l = addDim(bF1s1UB, 2)

    bF1s2LB_l = addDim(bF1s2LB, 1)
    bF1s2UB_l = addDim(bF1s2UB, 1)

    #Combine to get f(x1) + f(x2)
    bF1LB = MinkSum(bF1s1LB_l, bF1s2LB_l)
    bF1UB = MinkSum(bF1s1UB_l, bF1s2UB_l)


    bF1LB[5][3] > bF1UB[5][3]

    bF1LB[5][3] - bF1UB[5][3]
    #Try to plot
    if plotFlag
        xS = Any[tup[1] for tup in bF1s1LB]
        yS = Any[tup[1] for tup in bF1s2LB]
        surfDim = (size(yS)[1],size(xS)[1])
        exp2Plot = :($(grav_const/pend_len) * sin(x1) - $(friction/(pend_mass*pend_len^2)) * x2)
        plotSurf(exp2Plot, bF1LB, bF1UB, surfDim, xS, yS, true)
    end
    bounds = [[bF1LB, bF1UB]]
    return bounds
end

SinglePendulum = OvertPProblem(
    expr, # dynamics
    nothing, #decomposed form of the dynamics. Done manually
    control_coef, # control coefficient
    domain, # domain
    [:dθ], #List of variables that have OVERT bounds
	nothing, #undefined bounds to start
	single_pend_update_rule,
    single_pend_dynamics,
    bound_pend
)

query = OvertPQuery(
	SinglePendulum,    # problem
	controller,        # network file
	Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",             # query solver, "MIP" or "ReluPlex"
	numSteps,                # ntime
	dt,               # dt
	2,                # N_overt
    nothing,         # var_dict
    nothing,         # mod_dict
    2                # case
)

#Use concrete reachability to trace out the trajectory
query1 = deepcopy(query)
@time reachSets, boundSets = multi_step_concreach(query1)

plot(reachSets, title="Single Pendulum Concrete Reachability")

#Use concrete sets to compute symbolic reach set at time step 10
symQuery1 = deepcopy(query)
symQuery1.problem.bounds = boundSets
symQuery1.ntime = 

#############Testing Single Step Hybrid Symbolic Reachability############
@time reach_set = symReach(symQuery1, reachSets)
plot(reachSets[11], title="Comparing Concrete and Hyb Reach_$(symQuery1.ntime)", label="Concrete Reach Set")
plot!(reach_set, label="Hyb Reach Set")

#############Testing Multi Step Hybrid Symbolic Reachability############
symQuery2 = deepcopy(query)
symQuery2.problem.bounds = boundSets
symQuery2.ntime = 15

concEvery = 10
@time totalReachSets, totalBoundSets  = multi_step_hybreach(concEvery,symQuery2)
length(totalReachSets)
plot(totalReachSets, title="Single Pendulum Hybrid Reachability", fillcolor=:blue)
#############Trying out multi step straight shot reachability############
symQuery3 = deepcopy(query)
symQuery3.ntime = 10
@time reachSetsSS = multi_shot_reach(symQuery3)

plot!(reachSetsSS[end], label="Straight Shot Reach Set")
plot!(reachSets, title="Single Pendulum Straight Shot Reachability", fillcolor=:red)

############Testing Area############

boundsC = [
    Hyperrectangle(low=[1., 0.], high=[1.20000005, 0.20000000]), 
    Hyperrectangle(
        low=[0.00000000, 0.00000000, 0.06779400, 0.00000000, 0.00000000, 0.37062001, 0.72581995, 0.03107001, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.23027001, 0.00000000, 0.00000000, 0.00214000, 0.00000000, 0.66058999, 0.00000000, 0.00000000, 0.00000000, 0.40715000, 0.00000000, 0.00000000], 
        high=[0.00000000, 0.00000000, 0.16838601, 0.00000000, 0.00000000, 0.53822005, 1.00910604, 0.09147601, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.32687205, 0.00000000, 0.00000000, 0.08687800, 0.00000000, 0.82984000, 0.00000000, 0.00000000, 0.00000000, 0.49659804, 0.00000000, 0.00000000]), 
    Hyperrectangle(
        low=[0.33707854, 0.03688293, 0.00000000, 0.00000000, 0.65297151, 0.13813956, 0.00000000, 0.11324575, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.07053523, 0.39192262, 0.09188993, 0.01795765, 0.00000000, 0.00000000, 0.64817131, 0.00000000, 0.02351930, 0.00000000, 0.09244308, 0.51171654, 0.00000000],
        high=[0.46692976, 0.08296970, 0.04684000, 0.00000000, 0.84860194, 0.18440942, 0.00000000, 0.20420133, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.08734600, 0.55084288, 0.14003707, 0.05558294, 0.00000000, 0.01129147, 0.89140284, 0.00000000, 0.06163520, 0.00000000, 0.13134380, 0.65339273, 0.00000000]),
    Hyperrectangle(
        low=[-0.79562807],
        high=[-0.53514576]
    )
]
bboundsC = Vector{Hyperrectangle}(undef, 0)
#Enter the CROWN bounds in the appropriate format
for i = 1:4
    push!(bboundsC, boundsC[i])
end

queryCROWN = OvertPQuery(
	SinglePendulum,    # problem
	controller,        # network file
	Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",             # query solver, "MIP" or "ReluPlex"
	5,                # ntime
	0.1,               # dt
	2,                # N_overt
    nothing         # var_dict
)


######Simulating concreach
#OG method
query.problem.bounds = bound_pend(query.problem)
query.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
mipModel = encode_dynamics(query)
query.var_dict[:u] 
encode_control!(query, mipModel)
mipModel

#Crown method
queryCROWN.problem.bounds = bound_pend(queryCROWN.problem)
queryCROWN.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
mipModelCrown = encode_dynamics(queryCROWN)
queryCROWN.var_dict[:u]

input_set = queryCROWN.problem.domain   ###For controller MIP encoding, need model, network address, input set, input variable names, output variable names
network_file = queryCROWN.network_file

#Get dictionary of MIP variables 
input_vars = queryCROWN.var_dict[:x]
control_vars = queryCROWN.var_dict[:u][1]
output_vars = queryCROWN.var_dict[:y][1]

function add_ccontroller_constraints!(model, network_nnet_address, input_set, input_vars, output_vars; last_layer_activation=Id())
    """
    Encode controller as MIP. Directly taken from OVERTVerify
    """
    #Read network file 
    network = read_nnet(network_nnet_address, last_layer_activation=last_layer_activation)
    #Initialize neurons (adds variables)
    neurons = init_neurons(model, network)
    #Initialize deltas (adds binary variables)
    deltas = init_deltas(model, network)
    #Use CROWN
    boundsC = [
    Hyperrectangle(low=[1., 0.], high=[1.20000005, 0.20000000]), 
    Hyperrectangle(
        low=[0.00000000, 0.00000000, 0.06779400, 0.00000000, 0.00000000, 0.37062001, 0.72581995, 0.03107001, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.23027001, 0.00000000, 0.00000000, 0.00214000, 0.00000000, 0.66058999, 0.00000000, 0.00000000, 0.00000000, 0.40715000, 0.00000000, 0.00000000], 
        high=[0.00000000, 0.00000000, 0.16838601, 0.00000000, 0.00000000, 0.53822005, 1.00910604, 0.09147601, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.32687205, 0.00000000, 0.00000000, 0.08687800, 0.00000000, 0.82984000, 0.00000000, 0.00000000, 0.00000000, 0.49659804, 0.00000000, 0.00000000]), 
    Hyperrectangle(
        low=[0.33707854, 0.03688293, 0.00000000, 0.00000000, 0.65297151, 0.13813956, 0.00000000, 0.11324575, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.07053523, 0.39192262, 0.09188993, 0.01795765, 0.00000000, 0.00000000, 0.64817131, 0.00000000, 0.02351930, 0.00000000, 0.09244308, 0.51171654, 0.00000000],
        high=[0.46692976, 0.08296970, 0.04684000, 0.00000000, 0.84860194, 0.18440942, 0.00000000, 0.20420133, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.08734600, 0.55084288, 0.14003707, 0.05558294, 0.00000000, 0.01129147, 0.89140284, 0.00000000, 0.06163520, 0.00000000, 0.13134380, 0.65339273, 0.00000000]),
    Hyperrectangle(
        low=[-0.79562807],
        high=[-0.53514576]
    )
    ]
    bounds = Vector{Hyperrectangle}(undef, 0)

    for i = 1:4
        push!(bounds, boundsC[i])
    end
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())
    #Relate the NN variables to the dynamics variables
    @constraint(model, input_vars .== neurons[1])  # set inputvars
    @constraint(model, output_vars .== neurons[end])  # set outputvars
    return bounds[end]

end

controller_bound = add_ccontroller_constraints!(mipModelCrown, network_file, input_set, input_vars, control_vars)

controller_bound

mipModel
mipModelCrown

@time reach_solve(mipModel, query)
@time reach_solve(mipModelCrown, queryCROWN)