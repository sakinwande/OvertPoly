include("overtPoly_helpers.jl")
include("nn_mip_encoding.jl")
include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")
include("problems.jl")
include("reachability.jl")
using LazySets
using Dates

#Define problem parameters
pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.0
controller_type = "small" # pass from command line, e.g. "small"
controller = "Networks/nnet/single_pendulum_$(controller_type)_controller.nnet"
expr = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)
firstDec = parse_and_reduce(expr)
secDec = parse_and_reduce(firstDec[2])
control_coef = 8.0
firstDec[2] = secDec
dec_expr = firstDec


function single_pend_update_rule(input_vars,
    control_vars,
    overt_output_vars)
    ddth = overt_output_vars[1]
    integration_map = Dict(input_vars[1] => input_vars[2], input_vars[2] => ddth)
    return integration_map
end

function bound_pend(SinglePendulum)
    #Define the true dynamics
    # single_pend_θ_doubledot = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)
    
    #get decomposed dynamics 
    baseParsed = SinglePendulum.dec_expr

    #get input bounds 
    lbs, ubs = extrema(SinglePendulum.domain)

    #Bound f(x1)
    lb1 = lbs[1]
    ub1 = ubs[1]
    v2Func = baseParsed[2][2]
    v2f = Symbolics.build_function(v2Func, find_variables(v2Func)..., expression=Val{false})
    v2UB, v2LB = bound_univariate(v2Func, lb1, ub1, plotflag = false) 

    #Bound f(x2)
    lb2 = lbs[2]
    ub2 = ubs[2]
    v22Func = baseParsed[3]
    v22f = Symbolics.build_function(v22Func, find_variables(v22Func)..., expression=Val{false})
    v22UB = [(lb2, v22f(lb2)), (ub2, v22f(ub2))]
    v22LB = [(lb2, v22f(lb2)), (ub2, v22f(ub2))]

    #For future use, interpolate to ensure UB and LB for each is over the same set of points 
    nv2LB, nv2UB = interpol(v2LB, v2UB)
    nv22LB, nv22UB = interpol(v22LB, v22UB)
    # nv3LB, nv3UB = interpol(v3LB, v3UB)

    #Log transformation not required because f(x,u) = f(x) + f(u) already

    xS = Any[tup[1] for tup in nv2LB]
    yS = Any[tup[1] for tup in nv22LB]
    # uS = Any[tup[1] for tup in nv3LB]

    #Add y axis to f(x) overapprox
    lv2LBl = addDim(nv2LB, 2)
    lv2UBl = addDim(nv2UB, 2)

    #Add x axis to f(y) overapprox
    lv22LBl = addDim(nv22LB, 1)
    lv22UBl = addDim(nv22UB, 1)

    #Obtain f(x,y) overapprox
    lv2LB = MinkSum(lv2LBl, lv22LBl)
    lv2UB = MinkSum(lv2UBl, lv22UBl)

    # #Try to plot
    # xS
    # yS
    # surfDim = (size(yS)[1],size(xS)[1])
    # saveFlag = true
    # exp2Plot = :($(grav_const/pend_len) * sin(x1) - $(friction/(pend_mass*pend_len^2)) * x2)
    # plotSurf(exp2Plot, lv2LB, lv2UB, surfDim, xS, yS, true)
    bounds = [lv2LB, lv2UB]
    return bounds
end


SinglePendulum = OvertPProblem(
    expr, # dynamics
    dec_expr, #decomposed form of the dynamics
    control_coef, # control coefficient
    Hyperrectangle(low=[1., 0.], high=[1.2, 0.2]), # domain
	nothing, #undefined bounds to start
	single_pend_update_rule
)

query = OvertPQuery(
	SinglePendulum,    # problem
	controller,        # network file
	Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
	"MIP",             # query solver, "MIP" or "ReluPlex"
	25,                # ntime
	0.1,               # dt
	2,                # N_overt
    nothing         # var_dict
)

# #Use concrete reachability to trace out the trajectory
# reachSets, boundSets = multi_step_concreach(query)

# #Define new problem with symbolic dynamics
# symPendulum = OvertPProblem(
#     expr, # dynamics
#     dec_expr, #decomposed form of the dynamics
#     control_coef, # control coefficient
#     Hyperrectangle(low=[1., 0.], high=[1.2, 0.2]), # domain
# 	boundSets, #undefined bounds to start
# 	single_pend_update_rule
# )

# #Define new symbolic query
# symQuery = OvertPQuery(
#     symPendulum,    # problem
#     controller,        # network file
#     Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
#     "MIP",             # query solver, "MIP" or "ReluPlex"
#     25,                # ntime
#     0.1,               # dt
#     2,                # N_overt
#     nothing         # var_dict
# )

# ###############Individual Sym MIP Encoding################
# function ccSymEncoding(xS, yLB, yUB, Tri, symQuery, ind, model)
#     """
#     Method to encode a piecewise affine function as a mixed integer program following the convex combination method as defined in Gessler et. al. 2012
#         (https://www.dl.behinehyab.com/Ebooks/IP/IP011_655874_www.behinehyab.com.pdf#page=308)

#     args:
#         problem: OvertPProblem that encodes the dynamics of the system as well as some useful system info 
#         xS: List of vertices of the triangulation
#         yLB: List of lower bounds of the function at the vertices of the triangulation
#         yUB: List of upper bounds of the function at the vertices of the triangulation
#         Tri: List of simplices of the triangulation
#     """

#     #Following the convention from Gessler et. al. 
#     m = size(xS, 1) #Number of vertices
#     d = size(xS[1], 1) #Dimension of the space
#     n = size(Tri, 1) #Number of simplices

#     #Define indexed symbols for the convex coefficients and binary variables
#     lamb_var = Meta.parse("λ_$(ind)")
#     bin_var = Meta.parse("b_$(ind)")

#     #Define indexed convex coefficients as a MIP variable 
#     #NOTE: This is an anonymous variable. Won't appear in named model variables
#     λ_var = @variable(model, [1:m], base_name = "$lamb_var")
#     symQuery.var_dict[lamb_var] = λ_var

#     #Define indexed binary variables indicating with simplex is active. Use b to avoid conflict with network binary variables 
#     b_var = @variable(model, [1:n], Bin, base_name = "$bin_var")
#     symQuery.var_dict[bin_var] = b_var


#     #Begin constraining our auxilliary variables
#     #Convex combiation constraints (Gessler et. al. eq. 3.2)
#     symQuery.var_dict
#     @constraint(model, λ_var .>= 0)
#     @constraint(model, sum(λ_var) == 1)

#     #This is equation 3.4 from Gessler et. al.
#     #Here, we iterate through all vertices. Then, we constrain the convex coefficient of each vertex to be leq the sum of the binary variables corresponding to the simplices containing that vertex

#     #NOTE This relates a convex coefficient to its neighbors 
#     for j in 1:m
#         #Below we find all simplices where index j is present 
#         @constraint(model, λ_var[j] <= sum(b_var[i] for i in findall(x -> j in x, Tri)))
#     end

#     #Next, enforce that at most one simplex can be active at a time (Gessler et. al. eq. 3.5)
#     @constraint(model, sum(b_var) <= 1)


#     #Create indices for state variables and output variables
#     x_ind = Meta.parse("x_$(ind)")
#     y_ind = Meta.parse("y_$(ind)")
#     yl_ind = Meta.parse("yl_$(ind)")
#     yu_ind = Meta.parse("yu_$(ind)")
#     u_ind = Meta.parse("u_$(ind)")

#     #Now, define function variables as MIP variables
#     x_var = @variable(model, [1:d], base_name = "$x_ind")
#     symQuery.var_dict[x_ind] = x_var
#     y_var = @variable(model, [1], base_name = "$y_ind")
#     symQuery.var_dict[y_ind] = y_var
#     yₗ = @variable(model, [1], base_name = "$yl_ind")
#     symQuery.var_dict[yl_ind] = yₗ
#     yᵤ = @variable(model, [1], base_name = "$yu_ind")
#     symQuery.var_dict[yu_ind] = yᵤ
#     u = @variable(model, [1], base_name = "$u_ind")
#     symQuery.var_dict[u_ind] = u

#     #Define the generic vertex as a convex combination of its neighbors 
#     #This exploits casting. NOTE: Could be dangerous 
#     @constraint(model, x_var .== sum(λ_var[i]*[xS[i]...] for i in 1:m))

#     #Define the generic function value in terms of the convex combination of its upper and lower bounds
#     #NOTE: Control is changed here. Very bad 
#     @constraint(model, yₗ[1] == sum(λ_var[i]*yLB[i] for i in 1:m))
#     @constraint(model, yᵤ[1] == sum(λ_var[i]*yUB[i] for i in 1:m))
#     @constraint(model, yₗ[1] + symQuery.problem.control_coef*u[1] <= y_var[1])
#     @constraint(model, y_var[1] <= yᵤ[1] + symQuery.problem.control_coef*u[1])


#     #We will also need to define additional constraints on x and y, but those will be added later
#     return model 

# end

# ####### Encode Symbolic Dynamics ########
# function encode_sym_dynamics(symQuery::OvertPQuery)
#     """
#     Function to encode the dynamics over a trajectory of overapproximations. 
#     TODO: Revisit this. Can be done more efficiently by exploiting the overlap between the domains of the overapproximations. This should result in significantly fewer vertices in the final MIP encoding.
#     """
#     #Define symbolic MIP model
#     optimizer = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
#     #Define model
#     symMip = Model(optimizer)

#     #Define dictionary to store MIP variables
#     symQuery.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()

#     for ind in 1:size(symQuery.problem.bounds)[1]
#         boundVec = symQuery.problem.bounds[ind]
#         LBs = boundVec[1]
#         UBs = boundVec[2]

#         Tri = OA2PWA(LBs)

#         xS = [(tup[1:end-1]) for tup in LBs]
#         yUB = [tup[end] for tup in UBs]
#         yLB = [tup[end] for tup in LBs]

#         symMip = ccSymEncoding(xS, yLB, yUB, Tri, symQuery, ind, symMip)
#     end

#     symMip
#     return symMip
# end

# #######Encode Symbolic Controller########
# function encode_sym_control(sym_mip, symQuery, reachSets)
#     input_set = symQuery.problem.domain  
#     network_file = symQuery.network_file
#     ntime = symQuery.ntime
#     #Get dictionary of MIP variables 
#     varDict = symQuery.var_dict

#     #####Enter loop for time steps#################
#     for i = 1:symQuery.ntime
#         input_set = reachSets[i]
#         input_vars_curr = Meta.parse("x_$i") 
#         control_vars_curr = Meta.parse("u_$i") 

#         x_curr = varDict[input_vars_curr]
#         u_curr = varDict[control_vars_curr][1]

#         controller_bound = add_controller_constraints!(sym_mip, network_file, input_set, x_curr, u_curr)
#     end
#     return sym_mip
# end

# ##########Symbolically link time steps########
# function encode_time(sym_mip, symQuery)
#     """
#     Method to encode the time evolution of the system as a MIP. Links distinct overappoximation objects across time steps to complete the symbolic encoding. 
#     """
#     ##########First loop#############
#     for i = 1:symQuery.ntime-1
#         y_now = symQuery.var_dict[Meta.parse("y_$i")]
#         u_now = symQuery.var_dict[Meta.parse("u_$i")]
#         x_now = symQuery.var_dict[Meta.parse("x_$i")]
#         x_next = symQuery.var_dict[Meta.parse("x_$(i+1)")]

#         integration_map = single_pend_update_rule(x_now, u_now, y_now)

#         #######Second loop#############
#         for j = 1:length(x_now)
#             v = x_now[j]
#             dv = integration_map[v]
#             next_v = x_next[j]

#             @constraint(sym_mip, next_v == v + symQuery.dt*dv)
#         end
#     end
#     return sym_mip
# end

# sym_mip = encode_sym_dynamics(symQuery)
# sym_mip = encode_sym_control(sym_mip, symQuery, reachSets)
# sym_mip = encode_time(sym_mip, symQuery)

# #Solve the symbolic MIP
# reach_solve(sym_mip, symQuery, symQuery.ntime)


############Testing Area############
# network = read_nnet(controller, last_layer_activation=Id())
# input_set = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2]) # domain

# #This gives element-wise bounds 
# bounds = get_bounds(network, input_set)

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
	25,                # ntime
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