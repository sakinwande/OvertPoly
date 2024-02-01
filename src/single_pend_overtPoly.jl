include("overtPoly_helpers.jl")
include("nn_mip_encoding.jl")
include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")
using LazySets

pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.0


single_pend_θ_doubledot = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)

baseParsed = parse_and_reduce(single_pend_θ_doubledot)
bbaseParsed = parse_and_reduce(baseParsed[2])

#Bound f(x1)
lb1 = 1.0
ub1 = 1.2
v2Func = bbaseParsed[2]
v2f = Symbolics.build_function(v2Func, find_variables(v2Func)..., expression=Val{false})
v2UB, v2LB = bound_univariate(v2Func, lb1, ub1, plotflag = true) 

#Bound f(x2)
lb2 = 0.0
ub2 = 0.2
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

################Convert to MIP######################
LB, UB = lv2LB, lv2UB



Tri = OA2PWA(LB)
#These are the vertices of the triangulation
xS = [(tup[1:end-1]) for tup in LB]
yUB = [tup[end] for tup in UB]
yLB = [tup[end] for tup in LB]

mipModel = ccEncoding(xS, yLB, yUB, Tri)

###For controller MIP encoding, need model, network address, input set, input variable names, output variable names
network_file = "nnet_files/single_pendulum_small_controller.nnet"
input_set = Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
#Get dictionary of MIP variables 
varDict = JuMP.object_dictionary(mipModel)
input_vars = varDict[:x]
control_vars = varDict[:u]
output_vars = varDict[:y]

controller_bound = add_controller_constraints!(mipModel, network_file, input_set, input_vars, control_vars)

mipModel


function single_pend_update_rule(input_vars,
    control_vars,
    overt_output_vars)
ddth = overt_output_vars[1]
integration_map = Dict(input_vars[1] => input_vars[2], input_vars[2] => ddth)
return integration_map
end

#################Local Reachability Loop#####################
lows = Array{Float64}(undef, 0)
highs = Array{Float64}(undef, 0)

control_vars = output_vars
input_vars_last = keys(input_vars)
dt = 0.1
integration_map = single_pend_update_rule(keys(input_vars), control_vars, [output_vars])

input_vars_last
timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
#####Enter loop here#################
#Loop over elements of input_vars_last
for elem in input_vars_last
    v_ind = elem
    v = varDict[:x][v_ind]
    dv_ind = integration_map[elem]
    if dv_ind == varDict[:y]
        dv = varDict[:y]
    else
        dv = varDict[:x][dv_ind]
    end

    next_v = v + dt*dv
    push!(timestep_nplus1_vars, next_v)
    @objective(mipModel, Min, next_v)
    JuMP.optimize!(mipModel)
    objective_bound(mipModel)
    push!(lows, objective_bound(mipModel))
    @objective(mipModel, Max, next_v)
    JuMP.optimize!(mipModel)
    objective_bound(mipModel)
    push!(highs, objective_bound(mipModel))
end

lows
highs