using LazySets
using JuMP

export
    OvertPProblem,
    OvertPQuery

mutable struct OvertPProblem
	expr
	dec_expr
	control_coef
    domain::Hyperrectangle
    bounds
    update_rule
end

##Can include input vars and control vars 
#true_dynamics
#input_vars
#control_vars

mutable struct OvertPQuery
	problem::OvertPProblem
	network_file::String
	last_layer_activation ##::ActivationFunction
	solver::String
	ntime::Int64
	dt::Float64
	N_overt::Int64
	var_dict::Union{Nothing,Dict{Symbol,JuMP.Vector{VariableRef}}}
end