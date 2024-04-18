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
	varList #List of variables that have OVERT bounds
    bounds #List of bounds in the same order as varList
    update_rule
end

##Can include input vars and control vars 
#true_dynamics
#input_vars
#control_vars

mutable struct OvertPQuery
	problem::OvertPProblem
	bound_func::Function
	network_file::Union{Nothing,String}
	last_layer_activation ##::ActivationFunction
	solver::String
	ntime::Int64
	dt::Float64
	N_overt::Int64
	var_dict::Union{Nothing,Dict{Symbol,Vector{AbstractVector{VariableRef}}}} #holds [x, y, u]
	mod_dict::Union{Nothing,Dict{Symbol,JuMP.Model}}
	case #Determines if variables are case 1(x, y, z etc) or case 2 (x, dx, y, dy, etc)
end