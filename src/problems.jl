using LazySets
using JuMP

export
	DistrOvertProblem,
    DistrOvertQuery,
	RegOvertProblem,
	RegOvertQuery

# mutable struct OvertPProblem
# 	expr
# 	dec_expr
# 	control_coef
#     domain::Hyperrectangle
# 	varList #List of variables that have OVERT bounds
#     bounds #List of bounds in the same order as varList
# 	dynamics::Function 
# 	bound_func::Function
# 	control_func::Function
# 	link_func::Function
# end

# ##Can include input vars and control vars 
# #true_dynamics
# #input_vars
# #control_vars

# mutable struct OvertPQuery
# 	problem::OvertPProblem
# 	network_file::Union{Nothing,String}
# 	last_layer_activation ##::ActivationFunction
# 	solver::String
# 	ntime::Int64
# 	dt::Float64
# 	N_overt::Int64
# 	var_dict::Union{Nothing,Dict{Symbol,Vector{AbstractVector{VariableRef}}}} #holds [x, y, u]
# 	mod_dict::Union{Nothing,Dict{Symbol,Any}}
# 	case #Determines if variables are case 1(x, y, z etc) or case 2 (x, dx, y, dy, etc)
# end

#Struct for distributed problems
mutable struct DistrOvertProblem
	expr
	dec_expr
	control_coef
    domain::Hyperrectangle
	varList #List of variables that have OVERT bounds
    bounds #List of bounds in the same order as varList
	dynamics::Function 
	bound_func::Function
	control_func::Function
	link_func::Function
end

mutable struct DistrOvertQuery
	problem::DistrOvertProblem
	network_file::Union{Nothing,String}
	last_layer_activation ##::ActivationFunction
	solver::String
	ntime::Int64
	dt::Float64
	N_overt::Int64
	var_dict::Union{Nothing,Dict{Symbol,Vector{AbstractVector{VariableRef}}}} #holds [x, y, u]
	mod_dict::Union{Nothing,Dict{Symbol,Any}}
	case #Determines if variables are case 1(x, y, z etc) or case 2 (x, dx, y, dy, etc)
end


#Struct for regular problems 
mutable struct RegOvertProblem
	expr
	dec_expr
	control_coef
	control_dim
    domain::Hyperrectangle
	varList #List of variables that have OVERT bounds
    bounds #List of bounds in the same order as varList
	update_rule
	dynamics::Function 
	bound_func::Function
	control_func::Function
end

mutable struct RegOvertQuery
	problem::RegOvertProblem
	network_file::Union{Nothing,String}
	last_layer_activation ##::ActivationFunction
	solver::String
	ntime::Int64
	dt::Float64
	N_overt::Int64
	var_dict::Union{Nothing,Dict{Symbol,Vector{AbstractVector{VariableRef}}}} #holds [x, y, u]
	mod_dict::Union{Nothing,Dict{Symbol,Any}}
	case #Determines if variables are case 1(x, y, z etc) or case 2 (x, dx, y, dy, etc)
end

