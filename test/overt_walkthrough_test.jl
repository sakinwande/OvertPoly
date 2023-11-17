using Test
using Polyhedra
using CDDLib
using Gurobi
include("../src/overapprox_nd.jl")
using Symbolics
using IntervalRootFinding, IntervalArithmetic


# set_plotflag(true)
# set_plottype("html")
# # simpDyn = [:(sin(x₁))]
# # simpRag = Dict(:x₁ => [-pi, pi])
# # simpDyn_overt = overapprox(simpDyn[1], simpRag,N=2,ϵ=1e-3, rel_error_tol=1e-3)



# include("../src/overt_utils.jl")
# # a, b = get_regions_unary(:sin, -pi, pi)
# # a

# baseExpr = :(cos(x₁) + sin(x₁) + (x₁)^3)


# varSym = find_variables(baseExpr)[1]

# # Replace problem specific variable with x
# strExpr = string(baseExpr)
# strExpr = replace(strExpr, string(varSym) => "xₛ")
# reExpr = Meta.parse(strExpr)

# lb, ub, npoint = -pi, pi, 1
# #What we want is a segmentation of this function into convex and concave regions over a given interval
# #To do this, we can find the zeros of the second derivative and define those as the endpoints of the subintervals


# @variables xₛ
# symExpr = Symbolics.parse_expr_to_symbolic(reExpr, Main)

# D = Differential(xₛ)

# #Define second derivative
# D2 = Differential(xₛ)^2




# #Compute second derivative
# d2f = expand_derivatives(D2(symExpr))
# d2func = eval(Symbolics.build_function(d2f, :xₛ))


# # IntervalArithmetic.Interval(1,2)

# # #Extract symbolic form of second derivative
# # # d2f = Meta.parse(string(d2f))
# # #This was a bit tricky for me. Convert it to a generic function of x
# # # d2func = eval(Symbolics.build_function(d2f, :x))
# # #Then find the roots over the given interval
# rootVals = roots(d2func, IntervalArithmetic.Interval(lb, ub))
# # #TODO: This is not sound, make sound
# # rootsGuess = [mid.([root.interval for root in rootVals])]
# # d2f_zeros = sort(rootsGuess[1])


# # convex = nothing 

# # fun = eval(Symbolics.build_function(baseExpr, varSym))

# # D = Differential(x)
# # df = expand_derivatives(D(eval(reExpr)))
# # df = Meta.parse(string(df))
# # dfunc = eval(Symbolics.build_function(df, :x))

# # UB = bound(fun, lb, ub, npoint; rel_error_tol=rel_error_tol, conc_method="continuous", lowerbound=false, df = dfunc,d2f = d2func, d2f_zeros=d2f_zeros, convex=nothing, plot=true)
# # UBpoints = unique(sort(to_pairs(UB), by = x -> x[1]))

# # LB = bound(fun, lb, ub, npoint; rel_error_tol=rel_error_tol, conc_method="continuous", lowerbound=true, df = dfunc,d2f = d2func, d2f_zeros=d2f_zeros, convex=nothing, plot=true)
# # LBpoints = unique(sort(to_pairs(LB), by = x -> x[1]))

# # #Plot output
# # plotly()
# # global NPLOTS
# # NPLOTS += 1
# # fun_string = string(baseExpr)
# # p = plot(range(lb, ub, length=100), fun, label="function", color="black")
# # plot!(p, [p[1] for p in LBpoints], [p[2] for p in LBpoints],  color="purple", marker=:o, markersize=1, label="lower bound")
# # plot!(p, [p[1] for p in UBpoints], [p[2] for p in UBpoints], color="blue", marker=:diamond, markersize=1,  label="upper bound", legend=:right, title=fun_string, xlabel=string(x))
# # # display(p)
# # savefig(p, "plots/bound_"*string(NPLOTS)*".html")



baseExpr = :(cos(x₁) + sin(x₁) + (x₁)^3)
lb, ub, npoint = -pi, pi, 1

include("overtHZ_helpers.jl")
UBPts, LBPts = bound_univariate(baseExpr, lb, ub)
NPLOTS += 1
fun_string = string(baseExpr)


