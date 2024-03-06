include("overtPoly_helpers.jl")
include("overt_to_pwa.jl")
using PiecewiseLinearOpt, JuMP, Gurobi, MathOptInterface

function ccEncoding(xS, yLB, yUB, Tri, query)
    """
    Method to encode a piecewise affine function as a mixed integer program following the convex combination method as defined in Gessler et. al. 2012
        (https://www.dl.behinehyab.com/Ebooks/IP/IP011_655874_www.behinehyab.com.pdf#page=308)

    args:
        problem: OvertPProblem that encodes the dynamics of the system as well as some useful system info 
        xS: List of vertices of the triangulation
        yLB: List of lower bounds of the function at the vertices of the triangulation
        yUB: List of upper bounds of the function at the vertices of the triangulation
        Tri: List of simplices of the triangulation
    """

    optimizer = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

    #Following the convention from Gessler et. al. 

    m = size(xS, 1) #Number of vertices
    d = size(xS[1], 1) #Dimension of the space
    n = size(Tri, 1) #Number of simplices

    model = Model(optimizer)

    #Define convex coefficients as a MIP variable 
    #TODO Fix to be a float (not an int).Done 
    lamb_var = Meta.parse("λ")
    bin_var = Meta.parse("b")

    λ = @variable(model, [1:m], base_name = "$lamb_var")
    query.var_dict[lamb_var] = λ
    #Define binary variables indicating with simplex is active
    b = @variable(model, [1:n], Bin, base_name = "$bin_var")
    query.var_dict[bin_var] = b

    #Begin constraining our auxilliary variables
    #Convex combiation constraints (Gessler et. al. eq. 3.2)
    @constraint(model, λ .>= 0)
    @constraint(model, sum(λ) == 1)

    #This is equation 3.4 from Gessler et. al.
    #Here, we iterate through all vertices. Then, we constrain the convex coefficient of each vertex to be leq the sum of the binary variables corresponding to the simplices containing that vertex

    #NOTE This relates a convex coefficient to its neighbors 
    for j in 1:m
        #Below we find all simplices where index j is present 
        @constraint(model, λ[j] <= sum(b[i] for i in findall(x -> j in x, Tri)))
    end

    #Next, enforce that at most one simplex can be active at a time (Gessler et. al. eq. 3.5)
    @constraint(model, sum(b) <= 1)

    #Create symbols for anonymous variables
    x_sym = Meta.parse("x")
    y_sym = Meta.parse("y")
    yₗ_sym = Meta.parse("yₗ")
    yᵤ_sym = Meta.parse("yᵤ")
    u_sym = Meta.parse("u") 

    #Now, define function variables as MIP variables
    #the vertices are defined as a vector 
    x = @variable(model, [1:d], base_name = "$x_sym")
    query.var_dict[x_sym] = x
    y = @variable(model, [1], base_name = "$y_sym")
    query.var_dict[y_sym] = y
    yₗ = @variable(model, [1], base_name = "$yₗ_sym")
    query.var_dict[yₗ_sym] = yₗ
    yᵤ = @variable(model, [1], base_name = "$yᵤ_sym")
    query.var_dict[yᵤ_sym] = yᵤ
    u = @variable(model, [1], base_name = "$u_sym")
    query.var_dict[u_sym] = u

    #Define the generic vertex as a convex combination of its neighbors 
    #This exploits casting. NOTE: Could be dangerous 
    @constraint(model, x .== sum(λ[i]*[xS[i]...] for i in 1:m))

    #Define the generic function value in terms of the convex combination of its upper and lower bounds
    #NOTE: Control is changed here. Very bad 
    @constraint(model, yₗ[1] == sum(λ[i]*yLB[i] for i in 1:m))
    @constraint(model, yᵤ[1] == sum(λ[i]*yUB[i] for i in 1:m))
    @constraint(model, yₗ[1] + query.problem.control_coef*u[1] <= y[1])
    @constraint(model, y[1] <= yᵤ[1] + query.problem.control_coef*u[1])


    #We will also need to define additional constraints on x and y, but those will be added later
    return model 

end

function mip_summary(model)
    MathOptInterface = MOI
    const_types = list_of_constraint_types(model)
    l_lin = 0
    l_bin = 0
	println(Crayon(foreground = :yellow), "="^50)
	println(Crayon(foreground = :yellow), "="^18 * " mip summary " * "="^19)
    for i = 1:length(const_types)
        var = const_types[i][1]
        const_type = const_types[i][2]
        l = length(all_constraints(model, var, const_type))
        #println("there are $l constraints of type $const_type with variables type $var.")
        if const_type != MathOptInterface.ZeroOne
            l_lin += l
        else
            l_bin += l
        end
    end
	println(Crayon(foreground = :yellow), "there are $l_lin linear constraints and $l_bin binary constraints.")
	println(Crayon(foreground = :yellow), "="^50)
    println(Crayon(foreground = :white), " ")
end

function list_of_constraint_types(
    model::Model,
)::Vector{Tuple{DataType,DataType}}
    # We include an annotated return type here because Julia fails terribly at
    # inferring it, even though we annotate the type of the return vector.
    return Tuple{DataType,DataType}[
        (jump_function_type(model, F), S) for
        (F, S) in MOI.get(model, MOI.ListOfConstraints())
    ]
end

# expr=:(cos(x)cos(y)x*y^2 + sin(x)cos(y)y)
# lb, ub = -pi, pi
    
# LB, UB = bound_multivariate(expr, lb, ub)

# #There should be fewer simplices than there are vertices (since simplices are convex hulls of vertices)
# #This is actually false. The should be at most 2n - 2 - b simplices given n vertices (from the Euler characteristic)
# Tri = OA2PWA(LB)
# #These are the vertices of the triangulation
# xS = [(tup[1:end-1]) for tup in LB]
# yUB = [tup[end] for tup in UB]
# yLB = [tup[end] for tup in LB]
# mipModel = ccEncoding(xS, yLB, yUB, Tri)


# # #These are the values of the function at the vertices of the triangulation
# # yUB = [tup[end] for tup in UB]
# # yLB = [tup[end] for tup in LB]

# # yVal = yLB

# # lbPWL = BivariatePWLFunction(xS, yLB, Tri)
# # ubPWL = BivariatePWLFunction(xS, yUB, Tri)


