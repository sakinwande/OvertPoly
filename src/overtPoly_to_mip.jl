include("overtPoly_helpers.jl")
include("overt_to_pwa.jl")
using PiecewiseLinearOpt, JuMP, Gurobi

function ccEncoding(xS, yLB, yUB, Tri)
    """
    Method to encode a piecewise affine function as a mixed integer program following the convex combination method as defined in Gessler et. al. 2012
    """

    optimizer = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

    #Following the convention from Gessler et. al. 

    m = size(xS, 1) #Number of vertices
    d = size(xS[1], 1) #Dimension of the space
    n = size(Tri, 1) #Number of simplices

    model = Model(optimizer)

    #Define convex coefficients as a MIP variable 
    #TODO Fix to be a float (not an int)
    @variable(model, λ[1:m], Int)
    #Define binary variables indicating with simplex is active
    @variable(model, z[1:n], Bin)

    #Begin constraining our auxilliary variables
    #Convex combiation constraints (Gessler et. al. eq. 3.2)
    @constraint(model, λ .>= 0)
    @constraint(model, sum(λ) == 1)

    #This is equation 3.4 from Gessler et. al.
    #Here, we iterate through all vertices. Then, we constrain the convex coefficient of each vertex to be leq the sum of the binary variables corresponding to the simplices containing that vertex

    #NOTE This relates a convex coefficient to its neighbors 
    for j in 1:m
        #Below we find all simplices where index j is present 
        @constraint(model, λ[j] <= sum(z[i] for i in findall(x -> j in x, Tri)))
    end

    #Next, enforce that at most one simplex can be active at a time (Gessler et. al. eq. 3.5)
    @constraint(model, sum(z) <= 1)

    #Now, define function variables as MIP variables
    #the vertices are defined as a vector 
    @variable(model, x[1:d])
    @variable(model, y)
    @variable(model, yₗ)
    @variable(model, yᵤ)
    @variable(model, u)

    #Define the generic vertex as a convex combination of its neighbors 
    #This exploits casting. NOTE: Could be dangerous 
    @constraint(model, x .== sum(λ[i]*[xS[i]...] for i in 1:m))

    #Define the generic function value in terms of the convex combination of its upper and lower bounds
    #NOTE: Control is changed here. Very bad 
    @constraint(model, yₗ == sum(λ[i]*yLB[i] + u for i in 1:m))
    @constraint(model, yᵤ == sum(λ[i]*yUB[i] + u for i in 1:m))


    #We will also need to define additional constraints on x and y, but those will be added later
    return model 

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



