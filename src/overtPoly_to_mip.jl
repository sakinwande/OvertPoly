include("overtPoly_helpers.jl")
include("overt_to_pwa.jl")
include("problems.jl")
using JuMP, Gurobi, MathOptInterface
# ENV["GRB_LICENSE_FILE"] = "/barrett/scratch/akinwande/gurobi.lic"

function ccEncoding!(xS, yLB, yUB, Tri, query::GraphPolyQuery,sym, ind, model)
    """
    Method to encode a piecewise affine function as a mixed integer program following the convex combination method as defined in Gessler et. al. 2012
        (https://www.dl.behinehyab.com/Ebooks/IP/IP011_655874_www.behinehyab.com.pdf#page=308)
    Adds input and output variables to the query variable dictionary with the key sym

    args:
        problem: OvertPProblem that encodes the dynamics of the system as well as some useful system info 
        xS: List of vertices of the triangulation
        yLB: List of lower bounds of the function at the vertices of the triangulation
        yUB: List of upper bounds of the function at the vertices of the triangulation
        Tri: List of simplices of the triangulation
        ind: index of the model in the problem varList
    """
    #Following the convention from Gessler et. al. 

    m = size(xS, 1) #Number of vertices
    d = size(xS[1], 1) #Dimension of the space
    n = size(Tri, 1) #Number of simplices

    uCoef = query.problem.control_coef[ind][1]
    #Define convex coefficients as a MIP variable 
    #Overkill but doing this for sanity 
    λ =  @variable(model, λ[1:m])
    b = @variable(model, b[1:n], Bin)

    #Begin constraining our auxilliary variables
    #Convex combiation constraints (Gessler et. al. eq. 3.2)
    for i in 1:m
        @constraint(model, λ[i] >= 0)
    end
    @constraint(model, sum(λ) == 1)

    #This is equation 3.4 from Gessler et. al.
    #Here, we enforce the SOS-2 constraint on the convex coefficients. In short, we require that only vertices of adjacent simplices can be active at a time

    #NOTE This relates a convex coefficient to its neighbors 
    for j in 1:m
        #Below we find all simplices where index j is present 
        @constraint(model, λ[j] <= sum(b[i] for i in findall(x -> j in x, Tri)))
    end

    #Next, enforce that at most one simplex can be active at a time (Gessler et. al. eq. 3.5)
    @constraint(model, sum(b) <= 1)

    #Now, define function variables as MIP variables
    #the vertices are defined as a vector 
    @variable(model, x[1:d])
    @variable(model, y)
    @variable(model, yᵤ)
    @variable(model, yₗ)
    @variable(model, u) #Note that this assumes state variable has at most one control input  


    #Define the input as the convex combination of neighboring vertices
    #TODO: Check if this is sound. Appears to be 
    intermVar = sum(λ[i]*[xS[i]...] for i in 1:m)
    for i in 1:d
        @constraint(model, x[i] == intermVar[i])
    end


    
    #Define the generic function value in terms of the convex combination of its upper and lower bounds
    #NOTE: Control is changed here. Very bad 
    @constraint(model, yₗ == sum(λ[i]*yLB[i] for i in 1:m))
    @constraint(model, yᵤ == sum(λ[i]*yUB[i] for i in 1:m))
    @constraint(model, yₗ + uCoef*u <= y) #Vector valued lower bound contraint
    @constraint(model, y <= yᵤ + uCoef*u) #Vector valued upper bound constraint

    #Add model inputs and outputs to variable dictionary
    #NOTE: 
    query.var_dict[sym] = [x, [y], [u]]

    #We will also need to define additional constraints on x and y, but those will be added later
    return model 

end

function ccEncoding!(xS, yLB, yUB, Tri, query::FlatPolyQuery,sym, ind)
    """
    Method to encode a piecewise affine function as a mixed integer program following the convex combination method as defined in Gessler et. al. 2012
        (https://www.dl.behinehyab.com/Ebooks/IP/IP011_655874_www.behinehyab.com.pdf#page=308)
    Adds input and output variables to the query variable dictionary with the key sym

    args:
        problem: OvertPProblem that encodes the dynamics of the system as well as some useful system info 
        xS: List of vertices of the triangulation
        yLB: List of lower bounds of the function at the vertices of the triangulation
        yUB: List of upper bounds of the function at the vertices of the triangulation
        Tri: List of simplices of the triangulation
        ind: index of the model in the problem varList
    """

    #Following the convention from Gessler et. al. 

    m = size(xS, 1) #Number of vertices
    d = size(xS[1], 1) #Dimension of the space
    n = size(Tri, 1) #Number of simplices
    dU = query.problem.control_dim #Dimension of the control

    # uCoef = zeros(1, du) #Control coefficients. Only nonzero if control coefficients apply to the function value
    uCoef = query.problem.control_coef[ind]
    #uCoef = reshape(uCoef, 1, dU)
    
    model = query.mod_dict[sym]
    #set_silent(model)

    #Define convex coefficients as a MIP variable 
    #TODO Fix to be a float (not an int).Done 
    lamb_var = Meta.parse("λ_$(sym)")
    bin_var = Meta.parse("b_$(sym)")

    λ = @variable(model, [1:m], base_name = "$lamb_var")
    # query.var_dict[lamb_var] = λ
    #Define binary variables indicating with simplex is active
    b = @variable(model, [1:n], Bin, base_name = "$bin_var")
    # query.var_dict[bin_var] = b

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
    x_sym = Meta.parse("x_$(sym)")
    y_sym = Meta.parse("y_$(sym)")
    yₗ_sym = Meta.parse("yₗ_$(sym)")
    yᵤ_sym = Meta.parse("yᵤ_$(sym)")
    u_sym = Meta.parse("u_$(sym)") 

    #Now, define function variables as MIP variables
    #the vertices are defined as a vector 
    x = @variable(model, [1:d], base_name = "$x_sym")
    # query.var_dict[x_sym] = x
    y = @variable(model, [1:1], base_name = "$y_sym")
    # query.var_dict[y_sym] = y
    yₗ = @variable(model, [1:1], base_name = "$yₗ_sym")
    # query.var_dict[yₗ_sym] = yₗ
    yᵤ = @variable(model, [1:1], base_name = "$yᵤ_sym")
    # query.var_dict[yᵤ_sym] = yᵤ
    #TODO: Change size of control variable as needed. Currently set to equal size of state variable
    u = @variable(model, [1:dU], base_name = "$u_sym")
    # query.var_dict[u_sym] = u

    #Define the generic vertex as a convex combination of its neighbors 
    #This exploits casting. NOTE: Could be dangerous 
    @constraint(model, x .== sum(λ[i]*[xS[i]...] for i in 1:m))

    #Define the generic function value in terms of the convex combination of its upper and lower bounds
    #NOTE: Control is changed here. Very bad 
    @constraint(model, yₗ[1] == sum(λ[i]*yLB[i] for i in 1:m))
    @constraint(model, yᵤ[1] == sum(λ[i]*yUB[i] for i in 1:m))
    @constraint(model, yₗ[1] .+ uCoef.*u .<= y[1]) #Vector valued lower bound contraint
    @constraint(model, y[1] .<= yᵤ[1] .+ uCoef.*u) #Vector valued upper bound constraint

    #Add model inputs and outputs to variable dictionary
    query.var_dict[sym] = [x, y, u]

    #We will also need to define additional constraints on x and y, but those will be added later
    return model 

end

function dccEncoding!(xS, yLB, yUB, Tri, query::GraphPolyQuery, sym, ind, model)
    """
    Disaggregated Convex Combination (DCC) encoding of a piecewise affine function.
    Replaces the global λ variables in CC with per-simplex disaggregated weights δ[i],
    yielding a tighter LP relaxation at the cost of more continuous variables.

    Reference: Vielma, Ahmed & Nemhauser (2010), "Mixed-Integer Models for Nonseparable
    Piecewise-Linear Optimization: Unifying Framework and Extensions"
    """
    d = size(xS[1], 1)    # Dimension of the input space
    n = size(Tri, 1)       # Number of simplices

    uCoef = query.problem.control_coef[ind][1]

    # Binary variables: one per simplex (same as CC)
    b = @variable(model, b[1:n], Bin)
    @constraint(model, sum(b) == 1)

    # Disaggregated weights: one vector per simplex, length d+1
    δ = [@variable(model, [1:length(Tri[i])], lower_bound=0) for i in 1:n]
    for i in 1:n
        @constraint(model, sum(δ[i]) == b[i])
    end

    # State and output variables
    @variable(model, x[1:d])
    @variable(model, y)
    @variable(model, yᵤ)
    @variable(model, yₗ)
    @variable(model, u)

    # x as disaggregated convex combination of vertices
    intermVar = sum(sum(δ[i][k] * [xS[Tri[i][k]]...] for k in 1:length(Tri[i])) for i in 1:n)
    for j in 1:d
        @constraint(model, x[j] == intermVar[j])
    end

    # y bounds via disaggregated weights
    @constraint(model, yₗ == sum(sum(δ[i][k] * yLB[Tri[i][k]] for k in 1:length(Tri[i])) for i in 1:n))
    @constraint(model, yᵤ == sum(sum(δ[i][k] * yUB[Tri[i][k]] for k in 1:length(Tri[i])) for i in 1:n))
    @constraint(model, yₗ + uCoef * u <= y)
    @constraint(model, y <= yᵤ + uCoef * u)

    query.var_dict[sym] = [x, [y], [u]]
    return model
end

function dccEncoding!(xS, yLB, yUB, Tri, query::FlatPolyQuery, sym, ind)
    """
    Disaggregated Convex Combination (DCC) encoding of a piecewise affine function.
    Replaces the global λ variables in CC with per-simplex disaggregated weights δ[i],
    yielding a tighter LP relaxation at the cost of more continuous variables.

    Reference: Vielma, Ahmed & Nemhauser (2010), "Mixed-Integer Models for Nonseparable
    Piecewise-Linear Optimization: Unifying Framework and Extensions"
    """
    d = size(xS[1], 1)    # Dimension of the input space
    n = size(Tri, 1)       # Number of simplices
    dU = query.problem.control_dim

    uCoef = query.problem.control_coef[ind]
    model = query.mod_dict[sym]

    bin_var  = Meta.parse("b_$(sym)")
    delt_var = Meta.parse("δ_$(sym)")
    x_sym    = Meta.parse("x_$(sym)")
    y_sym    = Meta.parse("y_$(sym)")
    yₗ_sym   = Meta.parse("yₗ_$(sym)")
    yᵤ_sym   = Meta.parse("yᵤ_$(sym)")
    u_sym    = Meta.parse("u_$(sym)")

    # Binary variables: one per simplex (same as CC)
    b = @variable(model, [1:n], Bin, base_name = "$bin_var")
    @constraint(model, sum(b) <= 1)

    # Disaggregated weights: one vector per simplex, length d+1
    δ = [@variable(model, [1:length(Tri[i])], lower_bound=0,
                   base_name = "$(delt_var)_$(i)") for i in 1:n]
    for i in 1:n
        @constraint(model, sum(δ[i]) == b[i])
    end

    # State and output variables
    x  = @variable(model, [1:d],  base_name = "$x_sym")
    y  = @variable(model, [1:1],  base_name = "$y_sym")
    yₗ = @variable(model, [1:1],  base_name = "$yₗ_sym")
    yᵤ = @variable(model, [1:1],  base_name = "$yᵤ_sym")
    u  = @variable(model, [1:dU], base_name = "$u_sym")

    # x as disaggregated convex combination of vertices
    @constraint(model, x .== sum(sum(δ[i][k] * [xS[Tri[i][k]]...] for k in 1:length(Tri[i])) for i in 1:n))

    # y bounds via disaggregated weights
    @constraint(model, yₗ[1] == sum(sum(δ[i][k] * yLB[Tri[i][k]] for k in 1:length(Tri[i])) for i in 1:n))
    @constraint(model, yᵤ[1] == sum(sum(δ[i][k] * yUB[Tri[i][k]] for k in 1:length(Tri[i])) for i in 1:n))
    @constraint(model, yₗ[1] .+ uCoef .* u .<= y[1])
    @constraint(model, y[1] .<= yᵤ[1] .+ uCoef .* u)

    query.var_dict[sym] = [x, y, u]
    return model
end

"""
    _dlog_assign_codes(xS, Tri) -> (codes::Matrix{Int}, K_total::Int)

Assign a binary Gray-code identifier to each simplex for the Disaggregated
Logarithmic (DLOG) encoding.

**Algorithm (Vielma, Ahmed & Nemhauser 2010; Geißler et al. in Lee & Leyffer 2012):**

For a tensor-product grid triangulation the domain decomposes into axis-aligned
cells.  Within each cell there may be several simplices (sub-simplices).  We
assign codes as follows:

1. Per input dimension k, extract the sorted unique coordinate values G[k].
   Each cell boundary in dimension k is the interval [G[k][c], G[k][c+1]].

2. For simplex i, find its cell index c[k] (0-based) in each dimension by
   locating the minimum vertex coordinate along that axis in G[k].

3. Apply a reflected Gray code to c[k]:  gray(v) = v ⊻ (v >> 1).
   This ensures adjacent cells differ in exactly one bit, which is the key
   property that allows the logarithmic cut formulation to be valid.

4. Count the maximum number of simplices per cell; allocate
   K_sub = ⌈log₂(max_per_cell)⌉ bits for the intra-cell index.

5. Concatenate the per-dimension Gray bits (K_dims[k] = ⌈log₂(|G[k]|-1)⌉ each)
   and the K_sub intra-cell bits to form the final code of length K_total.

**Soundness preconditions:**
- xS must be a tensor-product grid (each dimension's coordinates form a
  regular lattice).  OVERT always produces such grids.
- Tri indices are 1-based into xS.

**Postconditions:**
- codes[i, l] ∈ {0,1} for all i, l.
- K_total == size(codes, 2).
- For each pair of simplices i₁ ≠ i₂ that lie in different cells, their codes
  differ in at least one of the grid-dimension bits.
"""
function _dlog_assign_codes(xS, Tri)
    n = length(Tri)
    d = length(xS[1])

    # Degenerate case: single simplex → no binary variables needed
    if n == 1
        return zeros(Int, 1, 0), 0
    end

    # -----------------------------------------------------------------------
    # Step 1: build sorted unique coordinate grids per dimension
    # -----------------------------------------------------------------------
    G = [sort(unique(xS[j][k] for j in eachindex(xS))) for k in 1:d]

    # Number of grid intervals per dimension (number of cells along axis k)
    n_cells = [max(1, length(G[k]) - 1) for k in 1:d]

    # Bits needed per dimension via Gray code:  ⌈log2(n_cells[k])⌉
    K_dims = [ceil(Int, log2(max(1, n_cells[k]))) for k in 1:d]

    # -----------------------------------------------------------------------
    # Step 2: find the cell index for each simplex
    # -----------------------------------------------------------------------
    # cell_idx[i][k] is the 0-based cell index of simplex i in dimension k
    cell_idx = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        verts = Tri[i]
        cell_idx[i] = Vector{Int}(undef, d)
        for k in 1:d
            min_coord = minimum(xS[v][k] for v in verts)
            # searchsortedfirst returns the index of the first G[k] element >= min_coord
            # Subtract 1 to get 0-based cell index; clamp to valid range
            ci = searchsortedfirst(G[k], min_coord) - 1
            cell_idx[i][k] = max(0, min(ci, n_cells[k] - 1))
        end
    end

    # -----------------------------------------------------------------------
    # Step 3: count sub-simplices per cell to determine K_sub
    # -----------------------------------------------------------------------
    cell_counts = Dict{Vector{Int}, Int}()
    for i in 1:n
        key = cell_idx[i]
        cell_counts[key] = get(cell_counts, key, 0) + 1
    end
    max_per_cell = maximum(values(cell_counts))
    K_sub = ceil(Int, log2(max(1, max_per_cell)))
    # If max_per_cell == 1, ceil(log2(1)) = 0, meaning no sub-simplex bits needed.

    K_total = sum(K_dims) + K_sub

    # -----------------------------------------------------------------------
    # Step 4: build the code matrix (two-pass)
    # -----------------------------------------------------------------------
    codes = zeros(Int, n, K_total)

    # Track per-cell assignment counter (for sub-simplex index)
    cell_sub_count = Dict{Vector{Int}, Int}()

    for i in 1:n
        key = cell_idx[i]
        sub_idx = get(cell_sub_count, key, 0)
        cell_sub_count[key] = sub_idx + 1

        col = 1  # current bit column (1-based)

        # Per-dimension Gray-coded cell bits
        for k in 1:d
            c = cell_idx[i][k]
            gray_c = c ⊻ (c >> 1)  # reflected Gray code
            for b in 0:(K_dims[k]-1)
                codes[i, col] = (gray_c >> b) & 1
                col += 1
            end
        end

        # Sub-simplex bits (plain binary of sub_idx)
        gray_sub = sub_idx ⊻ (sub_idx >> 1)
        for b in 0:(K_sub-1)
            codes[i, col] = (gray_sub >> b) & 1
            col += 1
        end
    end

    return codes, K_total
end


function dlogEncoding!(xS, yLB, yUB, Tri, query::GraphPolyQuery, sym, ind, model)
    """
    Disaggregated Logarithmic (DLOG) encoding of a piecewise affine function.

    Replaces the n binary variables of DCC with K_total = ⌈log₂(n)⌉ binary
    variables by exploiting the tensor-product grid structure of the OVERT
    triangulation.  The disaggregated continuous weight structure δ[i][k] from
    DCC is preserved, giving the same tight LP relaxation but with fewer binary
    variables → smaller branch-and-bound trees.

    References:
      Vielma, Ahmed & Nemhauser (2010), "Mixed-Integer Models for Nonseparable
        Piecewise-Linear Optimization"
      Geißler et al. in Lee & Leyffer (2012), Chapter 6: "Using Piecewise
        Linear Functions for Solving MINLPs"

    Soundness argument:
      When z is integer, the bit-partition constraints force each δ[i] to be
      zero unless code[i,:] == z[:].  Combined with the per-simplex sum
      Σ_k δ[i][k] ≥ 0 and the global sum-to-one Σ_i Σ_k δ[i][k] = 1, weight
      concentrates on the unique matching simplex (or the problem is infeasible
      if no simplex has code z, which is correct — that region doesn't exist).

    Preconditions:
      - xS forms a tensor-product grid (guaranteed by OVERT composition).
      - Tri contains 1-based vertex indices into xS.

    Postconditions:
      - query.var_dict[sym] = [x, [y], [u]] matching the CC/DCC convention.
      - Model contains K_total binary variables (vs n in CC/DCC).
    """
    d = size(xS[1], 1)    # Dimension of the input space
    n = size(Tri, 1)       # Number of simplices

    uCoef = query.problem.control_coef[ind][1]

    # ------------------------------------------------------------------
    # Assign binary codes to simplices
    # ------------------------------------------------------------------
    codes, K_total = _dlog_assign_codes(xS, Tri)

    # ------------------------------------------------------------------
    # Binary variables: K_total (log-scale), replaces n in DCC
    # ------------------------------------------------------------------
    z = @variable(model, z[1:K_total], Bin)

    # ------------------------------------------------------------------
    # Disaggregated weights: same structure as DCC
    # δ[i] is a vector of length |Tri[i]| of non-negative reals.
    # ------------------------------------------------------------------
    δ = [@variable(model, [1:length(Tri[i])], lower_bound=0) for i in 1:n]

    # Global sum-to-one (replaces Σ b[i] == 1 from DCC)
    @constraint(model, sum(sum(δ[i]) for i in 1:n) == 1)

    # ------------------------------------------------------------------
    # Bit-partition constraints (the logarithmic encoding core)
    # For each bit l, simplices with code[l]==1 must have total weight ≤ z[l],
    # and simplices with code[l]==0 must have total weight ≤ 1 - z[l].
    # Together these force weight onto exactly the simplex matching z.
    # ------------------------------------------------------------------
    for l in 1:K_total
        # Simplices with bit l == 1
        idx1 = findall(i -> codes[i, l] == 1, 1:n)
        # Simplices with bit l == 0
        idx0 = findall(i -> codes[i, l] == 0, 1:n)

        if !isempty(idx1)
            @constraint(model, sum(sum(δ[i]) for i in idx1) <= z[l])
        end
        if !isempty(idx0)
            @constraint(model, sum(sum(δ[i]) for i in idx0) <= 1 - z[l])
        end
    end

    # ------------------------------------------------------------------
    # State and output variables (identical to DCC)
    # ------------------------------------------------------------------
    @variable(model, x[1:d])
    @variable(model, y)
    @variable(model, yᵤ)
    @variable(model, yₗ)
    @variable(model, u)

    # x as disaggregated convex combination of simplex vertices
    intermVar = sum(sum(δ[i][k] * [xS[Tri[i][k]]...] for k in 1:length(Tri[i])) for i in 1:n)
    for j in 1:d
        @constraint(model, x[j] == intermVar[j])
    end

    # y bounds via disaggregated weights
    @constraint(model, yₗ == sum(sum(δ[i][k] * yLB[Tri[i][k]] for k in 1:length(Tri[i])) for i in 1:n))
    @constraint(model, yᵤ == sum(sum(δ[i][k] * yUB[Tri[i][k]] for k in 1:length(Tri[i])) for i in 1:n))
    @constraint(model, yₗ + uCoef * u <= y)
    @constraint(model, y <= yᵤ + uCoef * u)

    query.var_dict[sym] = [x, [y], [u]]
    return model
end


function dlogEncoding!(xS, yLB, yUB, Tri, query::FlatPolyQuery, sym, ind)
    """
    Disaggregated Logarithmic (DLOG) encoding of a piecewise affine function.

    Replaces the n binary variables of DCC with K_total = ⌈log₂(n)⌉ binary
    variables by exploiting the tensor-product grid structure of the OVERT
    triangulation.  The disaggregated continuous weight structure δ[i][k] from
    DCC is preserved, giving the same tight LP relaxation but with fewer binary
    variables → smaller branch-and-bound trees.

    References:
      Vielma, Ahmed & Nemhauser (2010), "Mixed-Integer Models for Nonseparable
        Piecewise-Linear Optimization"
      Geißler et al. in Lee & Leyffer (2012), Chapter 6: "Using Piecewise
        Linear Functions for Solving MINLPs"

    Soundness argument:
      When z is integer, the bit-partition constraints force each δ[i] to be
      zero unless code[i,:] == z[:].  Combined with the per-simplex sum
      Σ_k δ[i][k] ≥ 0 and the global sum-to-one Σ_i Σ_k δ[i][k] = 1, weight
      concentrates on the unique matching simplex (or the problem is infeasible
      if no simplex has code z, which is correct — that region doesn't exist).

    Preconditions:
      - xS forms a tensor-product grid (guaranteed by OVERT composition).
      - Tri contains 1-based vertex indices into xS.

    Postconditions:
      - query.var_dict[sym] = [x, y, u] matching the CC/DCC FlatPolyQuery convention.
      - Model contains K_total binary variables (vs n in CC/DCC).
    """
    d = size(xS[1], 1)    # Dimension of the input space
    n = size(Tri, 1)       # Number of simplices
    dU = query.problem.control_dim

    uCoef = query.problem.control_coef[ind]
    model = query.mod_dict[sym]

    # Base names for JuMP anonymous variables (matches DCC FlatPolyQuery pattern)
    z_var    = Meta.parse("z_$(sym)")
    delt_var = Meta.parse("δ_$(sym)")
    x_sym    = Meta.parse("x_$(sym)")
    y_sym    = Meta.parse("y_$(sym)")
    yₗ_sym   = Meta.parse("yₗ_$(sym)")
    yᵤ_sym   = Meta.parse("yᵤ_$(sym)")
    u_sym    = Meta.parse("u_$(sym)")

    # ------------------------------------------------------------------
    # Assign binary codes to simplices
    # ------------------------------------------------------------------
    codes, K_total = _dlog_assign_codes(xS, Tri)

    # ------------------------------------------------------------------
    # Binary variables: K_total (log-scale), replaces n in DCC
    # ------------------------------------------------------------------
    z = @variable(model, [1:K_total], Bin, base_name = "$z_var")

    # ------------------------------------------------------------------
    # Disaggregated weights: same structure as DCC
    # ------------------------------------------------------------------
    δ = [@variable(model, [1:length(Tri[i])], lower_bound=0,
                   base_name = "$(delt_var)_$(i)") for i in 1:n]

    # Global sum-to-one
    @constraint(model, sum(sum(δ[i]) for i in 1:n) == 1)

    # ------------------------------------------------------------------
    # Bit-partition constraints
    # ------------------------------------------------------------------
    for l in 1:K_total
        idx1 = findall(i -> codes[i, l] == 1, 1:n)
        idx0 = findall(i -> codes[i, l] == 0, 1:n)

        if !isempty(idx1)
            @constraint(model, sum(sum(δ[i]) for i in idx1) <= z[l])
        end
        if !isempty(idx0)
            @constraint(model, sum(sum(δ[i]) for i in idx0) <= 1 - z[l])
        end
    end

    # ------------------------------------------------------------------
    # State and output variables (identical to DCC FlatPolyQuery)
    # ------------------------------------------------------------------
    x  = @variable(model, [1:d],  base_name = "$x_sym")
    y  = @variable(model, [1:1],  base_name = "$y_sym")
    yₗ = @variable(model, [1:1],  base_name = "$yₗ_sym")
    yᵤ = @variable(model, [1:1],  base_name = "$yᵤ_sym")
    u  = @variable(model, [1:dU], base_name = "$u_sym")

    # x as disaggregated convex combination of simplex vertices
    @constraint(model, x .== sum(sum(δ[i][k] * [xS[Tri[i][k]]...] for k in 1:length(Tri[i])) for i in 1:n))

    # y bounds via disaggregated weights
    @constraint(model, yₗ[1] == sum(sum(δ[i][k] * yLB[Tri[i][k]] for k in 1:length(Tri[i])) for i in 1:n))
    @constraint(model, yᵤ[1] == sum(sum(δ[i][k] * yUB[Tri[i][k]] for k in 1:length(Tri[i])) for i in 1:n))
    @constraint(model, yₗ[1] .+ uCoef .* u .<= y[1])
    @constraint(model, y[1] .<= yᵤ[1] .+ uCoef .* u)

    query.var_dict[sym] = [x, y, u]
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


