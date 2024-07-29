############Constructing a MIP encoding for a bivariate function############
#Directly inspired by the PiecewiseLinearOpt.jl method. Modified here to fit the OvertPoly format
pwl = lbPWL

#A counter is needed to name intermediate variables 
counter = 0

dˣ = [_x[1] for _x in pwl.x]
dʸ = [_x[2] for _x in pwl.x]

#Define grid points for x and y
uˣ, uʸ = unique(dˣ), unique(dʸ)

#Define the triangulation
T = pwl.T

#Number of sample points 
nˣ, nʸ = length(uˣ), length(uʸ)

#There are some border cases where either n==1 or 0 but we can fix later

#Map grid points to indices in the triangulation
ˣtoⁱ = Dict(uˣ[i] => i for i in 1:nˣ)
ʸtoʲ = Dict(uʸ[i] => i for i in 1:nʸ)

fd = Array{Float64}(undef, nˣ, nʸ)

#iterate over each input-output tuple to construct a matrix of output values at each grid point
#TODO: couldn't this be achieved by reshaping the z vector? 
for (v,fv) in zip(pwl.x, pwl.z)
    # i is the linear index into pwl.x...really want (i,j) pair
    fd[ˣtoⁱ[v[1]],ʸtoʲ[v[2]]] = fv
end

#Variable for the PWL encoding 
z = JuMP.@variable(model, lower_bound=minimum(fd), upper_bound=maximum(fd), base_name="z_$counter")

#λ seems to be a continuous variable 
λ = JuMP.@variable(model, [1:nˣ,1:nʸ], lower_bound=0, upper_bound=1, base_name="λ_$counter")

#These constraints are in equation 12 of Vielma(2018)
JuMP.@constraint(model, sum(λ) == 1)
JuMP.@constraint(model, sum(λ[i,j]*uˣ[i]   for i in 1:nˣ, j in 1:nʸ) == x[1])
JuMP.@constraint(model, sum(λ[i,j]*uʸ[j]   for i in 1:nˣ, j in 1:nʸ) == x[2])
JuMP.@constraint(model, sum(λ[i,j]*fd[i,j] for i in 1:nˣ, j in 1:nʸ) == z)

# formulations with SOS2 along each dimension
Tx = [sum(λ[tˣ,tʸ] for tˣ in 1:nˣ) for tʸ in 1:nʸ]
Ty = [sum(λ[tˣ,tʸ] for tʸ in 1:nʸ) for tˣ in 1:nˣ]

# n = length(λ)-1
# k = ceil(Int, log2(n))
# y = JuMP.@variable(model, [1:k], Bin, base_name="y_$counter")

#Use LogE formulation for SOS2

function sos2_encoding_constraints!(m, λ, y, h, B)
    n = length(λ)-1
    for b in B
        JuMP.@constraints(m, begin
            dot(b,h[1])*λ[1] + sum(min(dot(b,h[v]),dot(b,h[v-1]))*λ[v] for v in 2:n) + dot(b,h[n])*λ[n+1] ≤ dot(b,y)
            dot(b,h[1])*λ[1] + sum(max(dot(b,h[v]),dot(b,h[v-1]))*λ[v] for v in 2:n) + dot(b,h[n])*λ[n+1] ≥ dot(b,y)
        end)
    end
    return nothing
end

function reflected_gray_codes(k::Int)
    if k == 0
        return Vector{Int}[]
    elseif k == 1
        return [[0],[1]]
    else
        codes′ = reflected_gray_codes(k-1)
        return vcat([vcat(code,0) for code in codes′],
                    [vcat(code,1) for code in reverse(codes′)])
    end
end

function unit_vector_hyperplanes(k::Int)
    hps = Vector{Int}[]
    for i in 1:k
        hp = zeros(Int,k)
        hp[i] = 1
        push!(hps, hp)
    end
    return hps
end

function sos2_logarithmic_formulation!(m::JuMP.Model, λ)
    n = length(λ)-1
    k = ceil(Int,log2(n))
    y = JuMP.@variable(m, [1:k], Bin, base_name="y_$counter")
    sos2_encoding_constraints!(m, λ, y, reflected_gray_codes(k), unit_vector_hyperplanes(k))
    return nothing
end

#Use logE formulation for SOS2
sos2_logarithmic_formulation!(model, Tx)
sos2_logarithmic_formulation!(model, Ty)

Eⁿᵉ = fill(false, nˣ-1, nʸ-1)
for (i,j,k) in pwl.T
    xⁱ, xʲ, xᵏ = pwl.x[i], pwl.x[j], pwl.x[k]
    iiˣ, iiʸ = ˣtoⁱ[xⁱ[1]], ʸtoʲ[xⁱ[2]]
    jjˣ, jjʸ = ˣtoⁱ[xʲ[1]], ʸtoʲ[xʲ[2]]
    kkˣ, kkʸ = ˣtoⁱ[xᵏ[1]], ʸtoʲ[xᵏ[2]]
    IJ = [(iiˣ,iiʸ), (jjˣ,jjʸ), (kkˣ,kkʸ)]
    im = min(iiˣ, jjˣ, kkˣ)
    iM = max(iiˣ, jjˣ, kkˣ)
    jm = min(iiʸ, jjʸ, kkʸ)
    jM = max(iiʸ, jjʸ, kkʸ)
    if ((im,jM) in IJ) && ((iM,jm) in IJ)
        Eⁿᵉ[im,jm] = true
    else
        #@assert (im,jm) in IJ && (iM,jM) in IJ
    end
end

wⁿᵉ = JuMP.@variable(model, [0:2], Bin, base_name="wⁿᵉ_$counter")
for o in 0:2
    Aᵒ = Set{Tuple{Int,Int}}()
    Bᵒ = Set{Tuple{Int,Int}}()
    for offˣ in o:3:(nˣ-2)
        SWinA = true # whether we put the SW corner of the next triangle to cover in set A
        for i in (1+offˣ):(nˣ-1)
            j = i - offˣ
            if !(1 ≤ i ≤ nˣ-1)
                continue
            end
            if !(1 ≤ j ≤ nʸ-1)
                continue # should never happen
            end
            if Eⁿᵉ[i,j] # if we need to cover the edge...
                if SWinA # figure out which set we need to put it in; this depends on previous triangle in our current line
                    push!(Aᵒ, (i  ,j  ))
                    push!(Bᵒ, (i+1,j+1))
                else
                    push!(Aᵒ, (i+1,j+1))
                    push!(Bᵒ, (i  ,j  ))
                end
                SWinA = !SWinA
            end
        end
    end
    for offʸ in (3-o):3:(nʸ-1)
        SWinA = true
        for j in (offʸ+1):(nʸ-1)
            i = j - offʸ
            if !(1 ≤ i ≤ nˣ-1)
                continue
            end
            if Eⁿᵉ[i,j]
                if SWinA
                    push!(Aᵒ, (i  ,j  ))
                    push!(Bᵒ, (i+1,j+1))
                else
                    push!(Aᵒ, (i+1,j+1))
                    push!(Bᵒ, (i  ,j  ))
                end
                SWinA = !SWinA
            end
        end
    end
    JuMP.@constraints(model, begin
        sum(λ[i,j] for (i,j) in Aᵒ) ≤     wⁿᵉ[o]
        sum(λ[i,j] for (i,j) in Bᵒ) ≤ 1 - wⁿᵉ[o]
    end)
end

wˢᵉ = JuMP.@variable(model, [0:2], Bin, base_name="wˢᵉ_$counter")
for o in 0:2
    Aᵒ = Set{Tuple{Int,Int}}()
    Bᵒ = Set{Tuple{Int,Int}}()
    for offˣ in o:3:(nˣ-2)
        SEinA = true
        # for i in (1+offˣ):-1:1
            # j = offˣ - i + 2
        for j in 1:(nʸ-1)
            i = nˣ - j - offˣ
            if !(1 ≤ i ≤ nˣ-1)
                continue
            end
            if !Eⁿᵉ[i,j]
                if SEinA
                    push!(Aᵒ, (i+1,j  ))
                    push!(Bᵒ, (i  ,j+1))
                else
                    push!(Aᵒ, (i  ,j+1))
                    push!(Bᵒ, (i+1,j  ))
                end
                SEinA = !SEinA
            end
        end
    end
    for offʸ in (3-o):3:(nʸ-1)
        SEinA = true
        for j in (offʸ+1):(nʸ-1)
            i = nˣ - j + offʸ
            if !(1 ≤ i ≤ nˣ-1)
                continue
            end
            if !Eⁿᵉ[i,j]
                if SEinA
                    push!(Aᵒ, (i+1,j  ))
                    push!(Bᵒ, (i  ,j+1))
                else
                    push!(Aᵒ, (i  ,j+1))
                    push!(Bᵒ, (i+1,j  ))
                end
                SEinA = !SEinA
            end
        end
    end
    JuMP.@constraints(model, begin
        sum(λ[i,j] for (i,j) in Aᵒ) ≤     wˢᵉ[o]
        sum(λ[i,j] for (i,j) in Bᵒ) ≤ 1 - wˢᵉ[o]
    end)
end



JuMP.@objective(model, Max, z)
JuMP.optimize!(model)
JuMP.value(z)

boo = 1

function ccEncoding(xS, yLB, yUB, Tri, query,sym, ind)
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

    optimizer = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

    #Following the convention from Gessler et. al. 

    m = size(xS, 1) #Number of vertices
    d = size(xS[1], 1) #Dimension of the space
    n = size(Tri, 1) #Number of simplices
    dU = query.problem.control_dim #Dimension of the control

    # uCoef = zeros(1, du) #Control coefficients. Only nonzero if control coefficients apply to the function value
    uCoef = query.problem.control_coef[ind]
    uCoef = reshape(uCoef, 1, dU)

    model = Model(optimizer)
    set_silent(model)

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
    @constraint(model, yₗ[1] .+ uCoef*u .<= y[1]) #Vector valued lower bound contraint
    @constraint(model, y[1] .<= yᵤ[1] .+ uCoef*u) #Vector valued upper bound constraint

    #Add model inputs and outputs to variable dictionary
    query.var_dict[sym] = [x, y, u]

    #We will also need to define additional constraints on x and y, but those will be added later
    return model 

end

function add_controller_constraints!(model, network_nnet_address, input_set, input_vars, output_vars; last_layer_activation=Id())
    """
    Encode controller as MIP. Directly taken from OVERTVerify
    """
    #Read network file 
    network = read_nnet(network_nnet_address, last_layer_activation=last_layer_activation)
    #Initialize neurons (adds variables)
    neurons = init_neurons(model, network)
    #Initialize deltas (adds binary variables)
    deltas = init_deltas(model, network)
    #Use Taylor Johnson paper (https://arxiv.org/abs/1708.03322) to get bounds  
    bounds = get_bounds(network, input_set)
    #Add NN MIP model to the given model
    #This is defined in the constraints.jl file. Appears to be the Tjeng paper encoding
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())
    #Relate the NN variables to the dynamics variables
    @constraint(model, input_vars .== neurons[1])  # set inputvars
    @constraint(model, output_vars .== neurons[end])  # set outputvars
    return bounds[end]
end

function encode_dynamics!(query::OvertPQuery)
    """
    Method to encode the dynamics as a MIP (using the Gessing et al. method)
    """
    i = 0
    #For each model with an overt approximation, encode the dynamics in a MIP model, and add to the corresponding dictionaryl
    for sym in query.problem.varList
        i += 1
        LB, UB = query.problem.bounds[i]
        Tri = OA2PWA(LB)
        #These are the vertices of the triangulation
        xS = [(tup[1:end-1]) for tup in LB]
        yUB = [tup[end] for tup in UB]
        yLB = [tup[end] for tup in LB]
        query.mod_dict[sym] = ccEncoding(xS, yLB, yUB, Tri, query,sym,i)
    end
end

function encode_control!(query::OvertPQuery)
    """
    Method to encode the controller as a MIP (using the Tjeng et al. method)
    Modifies the MIP model in place

    TODO: Fix to generalize to multiple control inputs and vector valued dynamics
    """
    input_set = query.problem.domain   ###For controller MIP encoding, need model, network address, input set, input variable names, output variable names
    network_file = query.network_file
    i=1
    for sym in query.problem.varList
        if maximum(query.problem.control_coef[i]) > 0 #Check if any control coefficients are nonzero before trying to encode control
            mipModel = query.mod_dict[sym]
            #Get dictionary of MIP variables 
            input_vars = query.var_dict[sym][1]
            control_vars = query.var_dict[sym][3][1]
            output_vars = query.var_dict[sym][2]

            #Get the inputs expected by the controller
            con_inp_vars, con_inp_set, con_vars = query1.problem.control_func(mipModel, input_vars, control_vars, output_vars, input_set)
            controller_bound = add_controller_constraints!(mipModel, network_file, con_inp_set, con_inp_vars, con_vars)
        end
        i += 1
    end
end