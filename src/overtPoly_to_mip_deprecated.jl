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