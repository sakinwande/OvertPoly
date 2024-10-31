include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

lb = -3
ub = 3
v2Func = :(cos(x)*x)
v2FuncLB, v2FuncUB = interpol(bound_univariate(v2Func, lb, ub, npoint =1)...)

v3Func = :(cos(y)*y^2)
v3FuncLB, v3FuncUB = interpol(bound_univariate(v3Func, lb, ub, npoint = 1)...)

#Find out how much to shift each log by
sX = inpShiftLog(lb, ub, bounds = v2FuncLB)
sY = inpShiftLog(lb, ub, bounds = v3FuncLB)

#Apply log 
v2FuncLB_l = [(tup[1:end-1]..., log(tup[end] + sX)) for tup in v2FuncLB]
v2FuncUB_l = [(tup[1:end-1]..., log(tup[end] + sX)) for tup in v2FuncUB]

v3FuncLB_l = [(tup[1:end-1]..., log(tup[end] + sY)) for tup in v3FuncLB]
v3FuncUB_l = [(tup[1:end-1]..., log(tup[end] + sY)) for tup in v3FuncUB]

#Add a dimension to prepare for Minkowski sum
v2FuncLB_ll = addDim(v2FuncLB_l, 2)
v2FuncUB_ll = addDim(v2FuncUB_l, 2)

v3FuncLB_ll = addDim(v3FuncLB_l, 1)
v3FuncUB_ll = addDim(v3FuncUB_l, 1)

#Combine to get log(xcos(x)cos(y)y^2)
funcLB_l = MinkSum(v2FuncLB_ll, v3FuncLB_ll)
funcUB_l = MinkSum(v2FuncUB_ll, v3FuncUB_ll)

#Apply exp to get bounds for xcos(x)cos(y)y^2
funcLB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in funcLB_l]
funcUB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in funcUB_l]

#Account for the shift 
funcLB = []
funcUB = []
for tup in funcLB_s
    #First find the corresponding f(x) and f(y) values
    xInd = findall(x -> x[1] == round(tup[1], digits=5), v2FuncLB)[1]
    yInd = findall(x -> x[1] == round(tup[2], digits=5), v3FuncLB)[1]

    #Quadratic shift down
    newXY = tup[end] - sY * v2FuncUB[xInd][end] - sX * v3FuncUB[yInd][end] - sX * sY

    push!(funcLB, (tup[1:end-1]..., newXY))
end

for tup in funcUB_s
    #First find the corresponding f(x) and f(y) values
    xInd = findall(x -> x[1] == round(tup[1], digits=5), v2FuncUB)[1]
    yInd = findall(x -> x[1] == round(tup[2], digits=5), v3FuncUB)[1]

    #Quadratic shift up
    newXY = tup[end] - sY * v2FuncLB[xInd][end] - sX * v3FuncLB[yInd][end] - sX * sY

    push!(funcUB, (tup[1:end-1]..., newXY))
end

funcLB
funcUB

xS = unique!(Any[tup[1] for tup in funcLB])
yS = unique!(Any[tup[2] for tup in funcLB])

surfDim = (size(yS)[1],size(xS)[1])
expr = :(x*cos(x)*cos(y)*y^2)

plotSurf(expr, funcLB, funcUB, surfDim, xS, yS, true)