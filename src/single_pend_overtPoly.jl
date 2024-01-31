include("overtPoly_helpers.jl")

pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.0


single_pend_θ_doubledot = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)

baseParsed = parse_and_reduce(single_pend_θ_doubledot)
bbaseParsed = parse_and_reduce(baseParsed[2])

#Bound f(x1)
lb1 = 1.0
ub1 = 1.2
v2Func = bbaseParsed[2]
v2f = Symbolics.build_function(v2Func, find_variables(v2Func)..., expression=Val{false})
v2UB, v2LB = bound_univariate(v2Func, lb1, ub1, plotflag = true) 

#Bound f(x2)
lb2 = 0.0
ub2 = 0.2
v22Func = baseParsed[3]
v22f = Symbolics.build_function(v22Func, find_variables(v22Func)..., expression=Val{false})
v22UB = [(lb2, v22f(lb2)), (ub2, v22f(ub2))]
v22LB = [(lb2, v22f(lb2)), (ub2, v22f(ub2))]

#Bound f(u)
# lbU = -0.979118
# ubU = -0.360603
# v3Func = bbaseParsed[3]
# v3f = Symbolics.build_function(v3Func, find_variables(v3Func)..., expression=Val{false})
# npoint=2
# v3LB = [(lbU, v3f(lbU)), (ub2, v3f(ubU))]
# v3UB = [(lbU, v3f(lbU)), (ub2, v3f(ubU))]

#Bound x2 #For later
# lb3 = 0.
# ub3 = 0.2
# v4Func = baseParsed[3]
# v4f = Symbolics.build_function(v4Func, find_variables(v4Func)..., expression=Val{false})
# v4LB = [(lb3, v4f(lb3)), (ub3, v4f(ub3))]
# v4UB = [(lb3, v4f(lb3)), (ub3, v4f(ub3))]

#For simplicity, define v5 to combine the linear terms

#For future use, interpolate to ensure UB and LB for each is over the same set of points 
nv2LB, nv2UB = interpol(v2LB, v2UB)
nv22LB, nv22UB = interpol(v22LB, v22UB)
# nv3LB, nv3UB = interpol(v3LB, v3UB)

#Log transformation not required because f(x,u) = f(x) + f(u) already

xS = Any[tup[1] for tup in nv2LB]
yS = Any[tup[1] for tup in nv22LB]
# uS = Any[tup[1] for tup in nv3LB]


#Add dimension to each tuple to make Minkowski sum feasible 
#For convenience, introduce new variables for these lifted bounds 

#Add y axis to f(x) overapprox
lv2LBl = addDim(nv2LB, 2)
lv2UBl = addDim(nv2UB, 2)

#Add x axis to f(y) overapprox
lv22LBl = addDim(nv22LB, 1)
lv22UBl = addDim(nv22UB, 1)

#Obtain f(x,y) overapprox
lv2LB = MinkSum(lv2LBl, lv22LBl)
lv2UB = MinkSum(lv2UBl, lv22UBl)

# #Add u axis to f(x,y) overapprox
# lv2LB = addDim(lv2LB, 3)
# lv2UB = addDim(lv2UB, 3)

# #Add x axis to f(u) overapprox
# lv3LBl = addDim(nv3LB, 1)
# lv3UBl = addDim(nv3UB, 1)

# #add y axis to f(u) overapprox
# lv3LBl = addDim(lv3LBl, 2)
# lv3UBl = addDim(lv3UBl, 2)

# #Compute Minkowski sum of these overapproximations 
# #So.. add lower bounds to lower bounds and upper bounds to upper bounds
# lv4LB = MinkSum(lv2LB, lv3LBl)
# lv4UB = MinkSum(lv2UB, lv3UBl)



surfDim = (size(yS)[1],size(xS)[1])
xS
#Again, the exponential is not required because f(x,u) = f(x) + f(u) already

expr = :($(grav_const/pend_len) * sin(x1)  - $(friction/(pend_mass*pend_len^2)) * x2)
plotSurf(expr, [[lb1,ub1],[lb2,ub2]], lv2LB, lv2UB, surfDim, xS, yS, xS, yS, true)

################Convert to MIP######################
LB, UB = lv2LB, lv2UB

include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")

Tri = OA2PWA(LB)
#These are the vertices of the triangulation
xS = [(tup[1:end-1]) for tup in LB]
yUB = [tup[end] for tup in UB]
yLB = [tup[end] for tup in LB]

mipModel = ccEncoding(xS, yLB, yUB, Tri)

###For controller MIP encoding, need model, network address, input set, input variable names, output variable names
using OVERTVerify
network_file = "nnet_files/single_pendulum_small_controller.nnet"
input_set = OVERTVerify.Hyperrectangle(low=[1., 0.], high=[1.2, 0.2])
mipModel.variables