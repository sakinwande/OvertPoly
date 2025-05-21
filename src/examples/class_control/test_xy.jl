include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo

#Define problem parameters
expr = [:(x*y)]
domain = Hyperrectangle(low=[0, 0], high=[1, 1])
# domain1 = Hyperrectangle(low=[-pi2_round, -pi2_round], high=[0, 0])
# domain2 = Hyperrectangle(low=[0, 0], high=[pi2_round, pi2_round])
npoint=1
#Bounding the pendulum. Break into smaller chunks
function bound_xy_ia(npoint) 
    lbs, ubs = extrema(domain)
    p1 = :(x)
    p1_LB_1_1, p1_UB_1_1 = interpol_nd(bound_univariate(p1, lbs[1], ubs[1],npoint=npoint)...)
    p2 = :(y*sin(y))
    p2_LB_1_1, p2_UB_1_1 = interpol_nd(bound_univariate(p2, lbs[2], ubs[2],npoint=npoint)...)

    l_p1_LB_11, l_p1_UB_11 = lift_OA([2], [1], p1_LB_1_1, p1_UB_1_1, lbs, ubs) 
    l_p2_LB_11, l_p2_UB_11 = lift_OA([1], [2], p2_LB_1_1, p2_UB_1_1, lbs, ubs)
    return l_p1_LB_11, l_p1_UB_11, l_p2_LB_11, l_p2_UB_11
end

domain = Hyperrectangle(low=[9.5,-4.5,2.1,1.5], high = [9.55,-4.45,2.11,1.51])

domS = Hyperrectangle(low=[8.780374446737326, -3.809831601924612, 2.491734297717635, 2.03413289454653], high=[8.846129598195503, -3.7438068609287094, 2.5136044305389933, 2.060266816649551])
domUS = Hyperrectangle(low=[8.780374446737326, -3.8098316019246123, 2.4919685925517854, 2.03413289454653], high=[8.846129598195503, -3.74380686092871, 2.513604430538992, 2.060266816649551])

x = (4,5,56,6,7)
length(x)

function raw_points(npoint)
    lbs, ubs = extrema(domain)
    ##Bound initial state variable (dx1 = x4*cos(x3))#####
    #K-A Decomposition exp(ln(x4) + ln(cos(x3)))
    #Bound ln(x4)
    #Weird behavior with Hyperrectangle
    lb_x4 = lbs[4]
    ub_x4 = ubs[4]

    #First bound x4
    x1FuncSub_1 = :(1*x4)
    x1FuncSub_1LB, x1FuncSub_1UB = interpol_nd(bound_univariate(x1FuncSub_1, lb_x4, ub_x4)...)

    #Also bound cos(x3)
    lb_x3 = lbs[3]
    ub_x3 = ubs[3]
    x1FuncSub_2 = :(cos(x3))
    x1FuncSub_2LB, x1FuncSub_2UB = interpol_nd(bound_univariate(x1FuncSub_2, lb_x3, ub_x3)...)

    #Lift the bounds to the same space
    #First lift the first component of dx1
    emptyList = [1]
    currList = [2]
    l_x1FuncSub_1LB, l_x1FuncSub_1UB = lift_OA(emptyList, currList, x1FuncSub_1LB, x1FuncSub_1UB, lbs[3:4], ubs[3:4])

    #Next lift the second component of dx1
    emptyList = [2]
    currList = [1]
    l_x1FuncSub_2LB, l_x1FuncSub_2UB = lift_OA(emptyList, currList, x1FuncSub_2LB, x1FuncSub_2UB, lbs[3:4], ubs[3:4])

    return l_x1FuncSub_1LB, l_x1FuncSub_1UB,l_x1FuncSub_2LB, l_x1FuncSub_2UB 
end

function bound_unicycle_old(npoint)

    lbs, ubs = extrema(domain)
    ##Bound initial state variable (dx1 = x4*cos(x3))#####
    #K-A Decomposition exp(ln(x4) + ln(cos(x3)))
    #Bound ln(x4)
    #Weird behavior with Hyperrectangle
    lb_x4 = lbs[4]
    ub_x4 = ubs[4]

    #First bound x4
    x1FuncSub_1 = :(1*x4)
    x1FuncSub_1LB, x1FuncSub_1UB = interpol_nd(bound_univariate(x1FuncSub_1, lb_x4, ub_x4)...)
    
    #Also bound cos(x3)
    lb_x3 = lbs[3]
    ub_x3 = ubs[3]
    x1FuncSub_2 = :(cos(x3))
    x1FuncSub_2LB, x1FuncSub_2UB = interpol_nd(bound_univariate(x1FuncSub_2, lb_x3, ub_x3)...)

    #Find how much to shift log x4 by 
    sx4 = inpShiftLog(lb_x4, ub_x4, bounds=x1FuncSub_1LB)
    sx3 = inpShiftLog(lb_x3, ub_x3, bounds=x1FuncSub_2LB)

    #Apply log 
    x1FuncSub_1LB_l = [(tup[1:end-1]..., log(tup[end] + sx4)) for tup in x1FuncSub_1LB]
    x1FuncSub_1UB_l = [(tup[1:end-1]..., log(tup[end] + sx4)) for tup in x1FuncSub_1UB]

    x1FuncSub_2LB_l = [(tup[1:end-1]..., log(tup[end] + sx3)) for tup in x1FuncSub_2LB]
    x1FuncSub_2UB_l = [(tup[1:end-1]..., log(tup[end] + sx3)) for tup in x1FuncSub_2UB]

    #Add a dimension to prepare for Minkowski sum. Put x3 before x4 :)
    x1FuncSub_1LB_ll = addDim(x1FuncSub_1LB_l, 1)
    x1FuncSub_1UB_ll = addDim(x1FuncSub_1UB_l, 1)
    
    x1FuncSub_2LB_ll = addDim(x1FuncSub_2LB_l, 2)
    x1FuncSub_2UB_ll = addDim(x1FuncSub_2UB_l, 2)

    #Combine to get log(x4*cos(x3))
    x1FuncLB_l = MinkSum(x1FuncSub_1LB_ll, x1FuncSub_2LB_ll)
    x1FuncUB_l = MinkSum(x1FuncSub_1UB_ll, x1FuncSub_2UB_ll)
    
    #Combine to get x4*cos(x3)
    x1FuncLB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1FuncLB_l]
    x1FuncUB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x1FuncUB_l]
    
    #Account for the shift
    x1FuncLB = Any[]
    x1FuncUB = Any[]
    
    tup = x1FuncLB_s[1]
    for tup in x1FuncLB_s
        #First find the corresponding f(x) and f(y) values
        #NOTE: Round to avoid floating point errors
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x1FuncSub_2LB)[1]
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x1FuncSub_1LB)[1]

        xInd = findall(x->x[1] == tup[1], x1FuncSub_2LB)[1]
        yInd = findall(y->y[1] == tup[2], x1FuncSub_1LB)[1]

        
        #Quadratic shift down
        #NOTE: You care about function value, not index value ;)
        #NOTE: Interval subtraction 
        newXY = tup[end] - sx3 * x1FuncSub_1LB[yInd][end] - sx4 * x1FuncSub_2LB[xInd][end] - sx3*sx4
        
        push!(x1FuncLB, (tup[1:end-1]..., newXY))
    end


    for tup in x1FuncUB_s
        #First find the corresponding f(x) and f(y) values
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x1FuncSub_2LB)[1]
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x1FuncSub_1LB)[1]

        xInd = findall(x->x[1] == tup[1], x1FuncSub_2LB)[1]
        yInd = findall(y->y[1] == tup[2], x1FuncSub_1LB)[1]
        
        #Quadratic shift down
        #NOTE: Interval subtraction
        newXY = tup[end] - sx3 * x1FuncSub_1UB[yInd][end] - sx4 * x1FuncSub_2UB[xInd][end] - sx3*sx4
        
        push!(x1FuncUB, (tup[1:end-1]..., newXY))
    end
    #dx1 and dx2 must be functions of x1 and x2 respectively
   emptyList = [1]
   currList = [2,3]
   
    #Retcon x1FuncLB and x1FuncUB to be unlifted 
    x1FuncLB_u = deepcopy(x1FuncLB)
    x1FuncUB_u = deepcopy(x1FuncUB)

    lbs_x1 = [lbs[1]]
    append!(lbs_x1, lbs[3:4])
    ubs_x1 = [ubs[1]]
    append!(ubs_x1, ubs[3:4])
    x1FuncLB, x1FuncUB = lift_OA(emptyList, currList, x1FuncLB_u, x1FuncUB_u, lbs_x1, ubs_x1)

   return x1FuncLB, x1FuncUB
    #return x1FuncSub_1LB, x1FuncSub_1UB, x1FuncSub_2LB, x1FuncSub_2UB
end

function bound_unicycle(npoint)
    lbs, ubs = extrema(domain)

    ##Bound initial state variable (dx1 = x4*cos(x3))#####
    #Weird behavior with Hyperrectangle
    lb_x4 = lbs[4]
    ub_x4 = ubs[4]

    #First bound x4
    x1FuncSub_1 = :(1*x4)
    x1FuncSub_1LB, x1FuncSub_1UB = interpol_nd(bound_univariate(x1FuncSub_1, lb_x4, ub_x4)...)
    
    #Also bound cos(x3)
    lb_x3 = lbs[3]
    ub_x3 = ubs[3]
    x1FuncSub_2 = :(cos(x3))
    x1FuncSub_2LB, x1FuncSub_2UB = interpol_nd(bound_univariate(x1FuncSub_2, lb_x3, ub_x3)...)

    #Lift the bounds to the same space
    #First lift the first component of dx1
    emptyList = [1]
    currList = [2]
    l_x1FuncSub_1LB, l_x1FuncSub_1UB = lift_OA(emptyList, currList, x1FuncSub_1LB, x1FuncSub_1UB, lbs[3:4], ubs[3:4])

    #Next lift the second component of dx1
    emptyList = [2]
    currList = [1]
    l_x1FuncSub_2LB, l_x1FuncSub_2UB = lift_OA(emptyList, currList, x1FuncSub_2LB, x1FuncSub_2UB, lbs[3:4], ubs[3:4])

    #Combine to get x4*cos(x3)
    x1FuncLB, x1FuncUB = prodBounds(l_x1FuncSub_1LB, l_x1FuncSub_1UB, l_x1FuncSub_2LB, l_x1FuncSub_2UB)

    # #Check if bounds are valid by plotting the surface
    # if plotFlag
    #     xS = unique!(Any[tup[1] for tup in x1FuncLB])
    #     yS = unique!(Any[tup[2] for tup in x1FuncLB])

    #     surfDim = (size(yS)[1], size(xS)[1])
    #     baseFunc = exprList[1]

    #     #Plot the surface
    #     plotSurf(baseFunc, x1FuncLB, x1FuncUB, surfDim, xS, yS, true)
    # end
    # #############Next, bound dx2 (dx2 = x4*sin(x3))#####
    # #Bound first component of dx2 (x4)
    # x2FuncSub1 = :(1*x4)
    # x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(x2FuncSub1, lb_x4, ub_x4)...)

    # #Bound second component of dx2 (sin(x3))
    # x2FuncSub2 = :(sin(x3))
    # x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(x2FuncSub2, lb_x3, ub_x3)...)

    # #Lift the bounds to the same space
    # #First lift the first component of dx2\
    # emptyList = [1]
    # currList = [2]
    # l_x2FuncSub1LB, l_x2FuncSub1UB = lift_OA(emptyList, currList, x2FuncSub1LB, x2FuncSub1UB, lbs[3:4], ubs[3:4])

    # #Next lift the second component of dx2
    # emptyList = [2]
    # currList = [1]
    # l_x2FuncSub2LB, l_x2FuncSub2UB = lift_OA(emptyList, currList, x2FuncSub2LB, x2FuncSub2UB, lbs[3:4], ubs[3:4])

    # #Combine to get x4*sin(x3)
    # x2FuncLB, x2FuncUB = prodBounds(l_x2FuncSub1LB, l_x2FuncSub1UB, l_x2FuncSub2LB, l_x2FuncSub2UB)
    # #Check if bounds are valid by plotting the surface
    # if plotFlag
    #     xS = unique!(Any[tup[1] for tup in x2FuncLB])
    #     yS = unique!(Any[tup[2] for tup in x2FuncLB])

    #     surfDim = (size(yS)[1], size(xS)[1])
    #     baseFunc = exprList[2]

    #     #Plot the surface
    #     plotSurf(baseFunc, x2FuncLB, x2FuncUB, surfDim, xS, yS, true)

    # end

   #dx1 and dx2 must be functions of x1 and x2 respectively
   emptyList = [1]
   currList = [2,3]
   
   #Retcon x1FuncLB and x1FuncUB to be unlifted 
   x1FuncUB_u = deepcopy(x1FuncUB)
   x1FuncLB_u = deepcopy(x1FuncLB)
   
   lbs_x1 = [lbs[1]]
   append!(lbs_x1, lbs[3:4])
   ubs_x1 = [ubs[1]]
   append!(ubs_x1, ubs[3:4])
   x1FuncLB, x1FuncUB = lift_OA(emptyList, currList, x1FuncLB_u, x1FuncUB_u, lbs_x1, ubs_x1)
   
    #    emptyList = [1]
    #    currList = [2,3]
    
    #    #Retcon x2FuncLB and x2FuncUB to be unlifted
    #    x2FuncLB_u = deepcopy(x2FuncLB)
    #    x2FuncUB_u = deepcopy(x2FuncUB)
    
    #    lbs_x2 = lbs[2:4]
    #    ubs_x2 = ubs[2:4]
    #    x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs_x2, ubs_x2)
    
    #    #############Next, bound dx3 (dx3 = u[2])#####
    #    #Since dx3 is solely a function of u[2], just use a constant
    #    x3Func = :(0*x3)
    #    x3FuncLB, x3FuncUB = interpol_nd(bound_univariate(x3Func, lb_x3, ub_x3)...)
    
    #    #############Finally, bound dx4 (dx4 = u[1] + w)#####
    #    #Here, dx4 is a function of u[1] and a disturbance term. Treat disturbance as a zero mean constant 
    #    x4Func = :(0*x4)
    #    x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(x4Func, lb_x4, ub_x4, ϵ = w)...)
    
    #    bounds = [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
   
   return x1FuncLB, x1FuncUB
end

lbs1,ubs1 = extrema(domUS)
lb_x31 = lbs1[3]
ub_x31 = ubs1[3]
x1FuncSub_2 = :(cos(x3))
x1FuncSub_2LB, x1FuncSub_2UB = interpol_nd(bound_univariate(x1FuncSub_2, lb_x31, ub_x31)...)
x1FuncSub_2LB_u, x1FuncSub_2UB_u = interpol_nd(bound_univariate(x1FuncSub_2, lb_x31, ub_x31)...)


inps1 = [tup[1:end-1] for tup in x1FuncSub_2UB]
inps2 = [tup[1:end-1] for tup in x1FuncSub_2UB_u]

sound_UB = gen_interpol_nd(x1FuncSub_2UB)
unsound_UB = gen_interpol_nd(x1FuncSub_2UB_u)

truthFlag = true
for (i,inp) in enumerate(inps2)
    truthFlag = sound_UB(inp...) >= unsound_UB(inp...)
    if !truthFlag
        println("Failed at $i")
        truthFlag=true
    end
end

sound_UB(inps1[2]...) >= unsound_UB(inps2[2]...)

############Debugging IA vs exp-Log###############
oldLB, oldUB = bound_unicycle_old(1)
newLB, newUB = bound_unicycle(1)

newUB

oldLB
sort!(oldLB)
sort!(newLB)

truthMat = []
for (i, pt) in enumerate(oldLB)
    push!(truthMat, oldLB[i][end] >= newLB[i][end])
end

truthMat
inpList = [(tup[1:end-1]) for tup in oldLB]

expr = :((x,y,z) -> z*cos(y))
exprFunc = eval(expr)

truthFlag = true
for (i,inp) in enumerate(inpList)
    truthFlag = oldLB[i][end] < exprFunc(inp...)

    if !(truthFlag)
        println("Failure at $i")
    end
end


lb1,ub1,lb2,ub2 = raw_points(1)
