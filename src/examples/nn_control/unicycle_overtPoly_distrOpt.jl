include("../../overtPoly_helpers.jl")
include("../../nn_mip_encoding.jl")
include("../../overtPoly_to_mip.jl")
include("../../overt_to_pwa.jl")
include("../../problems.jl")
include("../../distr_reachability.jl")
using LazySets
using Dates
using Plasmo
control_coef = [[0],[0],[1],[1]]

controller = "Networks/ARCH-COMP-2023/nnet/controllerUnicycle.nnet"
exprList = [:(cos(x3)*x4), :(sin(x3)*x4), :(0*x3), :(0*x4)]

##Define Unicycle Dynamics#####
function unicycle_dynamics(x, u)
    """
    Dynamics of the unicycle benchmark. NGL I don't know why I use this
    """
    dx1 = x[4]*cos(x[3])
    dx2 = x[4]*sin(x[3])
    dx3 = u[2]
    dx4 = u[1]

    xNew = x + [dx1, dx2, dx3, dx4].*dt

    return xNew
end


function unicycle_control(input_set)
    con_inp_set = input_set 
    return con_inp_set
end

dt = 0.2
#Should be 50?
numSteps = 10
w = 1e-4
domain = Hyperrectangle(low=[9.50,-4.50,2.10,1.50], high = [9.55,-4.45,2.11,1.51])
depMat = [[1,0,1,1],[0,1,1,1], [0,0,1,0], [0,0,0,1]]
########TEST: Debugging Bound Unicycle#########
# lbs, ubs = extrema(domain)
# plotFlag = true
#######################################

###Define Bound Unicycle########
function bound_unicycle_old(Unicycle; plotFlag=false,npoint=2)
    lbs, ubs = extrema(Unicycle.domain)

    #Round the bounds to avoid floating point errors
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
    
    for tup in x1FuncLB_s
        #First find the corresponding f(x) and f(y) values
        #NOTE:   to avoid floating point errors
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x1FuncSub_2LB)[1]
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x1FuncSub_1LB)[1]

        xInd = findall(x->x[1] == tup[1], x1FuncSub_2LB)[1]
        yInd = findall(y->y[1] == tup[2], x1FuncSub_1LB)[1]

        
        

        #Quadratic shift down
        #NOTE: You care about function value, not index value ;)
        #NOTE: Interval subtraction 
        newXY = tup[end] - sx3 * x1FuncSub_1UB[yInd][end] - sx4 * x1FuncSub_2UB[xInd][end] - sx3*sx4
        
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
        newXY = tup[end] - sx3 * x1FuncSub_1LB[yInd][end] - sx4 * x1FuncSub_2LB[xInd][end] - sx3*sx4
        
        push!(x1FuncUB, (tup[1:end-1]..., newXY))
    end

    #Check if bounds are valid by plotting the surface
    if plotFlag
        xS = unique!(Any[tup[1] for tup in x1FuncLB])
        yS = unique!(Any[tup[2] for tup in x1FuncLB])

        surfDim = (size(yS)[1], size(xS)[1])
        baseFunc = exprList[1]

        #Plot the surface
        plotSurf(baseFunc, x1FuncLB, x1FuncUB, surfDim, xS, yS, true)
    end
    #############Next, bound dx2 (dx2 = x4*sin(x3))#####
    #Bound first component of dx2 (x4)
    x2FuncSub1 = :(1*x4)
    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(x2FuncSub1, lb_x4, ub_x4)...)

    #Bound second component of dx2 (sin(x3))
    x2FuncSub2 = :(sin(x3))
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(x2FuncSub2, lb_x3, ub_x3)...)

    #Find how much to shift log x4 by
    sx4 = inpShiftLog(lb_x4, ub_x4, bounds=x2FuncSub1LB)
    sx3 = inpShiftLog(lb_x3, ub_x3, bounds=x2FuncSub2LB)

    #Apply log
    x2FuncSub1LB_l = [(tup[1:end-1]..., log(tup[end] + sx4)) for tup in x2FuncSub1LB]
    x2FuncSub1UB_l = [(tup[1:end-1]..., log(tup[end] + sx4)) for tup in x2FuncSub1UB]

    x2FuncSub2LB_l = [(tup[1:end-1]..., log(tup[end] + sx3)) for tup in x2FuncSub2LB]
    x2FuncSub2UB_l = [(tup[1:end-1]..., log(tup[end] + sx3)) for tup in x2FuncSub2UB]

    #Add a dimension to prepare for Minkowski sum. Put x3 before x4 :)
    x2FuncSub1LB_ll = addDim(x2FuncSub1LB_l, 1)
    x2FuncSub1UB_ll = addDim(x2FuncSub1UB_l, 1)

    x2FuncSub2LB_ll = addDim(x2FuncSub2LB_l, 2)
    x2FuncSub2UB_ll = addDim(x2FuncSub2UB_l, 2)

    #Combine to get log(x4*sin(x3))
    x2FuncLB_u = unique(MinkSum(x2FuncSub1LB_ll, x2FuncSub2LB_ll))
    x2FuncUB_u = unique(MinkSum(x2FuncSub1UB_ll, x2FuncSub2UB_ll))

    #Combine to get x4*sin(x3)
    x2FuncLB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x2FuncLB_u]
    x2FuncUB_s = [(tup[1:end-1]..., exp(tup[end])) for tup in x2FuncUB_u]

    #Account for the shift
    x2FuncLB = Any[]
    x2FuncUB = Any[]

    for tup in x2FuncLB_s
        #First find the corresponding f(x) and f(y) values
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x2FuncSub2LB)
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x2FuncSub1LB)
        xInd = findall(x->x[1] == tup[1], x2FuncSub2LB)
        yInd = findall(y->y[1] == tup[2], x2FuncSub1LB)

        #Quadratic shift down
        newXY = tup[end] - sx3 * x2FuncSub1UB[yInd][1][end] - sx4 * x2FuncSub2UB[xInd][1][end] - sx3*sx4

        push!(x2FuncLB, (tup[1:end-1]..., newXY))
    end

    for tup in x2FuncUB_s
        #First find the corresponding f(x) and f(y) values
        # xInd = findall(x->x[1] == round(tup[1], digits=5), x2FuncSub2LB)
        # yInd = findall(y->y[1] == round(tup[2], digits=5), x2FuncSub1LB)

        xInd = findall(x->x[1] == tup[1], x2FuncSub2LB)
        yInd = findall(y->y[1] == tup[2], x2FuncSub1LB)

        #Quadratic shift down
        newXY = tup[end] - sx3 * x2FuncSub1LB[yInd][1][end] - sx4 * x2FuncSub2LB[xInd][1][end] - sx3*sx4

        push!(x2FuncUB, (tup[1:end-1]..., newXY))
    end

    #Check if bounds are valid by plotting the surface
    if plotFlag
        xS = unique!(Any[tup[1] for tup in x2FuncLB])
        yS = unique!(Any[tup[2] for tup in x2FuncLB])

        surfDim = (size(yS)[1], size(xS)[1])
        baseFunc = exprList[2]

        #Plot the surface
        plotSurf(baseFunc, x2FuncLB, x2FuncUB, surfDim, xS, yS, true)

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

    emptyList = [1]
    currList = [2,3]

    #Retcon x2FuncLB and x2FuncUB to be unlifted
    x2FuncLB_u = deepcopy(x2FuncLB)
    x2FuncUB_u = deepcopy(x2FuncUB)

    lbs_x2 = lbs[2:4]
    ubs_x2 = ubs[2:4]
    x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs_x2, ubs_x2)

    #############Next, bound dx3 (dx3 = u[2])#####
    #Since dx3 is solely a function of u[2], just use a constant
    x3Func = :(0*x3)
    x3FuncLB, x3FuncUB = interpol_nd(bound_univariate(x3Func, lb_x3, ub_x3)...)

    #############Finally, bound dx4 (dx4 = u[1] + w)#####
    #Here, dx4 is a function of u[1] and a disturbance term. Treat disturbance as a zero mean constant 
    x4Func = :(0*x4)
    x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(x4Func, lb_x4, ub_x4, ϵ = w)...)

    bounds = [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]

    return bounds
end

function bound_unicycle(Unicycle; plotFlag=false, npoint=2)
    lbs, ubs = extrema(Unicycle.domain)
    #Round the bounds to avoid floating point errors

    #TEST: Round domain to avoid floating point errors
    lbs = floor.(lbs, digits=dig)
    ubs = ceil.(ubs, digits=dig)

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
    x1FuncSub_2LB, x1FuncSub_2UB = interpol_nd(bound_univariate(x1FuncSub_2, lb_x3, ub_x3, npoint=npoint)...)

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

    #Check if bounds are valid by plotting the surface
    if plotFlag
        xS = unique!(Any[tup[1] for tup in x1FuncLB])
        yS = unique!(Any[tup[2] for tup in x1FuncLB])

        surfDim = (size(yS)[1], size(xS)[1])
        baseFunc = exprList[1]

        #Plot the surface
        plotSurf(baseFunc, x1FuncLB, x1FuncUB, surfDim, xS, yS, true)
    end
    #############Next, bound dx2 (dx2 = x4*sin(x3))#####
    #Bound first component of dx2 (x4)
    x2FuncSub1 = :(1*x4)
    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(x2FuncSub1, lb_x4, ub_x4)...)

    #Bound second component of dx2 (sin(x3))
    x2FuncSub2 = :(sin(x3))
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(x2FuncSub2, lb_x3, ub_x3, npoint=npoint)...)

    #Lift the bounds to the same space
    #First lift the first component of dx2\
    emptyList = [1]
    currList = [2]
    l_x2FuncSub1LB, l_x2FuncSub1UB = lift_OA(emptyList, currList, x2FuncSub1LB, x2FuncSub1UB, lbs[3:4], ubs[3:4])

    #Next lift the second component of dx2
    emptyList = [2]
    currList = [1]
    l_x2FuncSub2LB, l_x2FuncSub2UB = lift_OA(emptyList, currList, x2FuncSub2LB, x2FuncSub2UB, lbs[3:4], ubs[3:4])

    #Combine to get x4*sin(x3)
    x2FuncLB, x2FuncUB = prodBounds(l_x2FuncSub1LB, l_x2FuncSub1UB, l_x2FuncSub2LB, l_x2FuncSub2UB)
    #Check if bounds are valid by plotting the surface
    if plotFlag
        xS = unique!(Any[tup[1] for tup in x2FuncLB])
        yS = unique!(Any[tup[2] for tup in x2FuncLB])

        surfDim = (size(yS)[1], size(xS)[1])
        baseFunc = exprList[2]

        #Plot the surface
        plotSurf(baseFunc, x2FuncLB, x2FuncUB, surfDim, xS, yS, true)

    end

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
   
   emptyList = [1]
   currList = [2,3]
   
   #Retcon x2FuncLB and x2FuncUB to be unlifted
   x2FuncLB_u = deepcopy(x2FuncLB)
   x2FuncUB_u = deepcopy(x2FuncUB)
   
   lbs_x2 = lbs[2:4]
   ubs_x2 = ubs[2:4]
   x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs_x2, ubs_x2)
   
   #############Next, bound dx3 (dx3 = u[2])#####
   #Since dx3 is solely a function of u[2], just use a constant
   x3Func = :(0*x3)
   x3FuncLB, x3FuncUB = interpol_nd(bound_univariate(x3Func, lb_x3, ub_x3)...)
   
   #############Finally, bound dx4 (dx4 = u[1] + w)#####
   #Here, dx4 is a function of u[1] and a disturbance term. Treat disturbance as a zero mean constant 
   x4Func = :(0*x4)
   x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(x4Func, lb_x4, ub_x4, ϵ = w)...)
   
   bounds = [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
   
   return bounds
end

function bound_unicycle_us(Unicycle; plotFlag=false,npoint=2)
    lbs, ubs = extrema(Unicycle.domain)

    #TEST: Round domain to avoid floating point errors
    lbs = floor.(lbs, digits=dig)
    ubs = ceil.(ubs, digits=dig)

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
    x1FuncLB, x1FuncUB = prodBounds2(l_x1FuncSub_1LB, l_x1FuncSub_1UB, l_x1FuncSub_2LB, l_x1FuncSub_2UB)

    #Check if bounds are valid by plotting the surface
    if plotFlag
        xS = unique!(Any[tup[1] for tup in x1FuncLB])
        yS = unique!(Any[tup[2] for tup in x1FuncLB])

        surfDim = (size(yS)[1], size(xS)[1])
        baseFunc = exprList[1]

        #Plot the surface
        plotSurf(baseFunc, x1FuncLB, x1FuncUB, surfDim, xS, yS, true)
    end
    #############Next, bound dx2 (dx2 = x4*sin(x3))#####
    #Bound first component of dx2 (x4)
    x2FuncSub1 = :(1*x4)
    x2FuncSub1LB, x2FuncSub1UB = interpol_nd(bound_univariate(x2FuncSub1, lb_x4, ub_x4)...)

    #Bound second component of dx2 (sin(x3))
    x2FuncSub2 = :(sin(x3))
    x2FuncSub2LB, x2FuncSub2UB = interpol_nd(bound_univariate(x2FuncSub2, lb_x3, ub_x3)...)

    #Lift the bounds to the same space
    #First lift the first component of dx2\
    emptyList = [1]
    currList = [2]
    l_x2FuncSub1LB, l_x2FuncSub1UB = lift_OA(emptyList, currList, x2FuncSub1LB, x2FuncSub1UB, lbs[3:4], ubs[3:4])

    #Next lift the second component of dx2
    emptyList = [2]
    currList = [1]
    l_x2FuncSub2LB, l_x2FuncSub2UB = lift_OA(emptyList, currList, x2FuncSub2LB, x2FuncSub2UB, lbs[3:4], ubs[3:4])

    #Combine to get x4*sin(x3)
    x2FuncLB, x2FuncUB = prodBounds2(l_x2FuncSub1LB, l_x2FuncSub1UB, l_x2FuncSub2LB, l_x2FuncSub2UB)
    #Check if bounds are valid by plotting the surface
    if plotFlag
        xS = unique!(Any[tup[1] for tup in x2FuncLB])
        yS = unique!(Any[tup[2] for tup in x2FuncLB])

        surfDim = (size(yS)[1], size(xS)[1])
        baseFunc = exprList[2]

        #Plot the surface
        plotSurf(baseFunc, x2FuncLB, x2FuncUB, surfDim, xS, yS, true)

    end

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
   
   emptyList = [1]
   currList = [2,3]
   
   #Retcon x2FuncLB and x2FuncUB to be unlifted
   x2FuncLB_u = deepcopy(x2FuncLB)
   x2FuncUB_u = deepcopy(x2FuncUB)
   
   lbs_x2 = lbs[2:4]
   ubs_x2 = ubs[2:4]
   x2FuncLB, x2FuncUB = lift_OA(emptyList, currList, x2FuncLB_u, x2FuncUB_u, lbs_x2, ubs_x2)
   
   #############Next, bound dx3 (dx3 = u[2])#####
   #Since dx3 is solely a function of u[2], just use a constant
   x3Func = :(0*x3)
   x3FuncLB, x3FuncUB = interpol_nd(bound_univariate(x3Func, lb_x3, ub_x3)...)
   
   #############Finally, bound dx4 (dx4 = u[1] + w)#####
   #Here, dx4 is a function of u[1] and a disturbance term. Treat disturbance as a zero mean constant 
   x4Func = :(0*x4)
   x4FuncLB, x4FuncUB = interpol_nd(bound_univariate(x4Func, lb_x4, ub_x4, ϵ = w)...)
   
   bounds = [[x1FuncLB, x1FuncUB], [x2FuncLB, x2FuncUB], [x3FuncLB, x3FuncUB], [x4FuncLB, x4FuncUB]]
   
   return bounds
end

###Next Define function to link control and relevant dynamics###
function unicycle_dyn_con_link!(query, neurons, graph, dynModel, netModel, t_ind=nothing)

    #Define variables that are inputs to the network model
    @variable(netModel, x1)
    @variable(netModel, x2)
    @variable(netModel, x3)
    @variable(netModel, x4)

    #Specify inputs to the network
    @constraint(netModel, neurons[1][1] == x1)
    @constraint(netModel, neurons[1][2] == x2)
    @constraint(netModel, neurons[1][3] == x3)
    @constraint(netModel, neurons[1][4] == x4)

    #Link network inputs to appropriate dynamics models
    @linkconstraint(graph, netModel[:x1] == dynModel[1][:x][1])
    @linkconstraint(graph, netModel[:x2] == dynModel[2][:x][1])
    @linkconstraint(graph, netModel[:x3] == dynModel[3][:x][1])
    @linkconstraint(graph, netModel[:x4] == dynModel[4][:x][1])

    #Link network output to x3 and x4
    @variable(netModel, u1)
    @variable(netModel, u2)

    #Normalizing output of the network as well
    #NOTE: This was the source of our error. Chelsea already normalized the output in the NNET file.
    @constraint(netModel, u1 == neurons[end][1])
    @constraint(netModel, u2 == neurons[end][2])

    @linkconstraint(graph, netModel[:u1] == dynModel[4][:u])
    @linkconstraint(graph, netModel[:u2] == dynModel[3][:u])

    #Finally, identify pertinent input variable for each model
    i = 0
    for sym in query.problem.varList
        if !isnothing(t_ind)
            sym_t = Meta.parse("$(sym)_$(t_ind)")
        else
            sym_t = sym
        end
        i += 1
        pertVar = dynModel[i][:x][1]
        push!(query.var_dict[sym_t], [pertVar])
    end

end

Unicycle = GraphPolyProblem(
    exprList, #dynamics
    nothing, #no decomposed dynamics
    control_coef, #control coefficients
    domain, #input domain
    [:x1,:x2,:x3,:x4], #Variables that have bounds
    nothing, #undefined bounds to start
    unicycle_dynamics,
    bound_unicycle,
    unicycle_control,
    unicycle_dyn_con_link!
)

query = GraphPolyQuery(
    Unicycle, #problem
    controller, #path to network file
    Id(), #last layer activation 
    "MIP", #query solver, can be MIP or Marabou 
    numSteps, #number of discrete intervals 
    dt, #time step size 
    2, #N_overt
    nothing, #var_dict
    nothing, #mod_dict
    2 #case. Delete this param
)

dig=15
#Next, test multi-step concrete reachability
query1 = deepcopy(query);
query1.N_overt = 2
query1.ntime = 1;
@time reachSet, boundSet = concreach!(query1, digits=dig);

query111 = deepcopy(query);
query111.ntime = 1;
query111.problem.bound_func = bound_unicycle_us;
@time reachSetUS, boundSetUS = concreach!(query111, digits=dig);

hypContained(reachSetUS, reachSet, digits=dig)
hypContained(reachSet, reachSetUS, digits=dig)

volume(reachSet)/volume(reachSetUS)

dig= 3
tHor = 15
#Next, test multi-step concrete reachability
query2 = deepcopy(query);
query2.N_overt = 2
query2.ntime = tHor;
@time reachSets, boundSets = multi_step_concreach(query2, digits=dig);

query222 = deepcopy(query);
query222.ntime = tHor;
query222.problem.bound_func = bound_unicycle_us;
@time reachSetsUS, boundSetsUS = multi_step_concreach(query222, digits=dig);


trueFlag = true
for (i,_) in enumerate(reachSets)
    trueFlag = hypContained(reachSetsUS[i], reachSets[i], digits=dig)
    if !trueFlag
        println("Failed at $(i)")
        trueFlag = true
    end
    println("Volume ratio $i: ", volume(reachSets[i])/volume(reachSetsUS[i]))
end

#Recall, boundSets[t] makes reachSets[t+1], but reachSets[t] is used to make boundSets[t], for t >= 1
# t = 6;
# j = 1;
# tf2 = true;

# sLB = gen_interpol_nd(boundSets[t][j][1]);
# sUB = gen_interpol_nd(boundSets[t][j][2]);
# usLB = gen_interpol_nd(boundSetsUS[t][j][1]);
# usUB = gen_interpol_nd(boundSetsUS[t][j][2]);

# usInps1 = [tup[1:end-1] for tup in boundSetsUS[t][j][1]];
# usInps2 = [tup[1:end-1] for tup in boundSetsUS[t][j][2]];
# sInps1 = [tup[1:end-1] for tup in boundSets[t][j][1]];
# sInps2 = [tup[1:end-1] for tup in boundSets[t][j][2]];
# usInps = unique(vcat(usInps1, usInps2));



# lbFlag = true;
# ubFlag = true;
# for (i,inp) in enumerate(usInps)
#     lbFlag = sLB(inp...) <= usLB(inp...)
#     ubFlag = sUB(inp...) >= usUB(inp...)
#     if !lbFlag
#         println("LB failed at $i")
#         lbFlag = true
#     end
#     if !ubFlag
#         println("UB failed at $i")
#         ubFlag = true
#     end
# end

# if j == 1
#     desSet = project(reachSetsUS[t], [1,3,4])
# elseif j == 2
#     desSet = project(reachSetsUS[t], [2,3,4])
# else j == 3
#     desSet = project(reachSetsUS[t], [j])
# end
# #Now try random sampling to be safe
# randomSamples = LazySets.API.sample(desSet,10000);
# for (i,inp) in enumerate(randomSamples)
#     lbFlag = sLB(inp...) <= usLB(inp...)
#     ubFlag = sUB(inp...) >= usUB(inp...)
#     if !lbFlag
#         println("LB failed at $i")
#         lbFlag = true
#     end
#     if !ubFlag
#         println("UB failed at $i")
#         ubFlag = true
#     end
# end

# hypContained(reachSetsUS[t], reachSets[t], digits=dig)

# extrema(reachSets[8])[1][2]
# extrema(reachSetsUS[8])[1][2]

########################################################
########################################################
digs= 3
t_sym = 3
#Next, test direct symreach 
tMid = t_sym
query3 = deepcopy(query);
query3.problem.bounds = boundSets;
query3.ntime = tMid;
@time sym_set = symreach(query3,reachSets, depMat,tMid,digits=digs);

# midQuery = deepcopy(query);
# midQuery.problem.domain = sym_set;
# midQuery.ntime= t_sym - tMid
# reachSetsMid, boundSetsMid = multi_step_concreach(midQuery, digits=digs);
# query3.problem.bounds = boundSetsMid;
# query3.ntime = t_sym - tMid;
# @time sym_set = symreach(query3,reachSetsMid, depMat,t_sym-tMid,digits=digs);

# query3v1 = deepcopy(query);
# concInt = [5,5]
# @time sym_setv1, conc_v1 = multi_step_hybreach(query3v1, depMat, concInt);


query333 = deepcopy(query);
query333.problem.bounds = boundSetsUS;
query333.problem.bound_func = bound_unicycle_us;
query333.ntime = t_sym;
@time sym_setUS = symreach(query333,reachSetsUS, depMat,t_sym,digits=digs);

volume(sym_set)/volume(sym_setUS)
#volume(sym_setv1[end])/volume(sym_setUS)
# volume(symReachv1)/volume(symReachUS)
# volume(symReachv2)/volume(symReachUS)
extrema(sym_set)
extrema(sym_setUS)
# extrema(sym_setv1[end])


# hypContained(sym_setUS,sym_set, digits=digs)
# hypContained(sym_setUS, sym_setv1[end], digits=digs)
# j = 4
# volume(project(sym_set, [j]))/volume(project(sym_setUS, [j]))
# volume(project(sym_setv1[end], [j]))/volume(project(sym_setUS, [j]))


# #Test hybrid reachability
# concInt = [2,2,2,2,2]
# query4 = deepcopy(query)
# @time reachSets = multi_step_hybreach(query4, depMat, concInt)

###########Trying hybrid symbolic##############
sQuery = deepcopy(query)
sconcInt = [10,10,10,10,10]
sconcInt = [2,2]
#NOTE: sconcInt is marginally safe, to be sound, use tighter horizons
altConcInt = [15,10,10,10,5]
# usConcInt = [5,5]
# usQuery = deepcopy(query)
# usQuery.problem.bound_func = bound_unicycle_us

@time sym_set, sound_conc = multi_step_hybreach(sQuery, depMat, altConcInt);
#@time us_set, us_conc = multi_step_hybreach(usQuery, depMat, usConcInt);

# extrema(sym_set[end])
# extrema(us_set[end])

goal_set  = Hyperrectangle(low=[-0.6, -0.2, -0.06, -0.3], high = [0.6, 0.2, 0.06, 0.3])

extrema(goal_set)
extrema(sym_set[end])

hypContained(sym_set[end], goal_set, digits=2)

floor(extrema(sym_set[end])[1][3], digits=3)
extrema(goal_set)[1][3]

# hypContained(us_set[end], sym_set[end], digits=digs)
# volume(sym_set[end])/volume(us_set[end])


