using Symbolics
using IntervalRootFinding, IntervalArithmetic
# using MacroTools: prewalk, postwalk
using Interpolations
using OVERT: bound, find_variables, to_pairs 
using Plots; plotly()
global NPLOTS = 0
# using CSV

function bound_univariate(baseExpr::Expr, lb, ub; ϵ=1e-12, npoint=2, rel_error_tol=5e-3, plotflag=false)
    """
    Method to bound a univariate function

    Args: 
        Expr: A Julia symbolic expression. Assumed to have been parsed/converted to a univariate function 
        lb: Lower bound of the interval over which to bound the function
        ub: Upper bound of the interval over which to bound the function
    """
    #Find the symbolic variable in the Julia expression (for replacement)
    varBase = find_variables(baseExpr)[1]

    #Define differentiation variable
    Symbolics.@variables xₚ

    #Define differentiation operators
    D = Differential(xₚ)
    D2 = Differential(xₚ)^2

    #Replace expression variable with xₚ
    #NB:  we need to do this because the expression will eventually be parsed by Julia Symbolics. For the parser to work, the symbols of the expression must not be defined in the parsing module.

    #I choose xₚ because I can define a case to ensure we don't use this variable in input expressions

    #NOTE: you can actually do better. Convert baseExpr to symExpr in its native form, then use symbolics.get_variables to get the variables in the expression. Next, use Symbolics.substitute to replace problem specific variables with xₚ. This way, you avoid using Meta.parse on a string
    #TODO: Implement this
    strExpr = string(baseExpr)
    strExpr = replace(strExpr, string(varBase) => xₚ)
    standExpr = Meta.parse(strExpr)

    #obtain functions that can be evaluated from the expressions
    f = Symbolics.parse_expr_to_symbolic(standExpr, Main) 
    func = Symbolics.build_function(f, xₚ, expression=false)
    df = expand_derivatives(D(f))
    dFunc = Symbolics.build_function(df, xₚ, expression=false)
    d2f = expand_derivatives(D2(f))
    d2Func = Symbolics.build_function(d2f, xₚ, expression=false)

    if isa(Symbolics.value(f), Number)
        #In this case, the expression is just a constant
        #throw(ArgumentError("Bounds not implemented for constant expressions"))
        #TEST: Return a constant bound
        UBPoints = [(lb, func(lb) + ϵ), (ub, func(ub) + ϵ)]
        LBPoints = [(lb, func(lb) - ϵ), (ub, func(ub) - ϵ)]
    elseif isa(Symbolics.value(df), Number)
        #In this case, the expression is linear
        ϵₗ = ϵ #small parameter to control linear conservativeness
        ϵₗ = 1e-3
        # step = (ub - lb)/(npoint + 1)
        # subints = [lb + i*step for i in 1:npoint]
        # pts = [lb, subints..., ub]
        pts = [lb, ub]
        UBPoints = [(pt, func(pt) + ϵₗ) for pt in pts]
        LBPoints = [(pt, func(pt) - ϵₗ) for pt in pts]
    elseif isa(Symbolics.value(d2f), Number)
        #In this case, the expression is quadratic. This is a constant curvature instance
        if d2f > 0
            #Convex case 
            UB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=false, df = dFunc, d2f = d2Func, d2f_zeros=nothing, convex=true, plot=true)
            LB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=true, df = dFunc, d2f = d2Func, d2f_zeros=nothing, convex=true, plot=true)
            
            UBPoints = unique(sort(to_pairs(UB), by = x->x[1]))
            LBPoints = unique(sort(to_pairs(LB), by = x->x[1]))
        else
            #Concave Case 
            UB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=false, df = dFunc, d2f = d2Func, d2f_zeros=nothing, convex=false, plot=true)
            LB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=true, df = dFunc, d2f = d2Func, d2f_zeros=nothing, convex=false, plot=true)

            # UBPoints = unique(sort(to_pairs(UB), by = x->x[1]))
            # LBPoints = unique(sort(to_pairs(LB), by = x->x[1]))

            UBPoints = sort(to_pairs(UB), by = x->x[1])
            LBPoints = sort(to_pairs(LB), by = x->x[1])
        end
    else
        #Mixed convexity case 
        try 
            #TODO: Review this 
            if standExpr == :(sin(xₚ))
                d2f_zeros, _ = get_sincos_regions(lb, ub)
            elseif standExpr == :(cos(xₚ))
                d2f_zeros, _ = get_sincos_regions(lb, ub, offset=pi/2)
            elseif standExpr == :(tan(xₚ))
                d2f_zeros, _ = get_tan_regions(lb, ub)
            else
                #Find the roots over the given interval 
                rootVals = IntervalRootFinding.roots(d2Func, IntervalArithmetic.Interval(lb, ub))
                #TODO: Fix soundness concerns. Soundness concern follows from the fact that the the roots are floating point numbers and technically the roots returned are an interval and not a single value
                #These intervals have widths on the order of 10^-8
                #Choose midpoints of these intervals 
                rootsGuess = [mid.([root.interval for root in rootVals])]
                d2f_zeros = sort(rootsGuess[1])
            end

            convex = nothing 

            UB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=false, df = dFunc, d2f = d2Func, d2f_zeros=d2f_zeros, convex=convex, plot=true)
            LB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=true, df = dFunc, d2f = d2Func, d2f_zeros=d2f_zeros, convex=convex, plot=true)

            UBPoints = unique(sort(to_pairs(UB), by = x->x[1]))
            LBPoints = unique(sort(to_pairs(LB), by = x->x[1]))
        catch
            #Case where second derivative has no zeros over the interval. This means that over the interval, the function has a constant curvature
            #NOTE: This could theoretically return an error for some other reason. Test this
            if d2Func(lb) > 0
                #Convex case
                UB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=false, df = dFunc, d2f = d2Func, d2f_zeros=nothing, convex=true, plot=true)
                LB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=true, df = dFunc, d2f = d2Func, d2f_zeros=nothing, convex=true, plot=true)

                UBPoints = unique(sort(to_pairs(UB), by = x->x[1]))
                LBPoints = unique(sort(to_pairs(LB), by = x->x[1]))
            else
                #Concave case
                UB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=false, df = dFunc, d2f = d2Func, d2f_zeros=nothing, convex=false, plot=true)
                LB = bound(func, lb, ub, npoint; rel_error_tol=rel_error_tol,conc_method="continuous", lowerbound=true, df = dFunc, d2f = d2Func, d2f_zeros=nothing, convex=false, plot=true)

                UBPoints = unique(sort(to_pairs(UB), by = x->x[1]))
                LBPoints = unique(sort(to_pairs(LB), by = x->x[1]))
            end
        end
    end


    if plotflag
        plotRes2d(baseExpr, func, lb, ub, LBPoints, UBPoints, varBase, true)
    end
    
    return LBPoints, UBPoints
end

function parse_and_reduce(expr::Expr)
    """
    Aim is to develop a method to parse and reduce julia expressions to chunks of like terms for overapproximations. 

    NOTE: Don't try to be efficient, use for loops and lists if needed to get the job done. We can optimize later

    TODO: Does not work yet. Fix later 
    """
    #within the case-specific statements, use commutative, associative, and distributive rules
    #follow a form of pemdas. * and / take precedence over + and -
    #For +/-, 
        #if the the remaining elements of the the subarg list are symbols and/or univariate functions, we can group into expressions of a single variable
        #i.e. [+, cos(x), cos(y), x, y] => [+, (cos(x) + x), (cos(y) + y)]

        #if the remaining elements of the subarg list are multivariate functions, we can decompose them using other case statements and then group them into a vector of subexpressions 
        #i.e. [+, cos(x)cos(y)x*y^2, sin(x)cos(y)y] => [+, [*, xcos(x), cos(y)y^2], [*, sin(x), cos(y)y]]
    #For */division,
        #if the remaining elements of the subarg list are symbols and/or univariate functions, we can group into expressions of a single variable
        #i.e. [*, cos(x), cos(y), x, y] => [*, (cos(x) * x), (cos(y) * y)]

        #if the remaining elements of the subarg list are multivariate functions, we can first use the distributive property, then decompose them using other case statements and then group them into a vector of subexpressions
        #i.e. [*, (cos(x) + cos(y)), x, y^2)] => [+, [*, cos(x), x, y^2], [*, cos(y), x, y^2]] => [+, [*, xcos(x), y^2], [*, x, cos(y)y^2]]

    #Unclear how to handle powers. We can try log transformation, but that's not always possible.
    #First break down every expression into its subexpressions
    subExprVec = postwalk(x -> @show(x) isa Expr ? x.args : x, expr)
    #Next, apply distributive rules to each subexpression
    if subExprVec[1] == :*
        subExprVec = distribMul(subExprVec)
        likeExprVec = [likeTerms(subExprVec, var) for var in find_variables(expr)]
        newExprVec = Any[array2expr(chunk) for chunk in likeExprVec]
    elseif subExprVec[1] == :+ || subExprVec[1] == :-
        newExprVec = Any[array2expr(chunk) for chunk in subExprVec[2:end]]
    end

    pushfirst!(newExprVec, subExprVec[1])
    #Ideally, reduce to chunks of single operation vectors  


    #Then given these single operation vectors, group like terms
    return newExprVec
end

function array2expr(arr)
    """
    Function to take an array of symbols and return and expression of the symbols
    """
    #Define recursively
    newArr = Any[]
    for elem in arr
        if typeof(elem) == Vector{Any}
            elem = array2expr(elem)
        end
        push!(newArr, elem)
    end
    arrExpr = Expr(:call, newArr...)
    return arrExpr
end

function distribMul(arr)
    """
    Function to distribute multiplication 
    """
    @assert arr[1] == :*
    #Check for multiplication over multiplication 
    if arr[1] == arr[2][1]
        #Check for recursive cases
        if arr[2][2][1] == arr[2][1]
            arr[2] = distribMul(arr[2])
        end
        newArr = Any[arr[1], arr[2][2:end]..., arr[3:end]...]
    #Check for multiplication over addition
    elseif arr[2][1] == :+
        #Skip recursive cases for now
        newArr = Any[arr[2][1], Any[Any[arr[1], innAdd, arr[3:end]...] for innAdd in arr[2][2:end]]...]
    else
        newArr = Any[arr[1], arr[2:end]...]
    end

    return newArr
end

function distribDiv(arr)
    """
    Function to distribute division
    """
    @assert arr[1] == :/
    #Check for division over division
    if arr[1] == arr[2][1]
        #Check for recursive cases
        if arr[2][2][1] == arr[2][1]
            arr[2] = distribDiv(arr[2])
        end
        newArr = Any[arr[1], arr[2][2:end]..., arr[3:end]...]
    #Check for division over addition
    elseif arr[2][1] == :+
        #Skip recursive cases for now
        newArr = Any[arr[2][1], Any[Any[arr[1], innAdd, arr[3:end]...] for innAdd in arr[2][2:end]]...]
    else
        newArr = Any[arr[1], arr[2:end]...]
    end

    return newArr
end
function likeTerms(arr, var)
    """
    Given a symbolic variable var, and an array of symbols or simple expressions arr, group like terms
    """
    likeTermList = Any[]
    push!(likeTermList, arr[1])
    for elem in arr[2:end]
        if isa(elem, Symbol)
            if elem == var
                push!(likeTermList, elem)
            end
        elseif isa(elem, Vector{Any})
            if var in elem
                push!(likeTermList, elem)
            end
        end
    end
    return likeTermList
end

function distribAdd(arr)
    """
    Function to distribute addition
    
    Addition arrays can have form [:+, :x, [:+, :x, :y], :z]. Sort to push arrays to the front and symbols to the back
    """
    @assert arr[1] == :+ 
    #Sort to push arrays to the front and symbols to the back
    newArr = Any[]
    for elem in arr[2:end]
        if isa(elem, Symbol)
            push!(newArr, elem)
        elseif isa(elem, Vector{Any})
            pushfirst!(newArr, elem)
        end
    end
    pushfirst!(newArr, arr[1])
    arr = newArr
    
    #Check check for vector of vectors
    if typeof(arr[2]) == Vector{Any} 
        #Check for addition over addition
        if arr[1] == arr[2][1]
            #Check for recursive cases
            if typeof(arr[2][2]) == Vector{Any}
                arr[2] = distribAdd(arr[2])
            end
            newArr = Any[arr[1], arr[2][2:end]..., arr[3:end]...]
        #Check for addition over multiplication.. equiv to multiplication over addition
        elseif arr[2][1] == :*
            #Skip recursive cases for now
            newArr = Any[arr[1], Any[Any[arr[2][1], innAdd, arr[3:end]...] for innAdd in arr[2][2:end]]...]
        else
            newArr = Any[arr[1], arr[2:end]...]
        end
    else
        newArr = Any[arr[1], arr[2:end]...]
    end
    return newArr
end

function plotRes2d(baseExpr,fun, lb, ub, LBpoints, UBpoints, varBase,saveFlag=false)
    #Plot output
    plotly()
    fun_string = string(baseExpr)
    p = plot(range(lb, ub, length=100), fun, label="function", color="black")
    plot!(p, [p[1] for p in LBpoints], [p[2] for p in LBpoints],  color="purple", marker=:o, markersize=1, label="lower bound")
    plot!(p, [p[1] for p in UBpoints], [p[2] for p in UBpoints], color="blue", marker=:diamond, markersize=1,  label="upper bound", legend=:right, title=fun_string, xlabel=string(varBase))
    # display(p)
    if saveFlag
        global NPLOTS
        NPLOTS += 1
        savefig(p, "plots/bound_"*string(NPLOTS)*".html")
    end
end

function MinkSum(vec1, vec2, roundFlag=false)
    """
    Returns the Minkowski sum of vectors of tuples. The vectors can have arbitrary length but tuples must have the same length

    (since the sets must have same dimension)
    """
    #TODO: Ensure that dimensions of tuples are the same

    minkResult = Any[]
    for tup in vec1
        for tup2 in vec2
            if roundFlag
                push!(minkResult, Tuple(round.(tup[i] + tup2[i], digits=8) for i in eachindex(tup)))
            else
                push!(minkResult, Tuple(tup[i] + tup2[i] for i in eachindex(tup)))
            end



        end
    end

    return minkResult

end

function plotSurf(baseFunc, lbVec, ubVec, surfDim, xS, yS, saveFlag=false)
    """
    Method to plot the surface overapproximation of a function
    Args:
        baseFunc: The bi-variate function we wish to plot 
        xS: The x values over which the function is overapproximated
        yS: The y values over which the function is overapproximated
        lbVec: Lower bound of the overapproximation (vector of values)
        ubVec: Upper bound of the overapproximation (vector of values)
        surfDim: The dimensions of the surface
        saveFlag: Flag to save the plot as an html file or not

        TODO: Modified plot function. Fix dependencies

    """

    
    fun1 = Symbolics.build_function(baseFunc, find_variables(baseFunc)..., expression=Val{false})
    xC = collect(range(minimum(xS), maximum(xS), length=100))
    yC = collect(range(minimum(yS), maximum(yS), length=100))

    lbMat = reshape([p[3] for p in sort(lbVec)], surfDim)
    ubMat = reshape([p[3] for p in sort(ubVec)], surfDim)

    sPlot = plot(xC, yC, fun1, st=:surface, camera=(-30,30), color="black", label="function", showscale = false, opacity = 1.0)
    plot!(sPlot, [p[1] for p in xS], [p[1] for p in yS], lbMat, st=:mesh, color="orange", label="lower bound", showscale=false, opacity=1.0)
    # plot!(sPlot, [p[1] for p in xS], [p[1] for p in yS], ubMat, st=:surface, color="blue", label="upper bound", showscale=false,opacity=1.0)

    if saveFlag
        global NPLOTS
        NPLOTS += 1
        savefig(sPlot, "plots/bound_"*string(NPLOTS)*".html")
    end

end

function addDim(vec, dim, zeroVal = 0.0)
    """
    Add a dimension to each tuple in a vector of tuples. This is equivalent to lifting a n-d polytope to a dimention nd+1

    Equiv to a cartesian product with the zero vector in the new dimension
    """
    newVec = Any[]
    tupSize = max(length(vec[1]), dim)
    #Initialize the tuple vector with zeros 
    for tup in vec
        tupVec = tupVec = zeroVal*ones(tupSize+1)
        for i in 1:tupSize
            if i < dim && i < length(tup)
                tupVec[i] = tup[i]
            elseif i > dim && i < length(tup) +1
                tupVec[i] = tup[i-1]
            end
            #Set the final values of each tuple to be the same 
            tupVec[end] = tup[end]
        end
        newTup = tuple(tupVec...)
        push!(newVec, newTup)
    end
    return newVec
end

function boundMV1(expr, lb, ub)
    """
    Function to bound a multivariate function over a given interval.

    Assume addition-free chunks
    """

    #Collect like terms/reduce to univariate chunks 
    parsed1 = parse_and_reduce(expr)

    v2Func = parsed1[2]
    #Make expression into Julia function 
    v2f = Symbolics.build_function(v2Func, find_variables(v2Func)..., expression=Val{false})
    #Get approximation tuples over the interval
    v2UB, v2LB = bound_univariate(v2Func, lb, ub, plotflag = true) 

    #Repeat for second chunk 
    v3Func = parsed1[3]
    v3f = Symbolics.build_function(v3Func, find_variables(v3Func)..., expression=Val{false})
    v3UB, v3LB = bound_univariate(v3Func, lb, ub, plotflag = true)

    #Check bounds
    sum([v2f(tup[1]) < tup[2] for tup in v2LB])
    sum([v2f(tup[1]) > tup[2] for tup in v2UB])

    sum([v3f(tup[1]) < tup[2] for tup in v3LB])
    sum([v3f(tup[1]) > tup[2] for tup in v3UB])

    #For future use, interpolate to ensure UB and LB for each is over the same set of points 
    nv2LB, nv2UB = interpol(v2LB, v2UB)
    nv3LB, nv3UB = interpol(v3LB, v3UB)

    #Check bounds. As expected, interpolation does not break anything new 
    sum([v2f(tup[1]) < tup[2] for tup in nv2LB])
    sum([v2f(tup[1]) > tup[2] for tup in nv2UB])
    sum([v3f(tup[1]) < tup[2] for tup in nv3LB])
    sum([v3f(tup[1]) > tup[2] for tup in nv3UB])


    #Find lower bounds to shift each chunk 
    v2l = minimum([pt[2] for pt in nv2LB]) #extract lower bounds
    v3l = minimum([pt[2] for pt in nv3LB]) #extract lower bounds

    #Define log transformation of shifted overapprox (shift by 2*abs(lower bound) to ensure positivity)
    lv2LB = Any[(tup[1], log(tup[2] + 2*abs(v2l))) for tup in nv2LB]
    lv2UB = Any[(tup[1], log(tup[2] + 2*abs(v2l))) for tup in nv2UB]
    lv3LB = Any[(tup[1], log(tup[2] + 2*abs(v3l))) for tup in nv3LB]
    lv3UB = Any[(tup[1], log(tup[2] + 2*abs(v3l))) for tup in nv3UB]

    # lbXs = Any[tup[1] for tup in lv2LB]
    # ubXs = Any[tup[1] for tup in lv2UB]
    # lbYs = Any[tup[1] for tup in lv3LB]
    # ubYs = Any[tup[1] for tup in lv3UB]

    xS = Any[tup[1] for tup in lv2LB]
    yS = Any[tup[1] for tup in lv3LB]

    # #For each, plot to check that the transformation is valid
    # lv2Func = :(log(cos(x)*x + $(2*abs(v2l))))

    # lv2f = Symbolics.build_function(lv2Func, find_variables(lv2Func)..., expression=Val{false})
    # plotRes2d(lv2Func, lv2f, lb, ub, lv2LB, lv2UB, find_variables(lv2Func)..., false)

    # lv3Func = :(log(cos(y)*y^2 + $(2*abs(v3l))))
    # lv3f = Symbolics.build_function(lv3Func, find_variables(lv3Func)..., expression=Val{false})
    # plotRes2d(lv3Func, lv3f, lb, ub, lv3LB, lv3UB, find_variables(lv3Func)..., false)

    #Add dimension to each tuple to make Minkowski sum feasible 
    #For convenience, introduce new variables for these lifted bounds 

    #Add y axis to x-z overapprox
    lv2LBl = addDim(lv2LB, 2)
    lv2UBl = addDim(lv2UB, 2)

    #Add x axis to y-z overapprox
    lv3LBl = addDim(lv3LB, 1)
    lv3UBl = addDim(lv3UB, 1)



    #Compute Minkowski sum of these overapproximations 
    #So.. add lower bounds to lower bounds and upper bounds to upper bounds
    lv4LB = MinkSum(lv2LBl, lv3LBl)
    lv4UB = MinkSum(lv2UBl, lv3UBl)

    # #I claim that this minkowski sum is equiv to log(x) + log(y). Visualize to prove
    # #Plot the overapproximation 
    # #dims have form (y,x)
    surfDim = (size(yS)[1],size(xS)[1])
    # combFun = :(log(cos(x)*x + $(2*abs(v2l)))  + log(cos(y)*y^2 + $(2*abs(v3l))))
    # combF = Symbolics.build_function(combFun, find_variables(combFun)..., expression=Val{false})


    # #Check lower bound 
    # sum([combF(tup[1], tup[2]) < tup[3] for tup in lv4LB])

    # #Check upper bound 
    # sum([combF(tup[1], tup[2]) > tup[3] for tup in lv4UB])

    # plotSurf(combFun, lb, ub, lv4LB, lv4UB, surfDim, lbXs, lbYs, ubXs, ubYs, true)

    #Compute exponential of the sum of shifted logs 
    v4LB = Any[(tup[1:end-1]..., exp(tup[end])) for tup in lv4LB]
    v4UB = Any[(tup[1:end-1]..., exp(tup[end])) for tup in lv4UB]


    # #Again, check bounds 
    # combFun2 = :(exp(log(cos(x)*x + $(2*abs(v2l)))  + log(cos(y)*y^2 + $(2*abs(v3l)))))
    # combF2 = Symbolics.build_function(combFun2, find_variables(combFun2)..., expression=Val{false})

    # #Check lower bound
    # sum([combF2(tup[1], tup[2]) < tup[3] for tup in v4LB])

    # #Check upper bound
    # sum([combF2(tup[1], tup[2]) > tup[3] for tup in v4UB])

    # #Exp maintains overapproximation
    # plotSurf(combFun2, lb, ub, v4LB, v4UB, surfDim, lbXs, lbYs, ubXs, ubYs, true)

    v5LB = Any[]
    v5UB = Any[]

    for tup in v4LB
        xInd = findall(x->x[1] == tup[1], nv2LB)
        yInd = findall(y->y[1] == tup[2], nv3LB)

        push!(v5LB, (tup[1:end-1]..., tup[end] -2*abs(v3l)*nv2UB[xInd][1][2] -2*abs(v2l)*nv3UB[yInd][1][2]- 4*abs(v2l)*abs(v3l)))
    end

    for tup in v4UB
        xInd = findall(x->x[1] == tup[1], nv2UB)
        yInd = findall(y->y[1] == tup[2], nv3UB)

        push!(v5UB, (tup[1:end-1]..., tup[end] -2*abs(v3l)*nv2LB[xInd][1][2] -2*abs(v2l)*nv3LB[yInd][1][2]- 4*abs(v2l)*abs(v3l)))
    end


    # #Check bounds
    # combFun3 = :(cos(x)cos(y)x*y^2)
    # combF3 = Symbolics.build_function(combFun3, find_variables(combFun3)..., expression=Val{false})

    # #Check lower bound
    # sum([combF3(tup[1], tup[2]) < tup[3] for tup in v5LB])

    # println(minimum([combF3(tup[1], tup[2]) - tup[3] for tup in v5LB]))

    # #Check upper bound
    # sum([combF3(tup[1], tup[2]) > tup[3] for tup in v5UB])

    # println(maximum([combF3(tup[1], tup[2]) - tup[3] for tup in v5UB]))
    #Plot the overapproximation
    plotSurf(expr, v5LB, v5UB, surfDim, xS, yS, true)

    return v5LB, v5UB
end 

function boundMV2(expr, lb, ub)
    # #Reduce to addition chunks
    # baseParsed = parse_and_reduce(expr)

    # #Divvy up into chunks. Technically, if it's multiplication or division, we can just distribute and not worry about chunks
    # baseFunc1 = baseParsed[2]
    # baseFunc2 = baseParsed[3]

    #Collect like terms/reduce to univariate chunks 
    # parsed1 = parse_and_reduce(baseFunc1)

    v2Func = :(cos(x)*x)
    #Make expression into Julia function 
    v2f = Symbolics.build_function(v2Func, find_variables(v2Func)..., expression=Val{false})
    #Get approximation tuples over the interval
    v2UB, v2LB = interpol(bound_univariate(v2Func, lb, ub, ϵ=1e-3, npoint = 1, plotflag = true)...) 

    #Repeat for second chunk 
    v3Func = :(cos(y)*y^2)
    v3f = Symbolics.build_function(v3Func, find_variables(v3Func)..., expression=Val{false})
    v3UB, v3LB = interpol(bound_univariate(v3Func, lb, ub, ϵ=1e-3, npoint=1, plotflag = true)...)

    # #Check bounds
    # sum([v2f(tup[1]) < tup[2] for tup in v2LB])
    # sum([v2f(tup[1]) > tup[2] for tup in v2UB])

    # sum([v3f(tup[1]) < tup[2] for tup in v3LB])
    # sum([v3f(tup[1]) > tup[2] for tup in v3UB])

    # #For future use, interpolate to ensure UB and LB for each is over the same set of points 
    # nv2LB, nv2UB = v2LB, v2UB
    # nv3LB, nv3UB = v3LB, v3UB

    # #Check bounds. As expected, interpolation does not break anything new 
    # sum([v2f(tup[1]) < tup[2] for tup in nv2LB])
    # sum([v2f(tup[1]) > tup[2] for tup in nv2UB])
    # sum([v3f(tup[1]) < tup[2] for tup in nv3LB])
    # sum([v3f(tup[1]) > tup[2] for tup in nv3UB])


    #Find lower bounds to shift each chunk 
    v2l = minimum([pt[2] for pt in nv2LB]) #extract lower bounds
    v3l = minimum([pt[2] for pt in nv3LB]) #extract lower bounds

    #Define log transformation of shifted overapprox (shift by 2*abs(lower bound) to ensure positivity)
    lv2LB = Any[(tup[1], log(tup[2] + 2*abs(v2l))) for tup in nv2LB]
    lv2UB = Any[(tup[1], log(tup[2] + 2*abs(v2l))) for tup in nv2UB]
    lv3LB = Any[(tup[1], log(tup[2] + 2*abs(v3l))) for tup in nv3LB]
    lv3UB = Any[(tup[1], log(tup[2] + 2*abs(v3l))) for tup in nv3UB]

    lbXs = Any[tup[1] for tup in lv2LB]
    ubXs = Any[tup[1] for tup in lv2UB]
    lbYs = Any[tup[1] for tup in lv3LB]
    ubYs = Any[tup[1] for tup in lv3UB]

    #For each, plot to check that the transformation is valid
    lv2Func = :(log(cos(x)*x + $(2*abs(v2l))))

    lv2f = Symbolics.build_function(lv2Func, find_variables(lv2Func)..., expression=Val{false})
    plotRes2d(lv2Func, lv2f, lb, ub, lv2LB, lv2UB, find_variables(lv2Func)..., false)

    lv3Func = :(log(cos(y)*y^2 + $(2*abs(v3l))))
    lv3f = Symbolics.build_function(lv3Func, find_variables(lv3Func)..., expression=Val{false})
    plotRes2d(lv3Func, lv3f, lb, ub, lv3LB, lv3UB, find_variables(lv3Func)..., false)

    #Add dimension to each tuple to make Minkowski sum feasible 
    #For convenience, introduce new variables for these lifted bounds 

    #Add y axis to x-z overapprox
    lv2LBl = addDim(lv2LB, 2)
    lv2UBl = addDim(lv2UB, 2)

    #Add x axis to y-z overapprox
    lv3LBl = addDim(lv3LB, 1)
    lv3UBl = addDim(lv3UB, 1)



    #Compute Minkowski sum of these overapproximations 
    #So.. add lower bounds to lower bounds and upper bounds to upper bounds
    lv4LB = MinkSum(lv2LBl, lv3LBl)
    lv4UB = MinkSum(lv2UBl, lv3UBl)

    #I claim that this minkowski sum is equiv to log(x) + log(y). Visualize to prove
    #Plot the overapproximation 
    #dims have form (y,x)
    surfDim = (size(lbYs)[1],size(lbXs)[1])
    combFun = :(log(cos(x)*x + $(2*abs(v2l)))  + log(cos(y)*y^2 + $(2*abs(v3l))))
    combF = Symbolics.build_function(combFun, find_variables(combFun)..., expression=Val{false})


    #Check lower bound 
    sum([combF(tup[1], tup[2]) < tup[3] for tup in lv4LB])

    #Check upper bound 
    sum([combF(tup[1], tup[2]) > tup[3] for tup in lv4UB])

    # plotSurf(combFun, lb, ub, lv4LB, lv4UB, surfDim, lbXs, lbYs, ubXs, ubYs, true)



    #Compute exponential of the sum of shifted logs 
    v4LB = Any[(tup[1:end-1]..., exp(tup[end])) for tup in lv4LB]
    v4UB = Any[(tup[1:end-1]..., exp(tup[end])) for tup in lv4UB]



    #Again, check bounds 
    combFun2 = :(exp(log(cos(x)*x + $(2*abs(v2l)))  + log(cos(y)*y^2 + $(2*abs(v3l)))))
    combF2 = Symbolics.build_function(combFun2, find_variables(combFun2)..., expression=Val{false})

    #Check lower bound
    sum([combF2(tup[1], tup[2]) < tup[3] for tup in v4LB])

    #Check upper bound
    sum([combF2(tup[1], tup[2]) > tup[3] for tup in v4UB])

    #Exp maintains overapproximation
    # plotSurf(combFun2, lb, ub, v4LB, v4UB, surfDim, lbXs, lbYs, ubXs, ubYs, true)

    #Shift to account for overapprox
    v5LB = Any[(tup[1:end-1]..., tup[end] -2*abs(v3l)*v2f(tup[1]) -2*abs(v2l)*v3f(tup[2])- 4*abs(v2l)*abs(v3l)) for tup in v4LB]
    v5UB = Any[(tup[1:end-1]..., tup[end] -2*abs(v3l)*v2f(tup[1]) -2*abs(v2l)*v3f(tup[2])- 4*abs(v2l)*abs(v3l)) for tup in v4UB]


    #Check bounds
    combFun3 = :(cos(x)*x*cos(y)*y^2)
    combF3 = Symbolics.build_function(combFun3, find_variables(combFun3)..., expression=Val{false})

    #Check lower bound
    sum([combF3(tup[1], tup[2]) < tup[3] for tup in v5LB])

    println(minimum([combF3(tup[1], tup[2]) - tup[3] for tup in v5LB]))

    #Check upper bound
    sum([combF3(tup[1], tup[2]) > tup[3] for tup in v5UB])

    println(maximum([combF3(tup[1], tup[2]) - tup[3] for tup in v5UB]))
    #Plot the overapproximation
    plotSurf(combFun3, v5LB, v5UB, surfDim, lbXs, lbYs,true)

    return v5LB, v5UB

end

function bound_multivariate(expr, lb, ub; sound_sub=true)
    """
    Ideally, this function can take an arbitrary multivariate function and compute bouds over a given interval. However, achieving this is difficult. 

    Continue to update this function as we find more cases that it can handle
    """
    #Reduce to addition chunks
    baseParsed = parse_and_reduce(expr)

    #We know that each LB, UB combination is over the same set of points. So we don't have to perform 4 interpolations
    LB1, UB1 = boundMV1(baseParsed[2], lb, ub)
    LB2, UB2 = boundMV1(baseParsed[3], lb, ub)

    #This interpolation is not sound. Fix now 
    nLB1, nLB2 = interpol(LB1, LB2)
    nUB1, nUB2 = interpol(UB1, UB2)

    #Interpolation results are unsorted.
    nLB1 = sort(nLB1)
    nUB1 = sort(nUB1)
    LBVec = [collect(tup) for tup in nLB1]
    UBVec = [collect(tup) for tup in nUB1]
    LBMat = copy(reduce(hcat,LBVec)')
    UBMat = copy(reduce(hcat,UBVec)')
    #Extract shape of surface
    #reverse
    LBMatR = reverse(LBMat, dims=2)
    #Surfdim expects the form (y,x). This is the source of the bug
    surfDim = Tuple(length(unique(col)) for col in eachcol(LBMatR[:,2:end]))

    print(surfDim)
    xS, yS = [unique(col) for col in eachcol(LBMat[:,1:end-1])] #xS and yS are sorted

    plotSurf(baseParsed[2], lb, ub, nLB1, nUB1, surfDim, xS, yS, xS, yS, true)

    nLB2 = sort(nLB2)
    nUB2 = sort(nUB2)
    plotSurf(baseParsed[3], lb, ub, nLB2, nUB2, surfDim, xS, yS, xS, yS, true)
    #Add separate chunks 

    LB3, UB3 = sound_IA(nLB1, nUB1, nLB2, nUB2, baseParsed[1])

    return LB3, UB3
end

function gen_interpol(oA)
    """
    Method to generate an interpolating function for an overt approximation
    """
    #First, convert inputs to a vector of vectors
    vecVecs = [collect(tup[1:end-1]) for tup in oA]
    #Convert to a matrix
    vecMat = copy(reduce(hcat,vecVecs)')
    #Convert matrix to a tuple of unique vectors
    tupVecs = Tuple(unique(col[:]) for col in eachcol(vecMat))

    #For the output, we need to basically convert the tuples to a surface
    #First, convert to a matrix
    outs = [collect(tup) for tup in oA]
    outMat = copy(reduce(hcat,outs)')
    #Extract shape of surface
    outMatR = reverse(outMat, dims=2)
    surfDim = Tuple(length(unique(col)) for col in eachcol(outMatR[:,2:end]))
    #  surfDim = Tuple(length(unique(col)) for col in eachcol(outMat[:,1:end-1]))
    Amat = reshape([tup[end] for tup in oA], surfDim)
    AmatT = copy(Amat')

    #Use extrapolation boundary condition.
    #Note that we are not exactly extrapolating, but we run into floating point soundness issues with irrational numbers, and so we allow for extrapolation to avoid this. We would never interpolation outside of the bounds anyway
    #WARNING: This may not be sound. We need to find a better way to handle this
        if size(outMat)[2] > 2
        interp = linear_interpolation(tupVecs, AmatT, extrapolation_bc = Interpolations.Line())
    else
        interp = linear_interpolation(tupVecs, Amat, extrapolation_bc = Interpolations.Line())

    end
    return interp    

end

function interpol(oA1, oA2, digits=5)
    """
    When it is not catching international criminals, this method interpolates two overt approximations to ensure that they are over the same set of points
    """
    #Create a new array of inputs that are the same for both approximations
    #TODO Consider using rounding here to ensure that the points are the same
    #Unsound flag, this is removing symmetric points. Basically, 
    oAComb = sort(vcat(oA1, oA2))
    #NOTE: This is where the precision of the input domain is set. 
    oAVecs = [round.(collect(tup[1:end-1]), digits=digits) for tup in oAComb]
    oAMat = copy(reduce(hcat,oAVecs)')
    oAMatR = reverse(oAMat, dims=2)
    oACol = [unique(col[:]) for col in eachcol(oAMat)]
    newInps = Iterators.product(oACol...)


    #Generate interpolation function for each approximation
    #NOTE: Sort overapproximation to ensure that the interpolation is sound
    interp1 = gen_interpol(sort(oA1))
    interp2 = gen_interpol(sort(oA2))

    #Loop through both approximations and interpolate to ensure that the bounds are over the same set of points (and that the points are evenly spaced)

    newOA1 = Any[]
    newOA2 = Any[]
    for tup in newInps
        tup1 = (tup..., interp1(tup...))
        tup2 = (tup..., interp2(tup...))
        push!(newOA1, tup1)
        push!(newOA2, tup2)
    end

    return sort(newOA1), sort(newOA2)

end

function sound_IA(LB1, UB1, LB2, UB2, op)
    """
    Method to soundly add or subtract two overapproximations
    """
    LB = Any[]
    UB = Any[]
    if op == :+
        for i = 1:size(LB1)[1]
            push!(LB, (LB1[i][1:end-1]..., LB1[i][end] + LB2[i][end]))
            push!(UB, (UB1[i][1:end-1]..., UB1[i][end] + UB2[i][end]))
        end
    elseif op == :-
        for i = 1:size(LB1)[1]
            push!(LB, (LB1[i][1:end-1]..., LB1[i][end] - UB2[i][end]))
            push!(UB, (UB1[i][1:end-1]..., UB1[i][end] - LB2[i][end]))
        end
    end
    return LB, UB
end

function inpShiftLog(lb,ub;bounds=nothing)
    """
    Method to shift the input domain of a function to ensure that the log transformation is sound

    returns: amount by which to shift f(x) to ensure that log(f(x)) is defined
    """
    shiftVal = 0
    if lb < 0 
        shiftVal = 4*abs(lb)
    end
    if !isnothing(bounds)
        boundVals = [tup[end] for tup in bounds]
        lowestBound = minimum(boundVals)
        if lowestBound < 0
            shiftVal = max(shiftVal, 2*abs(lowestBound))
        end
    end
    return shiftVal
end

function sample_hyperrectangle(h)
    """
    Method to sample a random point from a hyperrectangle

    args:
    h: Hyperrectangle object
    """
    x = zeros(dim(h))
    lbs = low(h)
    ubs = high(h)
    for i = 1:dim(h)
        x[i] = lbs[i] + rand()*(ubs[i] - lbs[i])
    end
    return x
end


function simulateTraj(query, ntraj)
    """
    Method to simulate random Trajectories for a given dynamical system

    args:
    query: OvertPQuery object
    ntraj: Number of trajectories to simulate
    """
    #Array to hold the trajectories 
    traj = []
    set = query.problem.domain
    totalSteps = query.ntime
    dt = query.dt
    for i = 1:ntraj
        #Sample a random point from the domain
        x0 = sample_hyperrectangle(set)
        for j = 1:totalSteps
            xNew = xNew = query.problem.dynamics(x0,dt)
            x0 = xNew
        end
        push!(traj, x0)
    end
    return traj  
end

function plotBounds(boundSets, expr)
    LBs, UBs = boundSets
    xS = unique([tup[1] for tup in LBs])
    yS = unique([tup[2] for tup in LBs])
    surfDim = (size(yS)[1], size(xS)[1])

    plotSurf(expr, sort(LBs), sort(UBs), surfDim, xS, yS, true)
end

function compute_coraSet(minFile, maxFile, time_ind)
    coraMins = CSV.File(minFile)
    coraMaxs = CSV.File(maxFile)

    coraLows = [coraMins[time_ind]...]
    coraHighs = [coraMaxs[time_ind]...]

    coraSet = Hyperrectangle(low=coraLows, high=coraHighs)
    return coraSet
end

function lift_OA(emptyList, currList, boundLB, boundUB, lbs, ubs, zeroVal=0.0)
    """
    Method to lift an overapproximation to a higher dimension
    by adding (constant) bounds for unused variables into the overapproximation 

    emptyList: List of unused variables. Modified in place. Must be sorted
    currList: List of used variables. Modified in place
    boundLB: Lower bounds for the overapproximation
    boundUB: Upper bounds for the overapproximation
    lbs, ubs: Lower and upper bounds for the domain of the specific function 

    returns: lifted bounds
    """
    # lbs, ubs = extrema(query.domain)
    boundLB_l = sort!(copy(boundLB))
    boundUB_l = sort!(copy(boundUB))
    plotFlag = false
    for i in copy(emptyList)
        #Find bounds for unused variable
        lbEmptyVar = lbs[i]
        ubEmptyVar = ubs[i]
        #Constant function to bound
        emptyEq = :(0*x)
        emptyLB, emptyUB = bound_univariate(emptyEq, lbEmptyVar, ubEmptyVar, plotflag = plotFlag)
        #Add corresponding dimension to the lifted bounds
        boundLB_l = addDim(boundLB_l, i, zeroVal)
        boundUB_l = addDim(boundUB_l, i, zeroVal)

        emptyLB_l = copy(emptyLB)
        emptyUB_l = copy(emptyUB)
        
        for idx in currList 
            emptyLB_l = addDim(emptyLB_l, idx, zeroVal)
            emptyUB_l = addDim(emptyUB_l, idx, zeroVal)
        end
        #Add the empty variable to the bounds 
        boundLB_l = unique(MinkSum(boundLB_l, emptyLB_l))
        boundUB_l = unique(MinkSum(boundUB_l, emptyUB_l))

        push!(currList, i)
        popfirst!(emptyList)
        #Sorting in place because we need to add indices in order
        sort!(currList)
    end

    return sort(boundLB_l), sort(boundUB_l)
end

function gen_interpol_nd(boundSet)
    """
    Method that takes an arbitrary dimensional bound set and returns an interpolator
    """
    #To define A, we need to define the nodes along each dimension 
    nDim = length(boundSet[1]) - 1 #Number of dimensions
    #Define the nodes along each dimension
    nodes = [unique([x[i] for x in boundSet]) for i in 1:nDim]
    #Convert to a tuple for compatibility with Interpolations
    nodesTup = Tuple(node for node in nodes)
    
    Ax_dims = [length(x) for x in nodes]

    #Define A
    A = zeros(Ax_dims...)
    for boundVal in boundSet
        node = boundVal[1:end-1]
        ind = [findall(x->x==node[i], nodes[i])[1] for i in 1:nDim]
        A[ind...] = boundVal[end]
    end
    
    itp = Interpolations.interpolate(nodesTup, A, Gridded(Linear()))
    return itp
end

function interpol_nd(boundSet_1, boundSet_2)
    bound1Inps = [x[1:end-1] for x in boundSet_1]
    bound2Inps = [x[1:end-1] for x in boundSet_2]

    sort!(bound1Inps)
    sort!(bound2Inps)
    
    #Find the union of the inputs
    unionInps = unique(vcat(bound1Inps, bound2Inps), dims=1)
    sort!(unionInps)
    interp1 = gen_interpol_nd(boundSet_1)
    interp2 = gen_interpol_nd(boundSet_2)
    
    newbounds1 = []
    newbounds2 = []
    #Interpolate the bounds
    for inp in unionInps
        push!(newbounds1, (inp..., interp1(inp...)))
        push!(newbounds2, (inp..., interp2(inp...)))
    end
    return sort(newbounds1), sort(newbounds2)
end

function validBounds(fexpr, vars, LBs, UBs, verbose = false)
    symVarVec = [Symbolics.variable(var) for var in vars]

    symTestFunc = Symbolics.build_function(fexpr, symVarVec..., expression=false)
    
    inps = [tup[1:end-1] for tup in UBs]
    trueFlag = true
    i = 
    for i = 1:length(inps)
        trueFlag = trueFlag && (symTestFunc(inps[i]...) >= LBs[i][end] && symTestFunc(inps[i]...) <= UBs[i][end])

        if !trueFlag && verbose
            println("Failed at index $i")
            println("Input: ", inps[i])
            println("True value: ", symTestFunc(inps[i]...))
            println("Computed LB: ", LBs[i][end])
            println("Computed UB: ", UBs[i][end])

            if symTestFunc(inps[i]...) <= LBs[i][end]
                println("Invalid lower bound")
            end
            if symTestFunc(inps[i]...) >= UBs[i][end]
                println("Invalid upper bound")
            end
            break
        end
    end
    return trueFlag
end

function ceil_n(x, n = 64)
    """
    Method to round a float towards infinity with a specified number of sig figs
    """
    return ceil(x, digits = n)
end

function floor_n(x, n = 64)
    """
    Method to round a float towards negative infinity with a specified number of sig figs
    """
    return floor(x, digits = n)
end

function divBounds(lb1, ub1, lb2, ub2)
    # """
    # Method to divide bound 1 by bound 2.
    # Assumes that 0 is not in the interval of bound 2
    # """
    try
        @assert minimum([tup[end] for tup in lb2]) > 0 || maximum([tup[end] for tup in ub2]) < 0
    catch
        println("Division by zero")
    end

    #Find the union of the inputs 
    #NOTE: Assume lb and ub have the same inputs 

    bound1Inps = [tup[1:end-1] for tup in lb1]
    bound2Inps = [tup[1:end-1] for tup in lb2]

    #Find the union of the inputs
    unionInps = sort(unique(vcat(bound1Inps, bound2Inps), dims=1))

    #Interpolate the bounds to ensure they are defined on the same input set
    lb1_i, lb2_i = interpol_nd(lb1, lb2)
    ub1_i, ub2_i = interpol_nd(ub1, ub2)

    #Define the output bounds
    divLB = []
    divUB = []

    for inp in unionInps 
        ind1 = findall(x->x[1:end-1] == inp, lb1_i)[1]
        lb1 = lb1_i[ind1][end]
        ub1 = ub1_i[ind1][end]

        ind2 = findall(x->x[1:end-1] == inp, lb2_i)[1]
        lb2 = lb2_i[ind2][end]
        ub2 = ub2_i[ind2][end]

        #Compute the division, use interval arithmetic, assumes 0 is not in interval 2
        lb = lb1/ub2
        ub = ub1/lb2

        push!(divLB, (inp..., lb))
        push!(divUB, (inp..., ub))

    end
    return divLB, divUB
end

function get_sincos_regions(a,b; offset=0)
    """
    Return inflection points for sin, (cos with offset of π/2)

    Lifted from OVERT
    """

    n̂_a = ceil((a - offset) / π)
    n̂_b = floor((b - offset) / π)
    n_array = [i for i in n̂_a:n̂_b]

    if length(n_array) == 0 # no inflection points within interval
        @assert sin(a + offset) != 0.0 
        return n_array, sin(a + offset) < 0 # true if convex, false implies concave
    else # greater than 0 length
        return [offset + n*π for n in n_array], nothing # nothing denotes mixed convexity
    end
end

function get_tan_regions(a,b)
    """
    Lifted from OVERT
    """
    # assert tan doesn't span discontinuity
    @assert (b-a) <= π
    # background to these cases:
    # tan is continuous for +/- π/2 centered around even multiples of π/2. e.g. -π/2 ≤ 0*π/2 ≤ π/2 and again for π/2 ≤ 2*π/2 ≤ 3π/2
    case1 = (ceil(a / (π/2)) == floor(b / (π/2))) && ((ceil(a/ (π/2)) % 2) == 0) # for when interval spans inflection point
    case2 = (floor(a / (π/2)) == floor(b / (π/2))) && ((floor(a/ (π/2)) % 2) == 0) # for when interval is to the right of inflection point
    case3 = (ceil(a / (π/2)) == ceil(b / (π/2))) && ((ceil(a/ (π/2)) % 2) == 0) # for when interval is to the left of inflection point
    @assert (case1 || case2 || case3)
    sin_zeros, sin_convexity = get_sincos_regions(a,b)
    cos_zeros, cos_convexity = get_sincos_regions(a,b, offset=π/2)
    tan_zeros = sort(vcat(sin_zeros, cos_zeros))
    if length(tan_zeros) == 0 # no inflection points within interval
        # if sin and cos convexity diff -> tan is concave (not convex)   (true, false) -> false   or   (false, true) -> false
        # # if sin and cos convexity same -> tan is convex   (true, true) -> true or (false, false) -> true
        return tan_zeros, !xor(sin_convexity, cos_convexity)
    else 
        return tan_zeros, nothing # nothing implies mixed convexity 
    end
end

function get_subgrids(unionInps)
    """
    Given a vector of inputs, return the subgrids that are used to define the input space. This is needed to achieve gridded interpolation.
    """
    tupList = unionInps
    if isempty(tupList)
        throw(ArgumentError("Empty list of tuples"))
    end

    dim = length(tupList[1])

    #Identify consituent grid elements 
    sub_grids = [unique(tup[i] for tup in tupList) for i in 1:dim]
    return sub_grids
end
function prodBounds(lb1, ub1, lb2, ub2)
    """
    Method to compute bounds for the product of two functions defined over the same domain

    NOTE: Does not check if the inputs are defined over the same domain. Use carefully
    """
    #Vector for outputs
    prodLB = []
    prodUB = []

    bound1Inps = [tup[1:end-1] for tup in lb1]
    bound2Inps = [tup[1:end-1] for tup in lb2]

    #Let's do gridded interpolation 
    vcat(bound1Inps, bound2Inps)

    #Find the union of the inputs
    #NOTE: Assumes lbs and ubs have same inputs
    unionInps = sort(unique(vcat(bound1Inps, bound2Inps), dims=1))

    #Find unique elements per dimension
    subgrids = get_subgrids(unionInps)

    #Gridded interpolators for each function
    lb1_int = gen_interpol_nd(lb1)
    ub1_int = gen_interpol_nd(ub1)
    lb2_int = gen_interpol_nd(lb2)
    ub2_int = gen_interpol_nd(ub2)

    #Now we're going to hack grid iterations 
    #Number of grid points per dimension
    subgrid_sizes = [length(subgrid) for subgrid in subgrids]
    #Number of 1-faces per dimension (#grid points - 1)
    subgrid_faces = [tuple(collect(1:length(subgrid)-1)...) for subgrid in subgrids]

    #Define array to hold the grid points
    gridMat = collect(Iterators.product(subgrids...))

    #Define arrays to hold upper and lower bounds
    lowMat = [(tup..., Inf) for tup in gridMat]
    ulowMat = [(tup..., Inf) for tup in gridMat]
    highMat = [(tup..., -Inf) for tup in gridMat]
    uhighMat = [(tup..., -Inf) for tup in gridMat]

    #Iterate across grid cells 
    for cell in Iterators.product(subgrid_faces...)
        #Get vertex indices associated with each face of the cell
        vertices = [(i, i+1) for i in cell]
        #Convert these into grid point indices 
        verts = collect(Iterators.product(vertices...))

        #Get the sub-grids associated with each face of the cell
        coordinates = [(subgrids[sg][i], subgrids[sg][i+1]) for (sg, i) in enumerate(cell)]
        #Use the sub-grids to get grid coordinates 
        coords = collect(Iterators.product(coordinates...))

        #Use pointwise interval arithmetic to get upper and lower bounds
        LBs = [min(lb1_int(coord...)*lb2_int(coord...), lb1_int(coord...)*ub2_int(coord...), ub1_int(coord...)*lb2_int(coord...), ub1_int(coord...)*ub2_int(coord...)) for coord in coords]
        UBs = [max(lb1_int(coord...)*lb2_int(coord...), lb1_int(coord...)*ub2_int(coord...), ub1_int(coord...)*lb2_int(coord...), ub1_int(coord...)*ub2_int(coord...)) for coord in coords]

        #Now for soundness, use cell-wise interval arithmetic bounds
        #TODO: Optimization here for tighter bounds 
        minMin = minimum(LBs)
        maxMax = maximum(UBs)

        uLBs = deepcopy(LBs)
        uUBs = deepcopy(UBs)

        LBs .= minMin
        UBs .= maxMax


        #To avoid flat bounds, incorporate (sound) random perturbation
        #Maximum addition is 1% of bound value 
        # LBs = LBs .- 0.2*rand(size(LBs)...).*abs.(LBs)
        # UBs = UBs .+ 0.2*rand(size(UBs)...).*abs.(UBs)

        #To avoid flat bounds, incorporate (sound) proportional perturbation
        # LBs = LBs .- κ*cLBs
        # UBs = UBs .+ κ*cUBs

        #Update the matrix of bounds
        for (i, vert) in enumerate(verts)
            pt = lowMat[vert...][1:end-1] 
            #Update the bounds only if the new bound is looser than the current
            lowMat[vert...]= (pt...,min(lowMat[vert...][end], LBs[i]))
            ulowMat[vert...]= (pt...,min(ulowMat[vert...][end], uLBs[i]))
            highMat[vert...] = (pt...,max(highMat[vert...][end], UBs[i]))
            uhighMat[vert...] = (pt...,max(uhighMat[vert...][end], uUBs[i]))
        end
    end

    #Return a list of bounds
    maxMin = maximum([tup[end] for tup in ulowMat])
    minMax = minimum([tup[end] for tup in uhighMat])

    #Shift bounds to try to match the "shape" of unsound bounds 
    sULBs = maxMin .- [tup[end] for tup in ulowMat]
    sUBs = minMax .- [tup[end] for tup in uhighMat]

    prodLB = [lowMat...]
    prodUB = [highMat...]
    for (i, tup) in enumerate(lowMat)
        prodLB[i] = (prodLB[i][1:end-1]...,prodLB[i][end] - abs(sULBs[i]))
        prodUB[i] = (prodUB[i][1:end-1]...,prodUB[i][end] + abs(sUBs[i]))
    end
    # prodLB = lowMat .+ abs.(sULBs)
    # prodUB = highMat .+ abs.(sUBs)
    # prodLB = [lowMat...]
    # prodUB = [highMat...]
    return prodLB, prodUB
end

function prodBounds2(lb1, ub1, lb2, ub2)
    """
    Method to compute bounds for the product of two functions defined over the same domain

    NOTE: Does not check if the inputs are defined over the same domain. Use carefully
    """
    #Vector for outputs
    prodLB = []
    prodUB = []

    bound1Inps = [tup[1:end-1] for tup in lb1]
    bound2Inps = [tup[1:end-1] for tup in lb2]

    #Let's do gridded interpolation 
    vcat(bound1Inps, bound2Inps)

    #Find the union of the inputs
    #NOTE: Assumes lbs and ubs have same inputs
    unionInps = sort(unique(vcat(bound1Inps, bound2Inps), dims=1))

    #Find unique elements per dimension
    subgrids = get_subgrids(unionInps)

    #Gridded interpolators for each function
    lb1_int = gen_interpol_nd(lb1)
    ub1_int = gen_interpol_nd(ub1)
    lb2_int = gen_interpol_nd(lb2)
    ub2_int = gen_interpol_nd(ub2)

    #Now we're going to hack grid iterations 
    #Number of grid points per dimension
    subgrid_sizes = [length(subgrid) for subgrid in subgrids]
    #Number of 1-faces per dimension (#grid points - 1)
    subgrid_faces = [tuple(collect(1:length(subgrid)-1)...) for subgrid in subgrids]

    #Define array to hold the grid points
    gridMat = collect(Iterators.product(subgrids...))

    #Define arrays to hold upper and lower bounds
    lowMat = [(tup..., Inf) for tup in gridMat]
    highMat = [(tup..., -Inf) for tup in gridMat]

    #Iterate across grid cells 
    for cell in Iterators.product(subgrid_faces...)
        #Get vertex indices associated with each face of the cell
        vertices = [(i, i+1) for i in cell]
        #Convert these into grid point indices 
        verts = collect(Iterators.product(vertices...))

        #Get the sub-grids associated with each face of the cell
        coordinates = [(subgrids[sg][i], subgrids[sg][i+1]) for (sg, i) in enumerate(cell)]
        #Use the sub-grids to get grid coordinates 
        coords = collect(Iterators.product(coordinates...))

        #Use pointwise interval arithmetic to get upper and lower bounds
        LBs = [min(lb1_int(coord...)*lb2_int(coord...), lb1_int(coord...)*ub2_int(coord...), ub1_int(coord...)*lb2_int(coord...), ub1_int(coord...)*ub2_int(coord...)) for coord in coords]
        UBs = [max(lb1_int(coord...)*lb2_int(coord...), lb1_int(coord...)*ub2_int(coord...), ub1_int(coord...)*lb2_int(coord...), ub1_int(coord...)*ub2_int(coord...)) for coord in coords]

        #Now for soundness, use cell-wise interval arithmetic bounds
        #TODO: Optimization here for tighter bounds 
        # LBs .= minimum(LBs)
        # UBs .= maximum(UBs)

        #Update the matrix of bounds
        for (i, vert) in enumerate(verts)
            pt = lowMat[vert...][1:end-1] 
            #Update the bounds only if the new bound is looser than the current
            lowMat[vert...]= (pt...,min(lowMat[vert...][end], LBs[i]))
            highMat[vert...] = (pt...,max(highMat[vert...][end], UBs[i]))
        end
    end

    #Return a list of bounds
    prodLB = [lowMat...]
    prodUB = [highMat...]
    return prodLB, prodUB
end

function sumBounds(bounds1LB, bounds1UB, bounds2LB, bounds2UB, diffFlag)
    """
    Method to compute bounds for the sum or difference of two functions defined over the same domain

    NOTE: Does not check if the inputs are defined over the same domain. Use carefully
    """
    #Vector for outputs 
    sumLB = []
    sumUB = []

    bound1Inps = [tup[1:end-1] for tup in bounds1LB]
    bound2Inps = [tup[1:end-1] for tup in bounds2LB]

    #Let's do gridded interpolation 
    vcat(bound1Inps, bound2Inps)

    #Find the union of the inputs
    #NOTE: Assumes lbs and ubs have same inputs
    unionInps = sort(unique(vcat(bound1Inps, bound2Inps), dims=1))

    #Find unique elements per dimension
    subgrids = get_subgrids(unionInps)

    #Gridded interpolators for each function
    lb1_int = gen_interpol_nd(bounds1LB)
    ub1_int = gen_interpol_nd(bounds1UB)
    lb2_int = gen_interpol_nd(bounds2LB)
    ub2_int = gen_interpol_nd(bounds2UB)

    #Now we're going to hack grid iterations 
    #Number of grid points per dimension
    subgrid_sizes = [length(subgrid) for subgrid in subgrids]
    #Number of 1-faces per dimension (#grid points - 1)
    subgrid_faces = [tuple(collect(1:length(subgrid)-1)...) for subgrid in subgrids]

    #Define array to hold the grid points
    gridMat = collect(Iterators.product(subgrids...))

    #Define arrays to hold upper and lower bounds
    lowMat = [(tup..., Inf) for tup in gridMat]
    highMat = [(tup..., -Inf) for tup in gridMat]

    #Iterate across grid cells 
    for cell in Iterators.product(subgrid_faces...)
        #Get vertex indices associated with each face of the cell
        vertices = [(i, i+1) for i in cell]
        #Convert these into grid point indices 
        verts = collect(Iterators.product(vertices...))

        #Get the sub-grids associated with each face of the cell
        coordinates = [(subgrids[sg][i], subgrids[sg][i+1]) for (sg, i) in enumerate(cell)]
        #Use the sub-grids to get grid coordinates 
        coords = collect(Iterators.product(coordinates...))

        #Use pointwise interval arithmetic to get upper and lower bounds
        if diffFlag
            LBs = [lb1_int(coord...) - ub2_int(coord...) for coord in coords]
            UBs = [ub1_int(coord...) - lb2_int(coord...) for coord in coords]
        else
            LBs = [lb1_int(coord...) + lb2_int(coord...) for coord in coords]
            UBs = [ub1_int(coord...) + ub2_int(coord...) for coord in coords]
        end

        #Update the matrix of bounds
        for (i, vert) in enumerate(verts)
            pt = lowMat[vert...][1:end-1] 
            #Update the bounds only if the new bound is looser than the current
            lowMat[vert...]= (pt...,min(lowMat[vert...][end], LBs[i]))
            highMat[vert...] = (pt...,max(highMat[vert...][end], UBs[i]))
        end
    end
    #Convert the matrices into a list of tuples
    sumLB = [lowMat...]
    sumUB = [highMat...]

    return sumLB, sumUB
end

function divBounds(lb1, ub1, lb2, ub2)
    """
    Method to compute bounds for the quotient of two functions defined over the same domain

    NOTE: Does not check if the inputs are defined over the same domain. Use carefully
    """
    # """
    # Method to divide bound 1 by bound 2.
    # Assumes that 0 is not in the interval of bound 2
    # """
    try
        @assert minimum([tup[end] for tup in lb2]) > 0 || maximum([tup[end] for tup in ub2]) < 0
    catch
        println("Division by zero")
    end

    #Find the union of the inputs 
    #NOTE: Assume lb and ub have the same inputs 
    #Define the output bounds
    divLB = []
    divUB = []

    bound1Inps = [tup[1:end-1] for tup in lb1]
    bound2Inps = [tup[1:end-1] for tup in lb2]

    #Let's do gridded interpolation 
    vcat(bound1Inps, bound2Inps)

    #Find the union of the inputs
    #NOTE: Assumes lbs and ubs have same inputs
    unionInps = sort(unique(vcat(bound1Inps, bound2Inps), dims=1))

    #Find unique elements per dimension
    subgrids = get_subgrids(unionInps)

    #Gridded interpolators for each function
    lb1_int = gen_interpol_nd(lb1)
    ub1_int = gen_interpol_nd(ub1)
    lb2_int = gen_interpol_nd(lb2)
    ub2_int = gen_interpol_nd(ub2)

    #Now we're going to hack grid iterations 
    #Number of grid points per dimension
    subgrid_sizes = [length(subgrid) for subgrid in subgrids]
    #Number of 1-faces per dimension (#grid points - 1)
    subgrid_faces = [tuple(collect(1:length(subgrid)-1)...) for subgrid in subgrids]

    #Define array to hold the grid points
    gridMat = collect(Iterators.product(subgrids...))

    #Define arrays to hold upper and lower bounds
    lowMat = [(tup..., Inf) for tup in gridMat]
    highMat = [(tup..., -Inf) for tup in gridMat]

    #Iterate across grid cells 
    for cell in Iterators.product(subgrid_faces...)
        #Get vertex indices associated with each face of the cell
        vertices = [(i, i+1) for i in cell]
        #Convert these into grid point indices 
        verts = collect(Iterators.product(vertices...))

        #Get the sub-grids associated with each face of the cell
        coordinates = [(subgrids[sg][i], subgrids[sg][i+1]) for (sg, i) in enumerate(cell)]
        #Use the sub-grids to get grid coordinates 
        coords = collect(Iterators.product(coordinates...))

        #Use pointwise interval arithmetic to get upper and lower bounds
        LBs = [min(lb1_int(coord...)/lb2_int(coord...), lb1_int(coord...)/ub2_int(coord...), ub1_int(coord...)/lb2_int(coord...), ub1_int(coord...)/ub2_int(coord...)) for coord in coords]
        UBs = [max(lb1_int(coord...)/lb2_int(coord...), lb1_int(coord...)/ub2_int(coord...), ub1_int(coord...)/lb2_int(coord...), ub1_int(coord...)/ub2_int(coord...)) for coord in coords]

        #Now for soundness, use cell-wise interval arithmetic bounds
        #TODO: Optimization here for tighter bounds 
        LBs .= minimum(LBs)
        UBs .= maximum(UBs)

        #Update the matrix of bounds
        for (i, vert) in enumerate(verts)
            pt = lowMat[vert...][1:end-1] 
            #Update the bounds only if the new bound is looser than the current
            lowMat[vert...]= (pt...,min(lowMat[vert...][end], LBs[i]))
            highMat[vert...] = (pt...,max(highMat[vert...][end], UBs[i]))
        end
    end

    #Return a list of bounds
    divLB = [lowMat...]
    divUB = [highMat...]

    return divLB, divUB
end
