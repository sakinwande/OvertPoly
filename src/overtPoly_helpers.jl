using Symbolics
using IntervalRootFinding, IntervalArithmetic
using MacroTools: prewalk, postwalk
using Interpolations
using OVERT: bound, find_variables, to_pairs 
using Plots
global NPLOTS = 0

function bound_univariate(baseExpr::Expr, lb, ub; ϵ=1e-4, npoint=2, rel_error_tol=5e-3, plotflag=false)
    """
    Method to bound a univariate function

    Args: 
        Expr: A Julia symbolic expression. Assumed to have been parsed/converted to a univariate function 
        lb: Lower bound of the interval over which to bound the function
        ub: Upper bound of the interval over which to bound the function
    """

    #Found the symbolic variable in the Julia expression (for replacement)
    varBase = find_variables(baseExpr)[1]

    #Define differentiation variable
    @variables xₚ

    #Define derivative
    D = Differential(xₚ)
    #Define second derivative
    D2 = Differential(xₚ)^2

    #Replace expression variable with xₚ
    #NB:  we need to do this because the expression will eventually be parsed by Julia Symbolics. For the parser to work, the symbols of the expression must not be defined in the parsing module.

    #I choose xₚ because I can define a case to ensure we don't use this variable in input expressions

    #NOTE: you can actually do better. Convert baseExpr to symExpr in its native form, then use symbolics.get_variables to get the variables in the expression. Next, use Symbolics.substitute to replace problem specific variables with xₚ. This way, you avoid using Meta.parse on a string
    strExpr = string(baseExpr)
    strExpr = replace(strExpr, string(varBase) => "xₚ")
    standExpr = Meta.parse(strExpr) #Standardized expression with xₚ as the variable

    symExpr = Symbolics.parse_expr_to_symbolic(standExpr, Main)

    #Return a standard Julia function that can be evaluated from the expression
    fun = Symbolics.build_function(baseExpr, varBase, expression=Val{false})
    df = expand_derivatives(D(eval(symExpr)))
    dfunc = Symbolics.build_function(df, :xₚ; expression=Val{false})

    #Compute second derivative
    d2f = expand_derivatives(D2(symExpr))
    #d2f is an expression. Convert to a Julia function so IntervalRootFinding can use it
    #NB: expression=Val{false} returns a runtime gen function to avoid world age issues. This way we avoid evaluating expressions 

    if d2f != 0
        d2func = Symbolics.build_function(d2f, :xₚ; expression = Val{false})

        #Then find the roots over the given interval using the function
        rootVals = IntervalRootFinding.roots(d2func, IntervalArithmetic.Interval(lb, ub))
        #TODO: This is not sound, make sound
        rootsGuess = [mid.([root.interval for root in rootVals])]
        d2f_zeros = sort(rootsGuess[1])


        convex = nothing 

        UB = bound(fun, lb, ub, npoint; rel_error_tol=rel_error_tol, conc_method="continuous", lowerbound=false, df = dfunc,d2f = d2func, d2f_zeros=d2f_zeros, convex=nothing, plot=true)
        UBpoints = unique(sort(to_pairs(UB), by = x -> x[1]))

        LB = bound(fun, lb, ub, npoint; rel_error_tol=rel_error_tol, conc_method="continuous", lowerbound=true, df = dfunc,d2f = d2func, d2f_zeros=d2f_zeros, convex=nothing, plot=true)
        LBpoints = unique(sort(to_pairs(LB), by = x -> x[1]))
    else
        #Account for linear case
        UBpoints = [(lb, fun(lb)), (ub, fun(ub))] 
        LBpoints = [(lb, fun(lb)), (ub, fun(ub))]

    end
    if plotflag
        plotRes2d(baseExpr, fun, lb, ub, LBpoints, UBpoints, varBase, true)
    end
    
    return UBpoints, LBpoints
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

function MinkSum(vec1, vec2)
    """
    Returns the Minkowski sum of vectors of tuples. The vectors can have arbitrary length but tuples must have the same length

    (since the sets must have same dimension)
    """
    #TODO: Ensure that dimensions of tuples are the same

    minkResult = Any[]
    for tup in vec1
        for tup2 in vec2
            push!(minkResult, Tuple(tup[i] + tup2[i] for i in eachindex(tup)))
        end
    end

    return minkResult

end

function plotSurf(baseFunc, lb, ub, lbVec, ubVec, surfDim, xlb, ylb, xub, yub, saveFlag=false)

    fun1 = Symbolics.build_function(baseFunc, find_variables(baseFunc)..., expression=Val{false})
    xC = collect(range(lb, ub, length=100))
    yC = collect(range(lb, ub, length=100))

    lbMat = reshape([p[3] for p in lbVec], surfDim)
    ubMat = reshape([p[3] for p in ubVec], surfDim)

    sPlot = plot(xC, yC, fun1, st=:surface, camera=(-30,30), color="black", label="function", opacity = 1.0)
    plot!(sPlot, [p[1] for p in xlb], [p[1] for p in ylb], lbMat, st=:surface, color="blue", label="lower bound", showscale=false, opacity=1.0)
    plot!(sPlot, [p[1] for p in xub], [p[1] for p in yub], ubMat, st=:surface, color="orange", label="upper bound", showscale=false,opacity=1.0)

    if saveFlag
        global NPLOTS
        NPLOTS += 1
        savefig(sPlot, "plots/bound_"*string(NPLOTS)*".html")
    end

end

function addDim(vec, dim)
    """
    Add a dimension to each tuple in a vector of tuples. This is equivalent to lifting a n-d polytope to a dimention nd+1

    Equiv to a cartesian product with the zero vector in the new dimension
    """
    newVec = Any[]
    for tup in vec
        newTup = Tuple(if i < dim tup[i] elseif i == dim 0.0 else tup[i-1] end for i in 1:length(tup)+1)
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
    plotSurf(expr, lb, ub, v5LB, v5UB, surfDim, xS, yS, xS, yS, true)

    return v5LB, v5UB
end 

function boundMV2(expr, lb, ub)
    #Reduce to addition chunks
    baseParsed = parse_and_reduce(expr)

    #Divvy up into chunks. Technically, if it's multiplication or division, we can just distribute and not worry about chunks
    baseFunc1 = baseParsed[2]
    baseFunc2 = baseParsed[3]

    #Collect like terms/reduce to univariate chunks 
    parsed1 = parse_and_reduce(baseFunc1)

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
    nv2LB, nv2UB = v2LB, v2UB
    nv3LB, nv3UB = v3LB, v3UB

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

    plotSurf(combFun, lb, ub, lv4LB, lv4UB, surfDim, lbXs, lbYs, ubXs, ubYs, true)



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
    plotSurf(combFun2, lb, ub, v4LB, v4UB, surfDim, lbXs, lbYs, ubXs, ubYs, true)

    #Shift to account for overapprox
    v5LB = Any[(tup[1:end-1]..., tup[end] -2*abs(v3l)*v2f(tup[1]) -2*abs(v2l)*v3f(tup[2])- 4*abs(v2l)*abs(v3l)) for tup in v4LB]
    v5UB = Any[(tup[1:end-1]..., tup[end] -2*abs(v3l)*v2f(tup[1]) -2*abs(v2l)*v3f(tup[2])- 4*abs(v2l)*abs(v3l)) for tup in v4UB]


    #Check bounds
    combFun3 = :(cos(x)cos(y)x*y^2)
    combF3 = Symbolics.build_function(combFun3, find_variables(combFun3)..., expression=Val{false})

    #Check lower bound
    sum([combF3(tup[1], tup[2]) < tup[3] for tup in v5LB])

    println(minimum([combF3(tup[1], tup[2]) - tup[3] for tup in v5LB]))

    #Check upper bound
    sum([combF3(tup[1], tup[2]) > tup[3] for tup in v5UB])

    println(maximum([combF3(tup[1], tup[2]) - tup[3] for tup in v5UB]))
    #Plot the overapproximation
    plotSurf(baseFunc1, lb, ub, v5LB, v5UB, surfDim, lbXs, lbYs, ubXs, ubYs,true)

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

function interpol(oA1, oA2)
    """
    When it is not catching international criminals, this method interpolates two overt approximations to ensure that they are over the same set of points
    """
    #Create a new array of inputs that are the same for both approximations
    #TODO Consider using rounding here to ensure that the points are the same
    #Unsound flag, this is removing symmetric points. Basically, 
    oAComb = sort(vcat(oA1, oA2))
    #NOTE: This is where the precision of the input domain is set. 
    oAVecs = [round.(collect(tup[1:end-1]), digits=5) for tup in oAComb]
    oAMat = copy(reduce(hcat,oAVecs)')
    oAMatR = reverse(oAMat, dims=2)
    oACol = [unique(col[:]) for col in eachcol(oAMat)]
    newInps = Iterators.product(oACol...)


    #Generate interpolation function for each approximation
    interp1 = gen_interpol(oA1)
    interp2 = gen_interpol(oA2)

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

# ####Debug 
# #Reduce to addition chunks
# expr=:(cos(x)cos(y)x*y^2 + sin(x)cos(y)y)
# lb, ub = -pi, pi
# baseParsed = parse_and_reduce(expr)

# #We know that each LB, UB combination is over the same set of points. So we don't have to perform 4 interpolations
# LB1, UB1 = boundMV1(baseParsed[2], lb, ub)
# LB2, UB2 = boundMV1(baseParsed[3], lb, ub)


# ##############################Debug Interpol########################
# # oA1 = LB1
# # oA2 = LB2

# # oAComb = sort(vcat(oA1, oA2))
# # oAVecs = [round.(collect(tup[1:end-1]), digits=5) for tup in oAComb]
# # oAMat = copy(reduce(hcat,oAVecs)')
# # oAMatR = reverse(oAMat, dims=2)
# # oACol = [unique(col[:]) for col in eachcol(oAMat)]
# # newInps = Iterators.product(oACol...)


# # #Generate interpolation function for each approximation
# # interp1 = gen_interpol(oA1)
# # interp2 = gen_interpol(oA2)

# # #Loop through both approximations and interpolate to ensure that the bounds are over the same set of points (and that the points are evenly spaced)

# # newOA1 = Any[]
# # newOA2 = Any[]
# # for tup in newInps
# #     tup1 = (tup..., interp1(tup...))
# #     tup2 = (tup..., interp2(tup...))
# #     push!(newOA1, tup1)
# #     push!(newOA2, tup2)
# # end
# ###################################################################

# #This interpolation is not sound. Fix now 
# nLB1, nLB2 = interpol(LB1, LB2)
# nUB1, nUB2 = interpol(UB1, UB2)


# #Interpolation results are sorted.
# nLB1 = sort(nLB1)
# nUB1 = sort(nUB1)

# LBVec = [collect(tup) for tup in nLB1]
# UBVec = [collect(tup) for tup in nUB1]

# LBMat = copy(reduce(hcat,LBVec)')
# UBMat = copy(reduce(hcat,UBVec)')
# #Extract shape of surface
# #reverse
# LBMatR = reverse(LBMat, dims=2)
# #Surfdim expects the form (y,x). This is the source of the bug
# surfDim = Tuple(length(unique(col)) for col in eachcol(LBMatR[:,2:end]))

# print(surfDim)
# xS, yS = [unique(col) for col in eachcol(LBMat[:,1:end-1])] #xS and yS are sorted

# plotSurf(baseParsed[2], lb, ub, nLB1, nUB1, surfDim, xS, yS, xS, yS, true)

# nLB2 = sort(nLB2)
# nUB2 = sort(nUB2)
# plotSurf(baseParsed[3], lb, ub, nLB2, nUB2, surfDim, xS, yS, xS, yS, true)
# #Add separate chunks 

# LB3, UB3 = sound_IA(nLB1, nUB1, nLB2, nUB2, baseParsed[1])




# ###Testing and Debugging bound_multiariate
# expr=:(cos(x)cos(y)x*y^2 + sin(x)cos(y)y)
# lb, ub = -pi, pi
    
# LB, UB = bound_multivariate(expr, lb, ub)

# #Plot this 
# plotSurf(expr, lb, ub, LB, UB, surfDim, xS, yS, xS, yS, true)



expr = :(0*x)
expr = :(2sin(x))

varBase = find_variables(expr)[1]

#Define differentiation variable
@variables xₚ

#Define derivative
D = Differential(xₚ)
#Define second derivative
D2 = Differential(xₚ)^2

strExpr = string(expr)
strExpr = replace(strExpr, string(varBase) => "xₚ")
standExpr = Meta.parse(strExpr) #Standardized expression with xₚ as the variable

symExpr = Symbolics.parse_expr_to_symbolic(standExpr, Main)

fun = Symbolics.build_function(expr, varBase, expression=Val{false})
df = expand_derivatives(D(eval(symExpr)))
dfunc = Symbolics.build_function(df, :xₚ; expression=Val{false})

#Compute second derivative
d2f = expand_derivatives(D2(symExpr))
#d2f is an expression. Convert to a Julia function so IntervalRootFinding can use it
#NB: expression=Val{false} returns a runtime gen function to avoid world age issues. This way we avoid evaluating expressions 
d2func = Symbolics.build_function(d2f, :xₚ; expression = Val{false})

typeof(d2f)
if eval(d2f) == 0
    boo = 1 
end

d2f
try 
    if d2f == 0
        boo = 1
    end
catch 
    println("Error")
end