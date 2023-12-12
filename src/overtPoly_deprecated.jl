function sameInp(LB, UB)
    """
    Takes an overt OA and interpolates to ensure that the lower and upper bounds are over the same set of points
    """
    newUB = Any[]
    newLB = Any[]

    newXs = sort(unique(vcat([tup[1] for tup in LB], [tup[1] for tup in UB])))

    for inp in newXs
        #Check if this input has a lower bound 
        lbInd = findall(x->x[1] == inp, LB)

        #If it does, add to newLB, else interpolate
        if !isempty(lbInd)
            push!(newLB, LB[lbInd[1]])
        else
            #Find the lower bound that is closest to the input
            lbInd = findall(x->x[1] < inp, LB)
            #Interpolate
            push!(newLB, (inp, interpol(inp, LB[lbInd[end]], LB[lbInd[end]+1])))
        end

        #Check if this input has an upper bound
        ubInd = findall(x->x[1] == inp, UB)

        #Similarly, if it does, add to newUB, else interpolate
        if !isempty(ubInd)
            push!(newUB, UB[ubInd[1]])
        else
            #Find the upper bound that is closest to the input
            ubInd = findall(x->x[1] < inp, UB)
            #Interpolate
            push!(newUB, (inp, interpol(inp, UB[ubInd[end]], UB[ubInd[end]+1])))
        end

    end

    return newLB, newUB

end

function interpol(xInp, tuplb, tupub)
    """
    When it's not catching international criminals, this method interpolates the lower and upper bounds of a tuple to ensure that the bounds are over the same set of points

    Args:
        xInp: the point at which we wish to generate an approximation
        tuplb: A tuple of (xlb, y) where xlb < xInp
        tupub: A tuple of (xub, y) where xub > xInp
        
    """

    #Assert that the input is between lb and ub
    @assert xInp >= tuplb[1] && xInp <= tupub[1]

    #Assert that the upper bound and lower boud are distinct 
    @assert tuplb[1] != tupub[1]

    #Use linear interpolation to find the output given UB and LB
    yInp = tuplb[2] + (xInp - tuplb[1])*(tupub[2] - tuplb[2])/(tupub[1] - tuplb[1])

    return yInp
end

function interpol(oA1, oA2)
    """
    When it is not catching international criminals, this method interpolates two overt approximations to ensure that they are over the same set of points
    """
    #Create a new array of inputs that are the same for both approximations
    #TODO: Consider using rounding here to ensure that the points are the same
    #Unsound flag, this is removing symmetric points. Basically, 
    newInps = sort(unique(vcat([tup[1:end-1] for tup in oA1], [tup[1:end-1] for tup in oA2])))

    #Generate interpolation function for each approximation
    interp1 = gen_interpol(oA1)
    interp2 = gen_interpol(oA2)

    #Loop through both approximations and interpolate to ensure that the bounds are over the same set of points
    newOA1 = Any[(tup..., interp1(tup...)) for tup in newInps]
    newOA2 = Any[(tup..., interp2(tup...)) for tup in newInps]

    return newOA1, newOA2

end