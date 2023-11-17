using MiniQhull


function OA2PWA(LB, UB)
    """
    Takes an overt upper/lower bound and returns a piecewise affine function (defined in the style of Boyd) with a corresponding triangulation.
    """

    lbDom = [tup[1:end-1] for tup in LB]
    ubDom = [tup[1:end-1] for tup in UB]

    lbTri = delaunay(lbDom)
    ubTri = delaunay(ubDom)

    lbInc = vertex2inc(lbTri)
    ubInc = vertex2inc(ubTri)

    return lbInc, ubInc
end

function vertex2inc(vert)
    """
    Takes a matrix of vertices and returns an incidence matrix M

    Args:
        vert: a matrix of size (d, n) where d is the number of vertices in a (d-1) dimensional simplex and n is the number of simplices 
    Returns:
        M: a matrix of size (nV, N) where Nv is the total number of vertices and N is the number of simplices. M[i,j] = 1 if vertex i is in simplex j and 0 otherwise.
        
    """

    M = zeros(maximum(vert), size(vert)[end])
    colCounter = 0
    for col in eachcol(vert)
        colCounter += 1
        for row in col
            M[row, colCounter] = 1
        end
    end
    return M
end
