include("overtHZ_helpers.jl")

baseFunc = :(cos(x)cos(y)x*y^2 + sin(x)cos(y)y)
vars = find_variables(baseFunc)


#Basic breakdown of a single chunk of the expression
dec = parse_and_reduce(baseFunc)

disCase = dec[2]
disTrib = distribMul(disCase)
distribMul(dec[3])

varVec = find_variables(baseFunc)

likeChunks = [likeTerms(disTrib, var) for var in varVec]

likeExpr = [array2expr(chunk) for chunk in likeChunks]

#Test multiplication over addition 
sumFunc = :((cos(x) + x)*x)
sumDec = parse_and_reduce(sumFunc)
sumDist = distribMul(sumDec)
array2expr(sumDist)


#Harder case 
sumFunc = :((cos(x) + x)*x + (cos(y) + y)*y)
sumDec = parse_and_reduce(sumFunc)
sumDist = distribMul(sumDec[3])
#Test nested multiplication over multiplication 
reCase = Any[:*, Any[:*, Any[:*, Any[:*, Any[:cos, :x], :x], Any[:cos, :x]], :x], Any[:cos, :x]]
distribMul(reCase)

#Test addition cases 
addFunc = :(x + (x + (x + y) + x) + (x + y))
addDec = parse_and_reduce(addFunc)

addDist = distribAdd(addDec[3])

subFunc = :(x - (x - (x - y) - x) - (x - y))
subDec = parse_and_reduce(subFunc)
subDist = distribAdd(subDec[2])