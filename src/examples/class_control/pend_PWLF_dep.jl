#Finer bounds (npoint = 5)
p1_LB_2_1, p1_UB_2_1 = interpol(bound_univariate(p1, lbs1[1], ubs1[1], npoint = 5)...)
p1_LB_2_2, p1_UB_2_2 = interpol(bound_univariate(p1, lbs2[1], ubs2[1], npoint = 5)...)
p2_LB_2_1, p2_UB_2_1 = interpol(bound_univariate(p2, lbs1[2], ubs1[2], npoint = 5)...)
p2_LB_2_2, p2_UB_2_2 = interpol(bound_univariate(p2, lbs2[2], ubs2[2], npoint = 5)...)

#Lift bounds to space of (x₁, x₂)
#Lift part 1 first
l_p1_LB_21, l_p1_UB_21 = lift_OA([2], [1], p1_LB_2_1, p1_UB_2_1, lbs1, ubs1)
l_p1_LB_22, l_p1_UB_22 = lift_OA([2], [1], p1_LB_2_2, p1_UB_2_2, lbs2, ubs2)

l_p1_LB_2 = sort(unique(vcat(l_p1_LB_21, l_p1_LB_22); dims =1))
l_p1_UB_2 = sort(unique(vcat(l_p1_UB_21, l_p1_UB_22); dims =1))

#Lift part 2
l_p2_LB_21, l_p2_UB_21 = lift_OA([1], [2], p2_LB_2_1, p2_UB_2_1, lbs1, ubs1)
l_p2_LB_22, l_p2_UB_22 = lift_OA([1], [2], p2_LB_2_2, p2_UB_2_2, lbs2, ubs2)

l_p2_LB_2 = sort(unique(vcat(l_p2_LB_21, l_p2_LB_22); dims =1))
l_p2_UB_2 = sort(unique(vcat(l_p2_UB_21, l_p2_UB_22); dims =1))

#Sum the bounds to recover 2d bounds
LB_2, UB_2 = sumBounds(l_p1_LB_2, l_p1_UB_2, l_p2_LB_2, l_p2_UB_2, true)
validBounds(:(sin(x1) - x2), [:x1, :x2],LB_2, UB_2, true)

LB_2_inps = [tup[1:end-1] for tup in LB_2]
LB_2_Tri = OA2PWA(LB_2)

#Write to file
open("pend_n5_bounds.txt", "w") do file 
    for tup in LB_2_inps
        write(file, string(tup[1], ",", tup[2], "\n"))
    end
end

open("pend_n5_tri.txt", "w") do file 
    for tup in LB_2_Tri
        write(file, string(tup[1], ",", tup[2], ",", tup[3], "\n"))
    end
end

#Finer bounds (npoint = 10)
p1_LB_3_1, p1_UB_3_1 = interpol(bound_univariate(p1, lbs1[1], ubs1[1], npoint = 10)...)
p1_LB_3_2, p1_UB_3_2 = interpol(bound_univariate(p1, lbs2[1], ubs2[1], npoint = 10)...)
p2_LB_3_1, p2_UB_3_1 = interpol(bound_univariate(p2, lbs1[2], ubs1[2], npoint = 10)...)
p2_LB_3_2, p2_UB_3_2 = interpol(bound_univariate(p2, lbs2[2], ubs2[2], npoint = 10)...)

#Lift bounds to space of (x₁, x₂)
#Lift part 1 first
l_p1_LB_31, l_p1_UB_31 = lift_OA([2], [1], p1_LB_3_1, p1_UB_3_1, lbs1, ubs1)
l_p1_LB_32, l_p1_UB_32 = lift_OA([2], [1], p1_LB_3_2, p1_UB_3_2, lbs2, ubs2)

l_p1_LB_3 = sort(unique(vcat(l_p1_LB_31, l_p1_LB_32); dims =1))
l_p1_UB_3 = sort(unique(vcat(l_p1_UB_31, l_p1_UB_32); dims =1))

#Lift part 2
l_p2_LB_31, l_p2_UB_31 = lift_OA([1], [2], p2_LB_3_1, p2_UB_3_1, lbs1, ubs1)
l_p2_LB_32, l_p2_UB_32 = lift_OA([1], [2], p2_LB_3_2, p2_UB_3_2, lbs2, ubs2)

l_p2_LB_3 = sort(unique(vcat(l_p2_LB_31, l_p2_LB_32); dims =1))
l_p2_UB_3 = sort(unique(vcat(l_p2_UB_31, l_p2_UB_32); dims =1))

#Sum the bounds to recover 2d bounds
LB_3, UB_3 = sumBounds(l_p1_LB_3, l_p1_UB_3, l_p2_LB_3, l_p2_UB_3, true)
validBounds(:(sin(x1) - x2), [:x1, :x2],LB_3, UB_3, true)

LB_3_inps = [tup[1:end-1] for tup in LB_3]
LB_3_Tri = OA2PWA(LB_3)

#Write to file
open("pend_n10_bounds.txt", "w") do file 
    for tup in LB_3_inps
        write(file, string(tup[1], ",", tup[2], "\n"))
    end
end

open("pend_n10_tri.txt", "w") do file 
    for tup in LB_3_Tri
        write(file, string(tup[1], ",", tup[2], ",", tup[3], "\n"))
    end
end

#Finer bounds (npoint = 50)
p1_LB_4_1, p1_UB_4_1 = interpol(bound_univariate(p1, lb11, ub11, npoint = 50)...)
p1_LB_4_2, p1_UB_4_2 = interpol(bound_univariate(p1, lb12, ub12, npoint = 50)...)
p2_LB_4_1, p2_UB_4_1 = interpol(bound_univariate(p2, lb21, ub21, npoint = 50)...)
p2_LB_4_2, p2_UB_4_2 = interpol(bound_univariate(p2, lb22, ub22, npoint = 50)...)

p1_LB_4 = sort(unique(vcat(p1_LB_4_1, p1_LB_4_2); dims =1))
p1_UB_4 = sort(unique(vcat(p1_UB_4_1, p1_UB_4_2); dims =1))
p2_LB_4 = sort(unique(vcat(p2_LB_4_1, p2_LB_4_2); dims =1))
p2_UB_4 = sort(unique(vcat(p2_UB_4_1, p2_UB_4_2); dims =1))

emptyList = [2]
currList = [1]
l_p1_LB_4, l_p1_UB_4 = lift_OA(emptyList, currList, p1_LB_4, p1_UB_4, lbList, ubList)

emptyList = [1]
currList = [2]
l_p2_LB_4, l_p2_UB_4 = lift_OA(emptyList, currList, p2_LB_4, p2_UB_4, lbList, ubList)

LB_4, UB_4 = sumBounds(l_p1_LB_4, l_p1_UB_4, l_p2_LB_4, l_p2_UB_4, true)

validBounds(:(sin(x1) - x2), [:x1, :x2],LB_4, UB_4, true)

LB_4_inps = [tup[1:end-1] for tup in LB_4]
LB_4_Tri = OA2PWA(LB_4)

#Write to file
open("pend_n50_bounds.txt", "w") do file 
    for tup in LB_4_inps
        write(file, string(tup[1], ",", tup[2], "\n"))
    end
end

open("pend_n50_tri.txt", "w") do file 
    for tup in LB_4_Tri
        write(file, string(tup[1], ",", tup[2], ",", tup[3], "\n"))
    end
end
#Finer bounds (npoint = 100)
p1_LB_5_1, p1_UB_5_1 = interpol(bound_univariate(p1, lb11, ub11, npoint = 100)...)
p1_LB_5_2, p1_UB_5_2 = interpol(bound_univariate(p1, lb12, ub12, npoint = 100)...)
p2_LB_5_1, p2_UB_5_1 = interpol(bound_univariate(p2, lb21, ub21, npoint = 100)...)
p2_LB_5_2, p2_UB_5_2 = interpol(bound_univariate(p2, lb22, ub22, npoint = 100)...)

p1_LB_5 = sort(unique(vcat(p1_LB_5_1, p1_LB_5_2); dims =1))
p1_UB_5 = sort(unique(vcat(p1_UB_5_1, p1_UB_5_2); dims =1)) 
p2_LB_5 = sort(unique(vcat(p2_LB_5_1, p2_LB_5_2); dims =1))
p2_UB_5 = sort(unique(vcat(p2_UB_5_1, p2_UB_5_2); dims =1))

emptyList = [2]
currList = [1]
l_p1_LB_5, l_p1_UB_5 = lift_OA(emptyList, currList, p1_LB_5, p1_UB_5, lbList, ubList)

emptyList = [1]
currList = [2]
l_p2_LB_5, l_p2_UB_5 = lift_OA(emptyList, currList, p2_LB_5, p2_UB_5, lbList, ubList)

LB_5, UB_5 = sumBounds(l_p1_LB_5, l_p1_UB_5, l_p2_LB_5, l_p2_UB_5, true)

validBounds(:(sin(x1) - x2), [:x1, :x2],LB_5, UB_5, true)

LB_5_inps = [tup[1:end-1] for tup in LB_5]
LB_5_Tri = OA2PWA(LB_5)

#Write to file
open("pend_n100_bounds.txt", "w") do file 
    for tup in LB_5_inps
        write(file, string(tup[1], ",", tup[2], "\n"))
    end
end

open("pend_n100_tri.txt", "w") do file 
    for tup in LB_5_Tri
        write(file, string(tup[1], ",", tup[2], ",", tup[3], "\n"))
    end
end