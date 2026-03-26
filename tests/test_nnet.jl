using JuMP
using LazySets
using Parameters
include("../src/nv/maxSens.jl")
include("../src/nv/activation.jl")
include("../src/nv/network.jl")
include("../src/nv/constraints.jl")
include("../src/nv/util.jl")

nnetFile = "Networks/ARCH-COMP-2023/nnet/controllerSinglePendulum.nnet"
read_nnet(nnetFile)