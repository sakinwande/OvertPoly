#!/bin/bash

#SBATCH --time=00:00:1200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=quad
#SBATCH -D /barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/nn_control

julia single_pend_overtPoly_graph_bench.jl