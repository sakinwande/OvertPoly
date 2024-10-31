#!/bin/bash

#SBATCH --time=00:00:1200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=octa
#SBATCH --chdir /barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/nn_control/
#SBATCH --job-name=L4DC_Bench
#SBATCH --error=error-%j.log
#SBATCH --output=output%j.log


module load julia
julia --startup-file=no single_pend_overtPoly_graph_bench.jl