#!/bin/bash

#SBATCH --time=00:00:1200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=octa
#SBATCH -D /barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/
#SBATCH --job-name=L4DC_Bench
#SBATCH -e /barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/error_log
#SBATCH -o /barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/output_log

module load julia
julia --startup-file=no single_pend_overtPoly_graph_bench.jl