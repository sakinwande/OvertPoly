#!/bin/bash

#SBATCH --time=00:00:1200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=octa
#SBATCH --chdir /barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/nn_control/
#SBATCH --job-name=l4dc_Bench
#SBATCH --error=/barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/nn_control/error_log/error-%j.log
#SBATCH --output=/barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/nn_control/output_log/output-%j.log

julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml single_pend_overtPoly_graph_bench.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml single_pend_overtPoly_graph_bench.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml acc_overtPoly_graph.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml acc_overtPoly_graph.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml tora_overtPoly_distrOpt.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml tora_overtPoly_distrOpt.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml unicycle_overtPoly_distrOpt.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml unicycle_overtPoly_distrOpt.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml quad_overtPoly_graph.jl
julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml quad_overtPoly_graph.jl