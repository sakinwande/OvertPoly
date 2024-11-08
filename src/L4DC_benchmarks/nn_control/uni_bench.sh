#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --qos=max2d
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=quad
#SBATCH --chdir /barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/nn_control/
#SBATCH --job-name=l4dc_Bench
#SBATCH --error=/barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/nn_control/error_log/error-%j.log
#SBATCH --output=/barrett/scratch/akinwande/OvertPoly/src/L4DC_benchmarks/nn_control/output_log/output-%j.log

julia --startup-file=no --project=/barrett/scratch/akinwande/OvertPoly/Project.toml unicycle_overtPoly_distrOpt.jl
