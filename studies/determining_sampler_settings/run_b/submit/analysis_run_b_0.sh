#!/bin/bash
#SBATCH --job-name=0_run_b
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=12
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=800
#SBATCH --output=run_b/log_data_analysis/0_run_b.log
#SBATCH --dependency=singleton

module load gcc/9.2.0 openmpi/4.0.2 mpi4py/3.0.3-python-3.7.4 lalsuite-lalsimulation/2.0.0 astropy/4.0.1-python-3.7.4 h5py/3.2.1-python-3.7.4
source /fred/oz980/avajpeyi/envs/gstar_venv/bin/activate

export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export MPI_PER_NODE=12

mpirun parallel_bilby_analysis run_b/data/run_b_data_dump.pickle --nact 10 --label run_b_0 --outdir /fred/oz980/avajpeyi/projects/agn_phenomenological_model/studies/determining_sampler_settings/run_b/result --sampling-seed 261672
