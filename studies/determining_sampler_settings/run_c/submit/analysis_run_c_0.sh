#!/bin/bash
#SBATCH --job-name=0_run_c
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=12
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=800
#SBATCH --output=run_c/log_data_analysis/0_run_c.log
#SBATCH --dependency=singleton

module load gcc/9.2.0 openmpi/4.0.2 mpi4py/3.0.3-python-3.7.4 lalsuite-lalsimulation/2.0.0 astropy/4.0.1-python-3.7.4 h5py/3.2.1-python-3.7.4
source /fred/oz980/avajpeyi/envs/gstar_venv/bin/activate

export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export MPI_PER_NODE=12

mpirun parallel_bilby_analysis run_c/data/run_c_data_dump.pickle --nact 20 --label run_c_0 --outdir /fred/oz980/avajpeyi/projects/agn_phenomenological_model/studies/determining_sampler_settings/run_c/result --sampling-seed 261672
