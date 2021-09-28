import glob
import os
import shutil

TEMPLATE = '''#! /bin/bash
#SBATCH --job-name={{{LABEL}}}
#SBATCH --output={{{LOG_DIR}}}/log_{{{LABEL}}}.out
#SBATCH --error={{{LOG_DIR}}}/log_{{{LABEL}}}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=400:00


module load anaconda3/5.1.0
module load gcc/9.2.0
module load openmpi/4.0.2
module load mpi4py/3.0.3-python-3.7.4
module load lalsuite-lalsimulation/2.0.0
module load astropy/4.0.1-python-3.7.4
module load scipy-bundle/2019.10-python-3.7.4
module load h5py/3.2.1-python-3.7.4
source /fred/oz980/avajpeyi/envs/gstar_venv/bin/activate


/fred/oz980/avajpeyi/envs/gstar_venv/bin/python setup_and_sample.py {{{ARGS}}}

'''


def create_slurm_submission_file(label, job_cli_args_str, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    txt = TEMPLATE
    txt = txt.replace("{{{LOG_DIR}}}", log_dir)
    txt = txt.replace("{{{LABEL}}}", label)
    txt = txt.replace("{{{ARGS}}}", job_cli_args_str)

    fname = f"slurm_{label}.sh"
    with open(fname, "w") as f:
        f.write(txt)
    return fname



def make_jobs(label):
    outdir = f"outdir_{label}"
    os.makedirs(outdir, exist_ok=True)

    job_file = create_slurm_submission_file(
        label=f"{label}",
        job_cli_args_str=label,
        log_dir=os.path.join(outdir, "logs")
    )
    print(f"\nsbatch {job_file}")



def main():
    make_jobs('run_a')
    make_jobs('run_b')


if __name__ == '__main__':
    main()
