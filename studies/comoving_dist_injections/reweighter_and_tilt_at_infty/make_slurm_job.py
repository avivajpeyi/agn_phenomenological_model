import glob
import os
import shutil

TEMPLATE = '''#! /bin/bash
#SBATCH --job-name={{{LABEL}}}
#SBATCH --output={{{LOG_DIR}}}/{{{LABEL}}}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00


module load anaconda3/5.1.0
module load gcc/9.2.0
module load openmpi/4.0.2
module load mpi4py/3.0.3-python-3.7.4
module load lalsuite-lalsimulation/2.0.0
module load astropy/4.0.1-python-3.7.4
module load scipy-bundle/2019.10-python-3.7.4
module load h5py/3.2.1-python-3.7.4
source /fred/oz980/avajpeyi/envs/gstar_venv/bin/activate

echo "{{{EXE}}} {{{ARGS}}}" 
{{{EXE}}} {{{ARGS}}} &> {{{LOG_DIR}}}/job_{{{LABEL}}}.err

'''


def create_slurm_submission_file(label, job_exe, job_cli_args_str, log_dir="job_logs"):
    os.makedirs(log_dir, exist_ok=True)
    txt = TEMPLATE
    txt = txt.replace("{{{LOG_DIR}}}", log_dir)
    txt = txt.replace("{{{LABEL}}}", label)
    txt = txt.replace("{{{EXE}}}", shutil.which(job_exe))
    txt = txt.replace("{{{ARGS}}}", job_cli_args_str)

    fname = f"slurm_{label}.sh"
    with open(fname, "w") as f:
        f.write(txt)
    return fname


EXE = "reweight_res"
REGEX = dict(
    pop_a="/fred/oz980/avajpeyi/projects/agn_phenomenological_model/studies/comoving_dist_injections/outdir_pop_a_validsnr_clean/out_*/result/*_result.json",
    pop_b="/fred/oz980/avajpeyi/projects/agn_phenomenological_model/studies/comoving_dist_injections/outdir_pop_b_validsnr_clean/out_*/result/*_result.json",
)
ARGS = dict(
    pop_a="0.5 3.0",
    pop_b="1 0.25",
)


def make_jobs(label):
    cwd = os.getcwd()
    outdir = f"out_{label}"
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    files = glob.glob(REGEX[label])
    job_commands = []
    for i, f in enumerate(files):
        args = f"{f} {ARGS[label]}"
        job_file = create_slurm_submission_file(
            label=f"{label}_{i}",
            job_exe=EXE,
            job_cli_args_str=args,
        )
        job_commands.append(f"sbatch {job_file}")

    with open("MAIN_RUNNER.sh", 'w') as f:
        f.writelines(job_commands)
    os.chdir(cwd)


def main():
    make_jobs('pop_a')
    make_jobs('pop_b')


if __name__ == '__main__':
    main()
