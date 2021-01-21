import os
import shutil
import glob

import pycondor

JOBNAME = "calculate_agn_spins"
PYTHON = shutil.which("python3")
SCRIPT = "add_agn_spins_to_samples.py"


def main():
    files = glob.glob("../data/gwtc2_samples/*.csv")
    args_list = [f"{SCRIPT} {f}" for f in files]
    make_condor_job(args_list)

def make_condor_job(args_list):
    maindir = os.path.abspath(f"./condor/{JOBNAME}")
    if os.path.isdir(maindir):
        shutil.rmtree(maindir)
    logdir = os.path.join(maindir, "log")
    subdir = os.path.join(maindir, "sub")
    dagman = pycondor.Dagman(name=JOBNAME, submit=subdir)
    for i, arg_dict in enumerate(args_list):
        job = pycondor.Job(
            name=f"{JOBNAME}_{i}",
            executable=PYTHON,
            output=logdir,
            error=logdir,
            submit=subdir,
            request_memory="1GB",
            getenv="True",
            universe="vanilla",
            extra_lines=[
                "accounting_group_user = avi.vajpeyi",
                "accounting_group = ligo.dev.o3.cbc.pe.lalinference"
            ],
            dag=dagman
        )
        job.add_arg(args_list[i])
    dagman.build_submit(submit_options="True", fancyname=False)


if __name__ == '__main__':
    main()
