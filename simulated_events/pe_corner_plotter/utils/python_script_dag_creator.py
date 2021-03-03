import os
import sys
from typing import Dict, List

import pycondor

ACCOUNTING_GROUP = "ligo.prod.o3.cbc.pe.lalinference"


def create_python_script_job(python_script: str, job_name: str,
                             job_args_dict: Dict[str, str], logdir: str, subdir:str,
                             dag: pycondor.Dagman):
    """ Creates job-node for  python script
    :param python_script: python script path (eg test.py)
    :param job_name: unique name for this job
    :param job_args_dict: {job_arg: arg_val}
    :param logdir: the dir to store logs
    :param subdir: the dir to store submit file
    :param dag: the dag this  job is attached to
    :return: constructed pycondor.Job
    """
    return pycondor.Job(
        name=job_name,
        executable=sys.executable,
        error=logdir,
        log=logdir,
        output=logdir,
        submit=subdir,
        getenv=True,
        universe="vanilla",
        dag=dag,
        arguments=f"{python_script} {convert_args_dict_to_str(job_args_dict)}",
        extra_lines=[f"accounting_group = {ACCOUNTING_GROUP}"]
    )


def convert_args_dict_to_str(job_args):
    return " ".join([f"--{arg} {val}" for arg, val in job_args.items()])


def create_python_script_jobs(
        main_job_name: str, run_dir: str, python_script: str,
        job_args_list: List[Dict],
        job_names_list: List[str]
):
    """ Creates a set of parallel jobs for a python script

    :param main_job_name: name of main job
    :param run_dir: the absdir where the run will be executed from
    :param python_script: the abspath to the python script
    :param job_args_list: list of args-dicts for the python script [{job_arg: arg_val}]
    :param job_names_list: list of the individual job names
    :return: None
    """
    run_dir = os.path.abspath(run_dir)
    python_script = os.path.abspath(python_script)
    print(f"Making jobs for {python_script}.")
    subdir = os.path.join(run_dir, "submit")
    logdir = os.path.join(run_dir, "logs")
    dag = pycondor.Dagman(
        name=main_job_name,
        submit=subdir,
    )
    jobs = []
    for job_name, job_args in zip(job_names_list, job_args_list):
        jobs.append(
            create_python_script_job(python_script, job_name, job_args, logdir, subdir, dag)
        )
    dag.build(makedirs=True, fancyname=False)
    command_line = "$ condor_submit_dag {}".format(
        os.path.relpath(dag.submit_file)
    )
    print(f"Created {len(jobs)} jobs. Run:\n{command_line}\n")
