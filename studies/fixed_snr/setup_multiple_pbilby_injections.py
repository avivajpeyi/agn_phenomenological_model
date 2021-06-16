"""
Module to create an injection file + pbilby inis for the injections.
"""
import logging
import os
import shutil
import pandas as pd
from bilby_pipe.create_injections import create_injection_file

logging.getLogger().setLevel(logging.INFO)

LABEL = "snr60"  # the main name of the injections
INJECTION_FILE = f"./datafiles/{LABEL}_injections.dat"
PRIOR_FILE = "./datafiles/bbh.prior"


def create_ini(injection_idx: int):
    unique_label = f"{LABEL}_{injection_idx:02}"
    outdir = f"out_{unique_label}"
    ini = f"{unique_label}.ini"
    with open("pbilby_config_template.ini", "r") as f:
        txt = f.read()
        txt = txt.replace("{{{IDX}}}", str(injection_idx))
        txt = txt.replace("{{{LABEL}}}", unique_label)
        txt = txt.replace("{{{OUTDIR}}}", outdir)
        txt = txt.replace("{{{PRIOR_FILE}}}", PRIOR_FILE)
        txt = txt.replace("{{{INJECTION_FILE}}}", INJECTION_FILE)
    with open(ini, "w") as f:
        f.write(txt)


def create_data_generation_slurm_submission_file(num_inj):
    os.makedirs("generation_log", exist_ok=True)
    with open("pbilby_generation_job_template.ini", "r") as f:
        txt = f.read()
        txt = txt.replace("{{{GENERATION_LOG_DIR}}}", "generation_log")
        txt = txt.replace("{{{NUM_INJECTIONS}}}", str(num_inj))
        txt = txt.replace("{{{LABEL}}}", LABEL)
        txt = txt.replace(
            "{{{GENERATION_EXE}}}", shutil.which("parallel_bilby_generation")
        )
    with open("slurm_data_generation.sh", "w") as f:
        f.write(txt)


def create_analysis_bash_runner(num_inj):
    file_contents = "#! /bin/sh\n"
    for i in range(num_inj):
        label = f"{LABEL}_{i:02}"
        analysis_file = f"out_{label}/submit/bash_{label}.sh"
        file_contents += f"bash {analysis_file}\n"
    with open("start_data_analysis.sh", "w") as f:
        f.write(file_contents)


def main():
    logging.info("Generating parallel bilby ini files + submission scripts")
    num_inj = len(pd.read_csv(INJECTION_FILE))
    for i in range(num_inj):
        create_ini(injection_idx=i)

    create_data_generation_slurm_submission_file(num_inj)
    logging.info("Start generation jobs with:\n$ sbatch slurm_data_generation.sh")

    create_analysis_bash_runner(num_inj)
    logging.info(
        "After generation, begin analysis with:\n$ bash start_data_analysis.sh"
    )


if __name__ == "__main__":
    main()