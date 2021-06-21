"""
Module to create an injection file + pbilby inis for the injections.
"""
import logging
import os
import shutil

import pandas as pd

logging.getLogger().setLevel(logging.INFO)

DIR = os.path.dirname(__file__)
CONFIG_TEMPLATE = os.path.join(DIR, "pbilby_config_template.ini")
GEN_TEMPLATE = os.path.join(DIR, "pbilby_generation_job_template.ini")


def create_ini(injection_idx: int, injection_file: str, prior_file: str, label: str, psd_file: str, waveform: str):
    unique_label = f"{label}_{injection_idx:02}"
    outdir = f"outdir_{label}/out_{unique_label}"
    ini = f"outdir_{label}/{unique_label}.ini"
    os.makedirs(outdir, exist_ok=True)
    with open(CONFIG_TEMPLATE, "r") as f:
        txt = f.read()
        txt = txt.replace("{{{IDX}}}", str(injection_idx))
        txt = txt.replace("{{{LABEL}}}", unique_label)
        txt = txt.replace("{{{OUTDIR}}}", os.path.abspath(outdir))
        txt = txt.replace("{{{PRIOR_FILE}}}", prior_file)
        txt = txt.replace("{{{INJECTION_FILE}}}", injection_file)
        txt = txt.replace("{{{PSD_FILES}}}", "{" + f"H1={psd_file}, L1={psd_file}" + "}")
        txt = txt.replace("{{{WAVEFORM}}}", waveform)
    with open(ini, "w") as f:
        f.write(txt)


def create_data_generation_slurm_submission_file(num_inj, label):
    os.makedirs("generation_log", exist_ok=True)
    with open(GEN_TEMPLATE, "r") as f:
        txt = f.read()
        txt = txt.replace("{{{GENERATION_LOG_DIR}}}", "generation_log")
        txt = txt.replace("{{{NUM_INJECTIONS}}}", str(num_inj))
        txt = txt.replace("{{{LABEL}}}", label)
        txt = txt.replace(
            "{{{GENERATION_EXE}}}", shutil.which("parallel_bilby_generation")
        )

    with open(f"slurm_data_generation_{label}", "w") as f:
        f.write(txt)


def create_analysis_bash_runner(num_inj, label):
    file_contents = "#! /bin/sh\n"
    for i in range(num_inj):
        ulabel = f"{label}_{i:02}"
        analysis_file = f"outdir_{label}/out_{ulabel}/submit/bash_{ulabel}.sh"
        file_contents += f"bash {analysis_file}\n"
    with open(f"start_data_analysis_{label}", "w") as f:
        f.write(file_contents)


def main_generator(injection_file, label, prior_file, psd_file, waveform):
    logging.info("Generating parallel bilby ini files + submission scripts")

    injection_file = os.path.abspath(injection_file)
    if os.path.isfile(prior_file):
        prior_file = os.path.abspath(prior_file)
    psd_file = os.path.abspath(psd_file)

    num_inj = len(pd.read_csv(injection_file))
    for i in range(num_inj):
        create_ini(injection_idx=i, injection_file=injection_file, prior_file=prior_file, label=label,
                   psd_file=psd_file, waveform=waveform)

    create_data_generation_slurm_submission_file(num_inj, label=label)
    logging.info("Start generation jobs with:\n$ sbatch slurm_data_generation.sh")

    create_analysis_bash_runner(num_inj, label=label)
    logging.info(
        "After generation, begin analysis with:\n$ bash start_data_analysis.sh"
    )
