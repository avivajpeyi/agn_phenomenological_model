"""
Module to create an injection file + pbilby inis for the injections.
"""
import logging
import os
import shutil
from typing import Optional

import pandas as pd

logging.getLogger().setLevel(logging.INFO)

DIR = os.path.dirname(__file__)
CONFIG_TEMPLATE = os.path.join(DIR, "pbilby_config_template.ini")
GEN_TEMPLATE = os.path.join(DIR, "pbilby_generation_job_template.ini")

SSTAR_ENV = "/fred/oz980/avajpeyi/envs/sstar_venv/bin/activate"
SSTAR_TASKS = 14

GSTAR_ENV = "/fred/oz980/avajpeyi/envs/gstar_venv/bin/activate"
GSTAR_TASKS = 12

SSTAR_KWAGS = dict(tasks=SSTAR_TASKS, env=SSTAR_ENV)
GSTAR_KWAGS = dict(tasks=GSTAR_TASKS, env=GSTAR_ENV)

CLUSTER_KWARGS = dict(sstar=SSTAR_KWAGS, gstar=GSTAR_KWAGS)


def create_ini(injection_idx: int, injection_file: str, prior_file: str, label: str, psd_file: str, waveform: str, fref:float,
               nlive: Optional[int] = 1000, nact: Optional[int] = 5, nodes: Optional[int] = 1,
               tasks: Optional[int] = 14, time: Optional[str] = "24:00:00", mem: Optional[int] = 800,
               env: Optional[str] = SSTAR_ENV,
               ):
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
        txt = txt.replace("{{{NLIVE}}}", str(nlive))
        txt = txt.replace("{{{NACT}}}", str(nact))
        txt = txt.replace("{{{NODES}}}", str(nodes))
        txt = txt.replace("{{{TASKS}}}", str(tasks))
        txt = txt.replace("{{{TIME}}}", time)
        txt = txt.replace("{{{MEM}}}", str(mem))
        txt = txt.replace("{{{ENV}}}", env)
        txt = txt.replace("{{{FREF}}}", str(fref))
    with open(ini, "w") as f:
        f.write(txt)
    print("Completed ini writing")


def create_data_generation_slurm_submission_file(num_inj, label):
    gen_log_dir = f"outdir_{label}/generation_log"
    os.makedirs(f"outdir_{label}/generation_log", exist_ok=True)
    with open(GEN_TEMPLATE, "r") as f:
        txt = f.read()
        txt = txt.replace("{{{GENERATION_LOG_DIR}}}", os.path.abspath(gen_log_dir))
        txt = txt.replace("{{{NUM_INJECTIONS}}}", str(num_inj))
        txt = txt.replace("{{{LABEL}}}", label)
        txt = txt.replace(
            "{{{GENERATION_EXE}}}", shutil.which("parallel_bilby_generation")
        )

    with open(f"outdir_{label}/slurm_data_generation_{label}", "w") as f:
        f.write(txt)


def create_analysis_bash_runner(num_inj, label):
    file_contents = "#! /bin/sh\n"
    for i in range(num_inj):
        ulabel = f"{label}_{i:02}"
        analysis_file = os.path.abspath(f"outdir_{label}/out_{ulabel}/submit/bash_{ulabel}.sh")
        file_contents += f"bash {analysis_file}\n"
    with open(f"outdir_{label}/start_data_analysis_{label}", "w") as f:
        f.write(file_contents)


def pbilby_jobs_generator(injection_file, label, prior_file, psd_file, waveform, cluster, fref):
    logging.info("Generating parallel bilby ini files + submission scripts")

    injection_file = os.path.abspath(injection_file)
    if os.path.isfile(prior_file):
        prior_file = os.path.abspath(prior_file)
    psd_file = os.path.abspath(psd_file)

    num_inj = len(pd.read_csv(injection_file))

    for i in range(num_inj):
        create_ini(
            injection_idx=i,
            injection_file=injection_file,
            prior_file=prior_file,
            label=label,
            psd_file=psd_file,
            waveform=waveform,
            fref=fref,
            **CLUSTER_KWARGS[cluster]
        )

    create_data_generation_slurm_submission_file(num_inj, label=label)
    logging.info("Start generation jobs with:\n$ sbatch slurm_data_generation.sh")

    create_analysis_bash_runner(num_inj, label=label)
    logging.info(
        "After generation, begin analysis with:\n$ bash start_data_analysis.sh"
    )
