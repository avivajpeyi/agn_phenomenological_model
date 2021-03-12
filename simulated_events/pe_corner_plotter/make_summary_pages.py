"""
Given a GW event name, gets the NRsur and LVC result and makes the summary pages.
-- can make a dag to make all summary pages
"""

from __future__ import print_function

import argparse
import os
import glob
import re

from utils import (
    create_python_script_jobs,
)

SUMMARY_EXE = "/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py37-20210107/bin/summarypages"


def get_event_name(fname):
    name = re.findall(r"(\w*\d{6}[a-z]*)", fname)
    if len(name)==0:
        name = re.findall(r"inj\d+", fname)
    if len(name)==0:
        name = re.findall(r"data\d+", fname)
    if len(name) == 0:
        name = os.path.basename(fname).split(".")
    return name[0]


def make_summary_for_event(event_path, outdir, email):
    print(f"Making summary page for event: {event_path}")
    event_name = get_event_name(event_path)
    event_outdir = os.path.join(outdir, event_name)
    gw_data_path = glob.glob(event_path.split('/result/')[0] + "/data/*.pickle")[0]
    os.makedirs(event_outdir, exist_ok=True)
    event_labels = event_name
    event_paths = event_path
    command = f"""
   {SUMMARY_EXE}
    --samples {event_paths} 
    --labels {event_labels}
    --disable_interactive
    --no_ligo_skymap
    --email {email}
    --webdir {event_outdir}
    --aapproximant IMRPhenomXPHM
    --gwdata {gw_data_path}
    """.replace("\n", " ")
    print(f"Running: {command}")
    os.system(command)


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-dag", help="Make dag", action="store_true")
    parser.add_argument("--outdir", help="outdir for summary page", type=str,
                        default=".")
    parser.add_argument("--event-path", help="Event id (eg GW150914)", type=str,
                        default="GW150914")
    parser.add_argument("--email", help="email to notify", type=str,
                        default="")
    args = parser.parse_args()
    return args


def make_summary_dag(outdir, email):
    paths = glob.glob("../bilby_pipe_jobs/out*/result/*result.json")
    names = [get_event_name(f) for f in paths]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    args = [{"event-path": n, "outdir": outdir, "email": email} for n in paths]
    create_python_script_jobs(
        main_job_name="summary_generator",
        run_dir=".",
        python_script=os.path.abspath(__file__),
        job_args_list=args,
        job_names_list=names
    )


def main():
    args = create_parser_and_read_args()
    if args.make_dag:
        make_summary_dag(args.outdir, args.email)
    else:
        make_summary_for_event(args.event_path, args.outdir, args.email)


if __name__ == "__main__":
    main()
