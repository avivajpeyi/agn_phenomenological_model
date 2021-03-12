"""
Given a GW event name, gets the NRsur and LVC result and plots the samples on a corner.

"""

from __future__ import print_function

import argparse
import glob
import os

from bilby.gw.result import CBCResult
from utils import (
    create_python_script_jobs
)


def plot_event(event_path, outdir):
    print(f"Plotting event: {event_path}")
    result = CBCResult.from_json(event_path, outdir)
    result.plot_corner(filename=os.path.join(outdir,
                                             os.path.basename(event_path).replace(
                                                 ".json", ".png")), priors=True)

    print("Saved fig")


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-dag", help="Make dag", action="store_true")
    parser.add_argument("--outdir", help="outdir for plot", type=str, default=".")
    parser.add_argument("--event-path", help="path", type=str,
                        default="GW150914")

    args = parser.parse_args()
    return args


def make_plotter_dag(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    paths = glob.glob("../bilby_pipe_jobs/out*/result/*result.json")
    args = [{"event-path": n, "outdir": outdir} for n in paths]
    create_python_script_jobs(
        main_job_name="corner_plotter",
        run_dir=".",
        python_script=os.path.abspath(__file__),
        job_args_list=args,
        job_names_list=[os.path.basename(p).replace(".json", "") for p in paths]
    )


def main():
    args = create_parser_and_read_args()
    if args.make_dag:
        make_plotter_dag(args.outdir)
    else:
        plot_event(args.event_path, args.outdir)


if __name__ == "__main__":
    main()
