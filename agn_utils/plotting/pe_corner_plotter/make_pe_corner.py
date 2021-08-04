"""
Given a GW event name, gets the NRsur and LVC result and plots the samples on a corner.

"""

from __future__ import print_function

import argparse
import glob
import os
import re
from pprint import pprint

import numpy as np
import pandas as pd
from astropy import units
from astropy.cosmology import Planck15
from bilby.core.prior import Constraint, Cosine, PowerLaw, Sine, Uniform
from bilby.gw import conversion
from bilby.gw.prior import PriorDict
from bilby.gw.result import CBCResult
from scipy.interpolate import interp1d

from ..overlaid_corner_plotter import overlaid_corner
from ...batch_processing import create_python_script_jobs
from ...bbh_population_generators.calculate_extra_bbh_parameters import add_cos_theta_12_from_component_spins

PARAMS = {
    "chirp_mass": dict(latex_label="$M_{c}$", range=(5, 200)),
    "cos_tilt_2": dict(latex_label="$\\cos \\mathrm{tilt}_2$", range=(-1, 1)),
    "cos_tilt_1": dict(latex_label="$\\cos \\mathrm{tilt}_1$", range=(-1, 1)),
    "cos_theta_12": dict(latex_label="$\\cos \\theta_{12}$", range=(-1, 1)),
    "phi_12": dict(latex_label="$\\phi_{12}$", range=(0, 2*np.pi)),
    "chi_p": dict(latex_label="$\\chi_p$", range=(0, 1)),
    "chi_eff": dict(latex_label="$\\chi_{\\rm{eff}}$", range=(-1, 1)),
    "snr": dict(latex_label="$\\mathrm{SNR}$", range=(-1, 1)),
}


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-dag", help="Make dag", action="store_true")
    parser.add_argument(
        "--outdir", help="outdir for plots", type=str, default="."
    )
    parser.add_argument(
        "--event-path", help="path", type=str, default=""
    )
    parser.add_argument(
        "--regex", help="path", type=str, default=""
    )

    args = parser.parse_args()
    return args


def make_plotter_dag(outdir, regex):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    paths = glob.glob(regex)
    event_names = [get_event_name(p) for p in paths]
    args = [{"event-path": p, "outdir": outdir} for p in paths]
    create_python_script_jobs(
        main_job_name="corner_plotter",
        python_script=os.path.abspath(__file__),
        job_args_list=args,
        job_names_list=event_names,
    )

def main():
    args = create_parser_and_read_args()
    if args.make_dag:
        make_plotter_dag(args.outdir, args.regex)
    else:
        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)
        plot_event(
            event_path=args.event_path,
            outdir=args.outdir,
            params=[p for p in PARAMS.keys()],
        )


def truncate_samples(df, params):
    if "mass_1" not in df:
        df["total_mass"] = conversion.chirp_mass_and_mass_ratio_to_total_mass(
            df["chirp_mass"], df["mass_ratio"]
        )
        (
            df["mass_1"],
            df["mass_2"],
        ) = conversion.total_mass_and_mass_ratio_to_component_masses(
            df["mass_ratio"], df["total_mass"]
        )
    if "reference_frequency" not in df:
        df["reference_frequency"] = 20

    if "H1_optimal_snr" in df:
        matched_filter_snr = np.sqrt(((np.abs(df["H1_matched_filter_snr"])**2) + (np.abs(df["L1_matched_filter_snr"])**2)))
        optimal_snr = np.sqrt(((np.abs(df["H1_optimal_snr"])**2) + (np.abs(df["L1_optimal_snr"])**2)))
        df['matched_filter_snr'] = matched_filter_snr
        df['snr'] = optimal_snr
    else:
        df['snr'] = df['network_snr']

    df = conversion.generate_spin_parameters(df)
    df = conversion.generate_mass_parameters(df)
    df = add_cos_theta_12_from_component_spins(df)
    if isinstance(df, pd.DataFrame):
        df = df[params]
    else:
        df = {p: df[p] for p in params}
    return df



def load_res(event_path, params):
    result = CBCResult.from_json(event_path)
    truths = truncate_samples(result.injection_parameters, params)
    posterior = truncate_samples(result.posterior, params)
    pprint(truths)
    return posterior, truths



def plot_event(event_path, outdir, params):
    print(f"Loading {os.path.basename(event_path)}")
    res_samples, true_sample = load_res(event_path, params)
    fname = os.path.join(
        outdir, os.path.basename(event_path).replace(".json", ".png")
    )
    print("Plotting corner")
    overlaid_corner(
        [res_samples],
        [get_event_name(event_path).replace("_", " "), "Truth"],
        params=params,
        samples_colors=["tab:blue", "tab:orange"],
        fname=fname,
        truths=true_sample,
        ranges=None
    )
    print(f"Saved {fname}")


def get_event_name(fname):
    name = re.findall(r"(\w*\d{6}[a-pred_z]*)", fname)
    if len(name) == 0:
        name = re.findall(r"inj\d+", fname)
    if len(name) == 0:
        name = re.findall(r"posteriors_list\d+", fname)
    if len(name) == 0:
        name = os.path.basename(fname).split(".")
    return name[0]


if __name__ == "__main__":
    main()
