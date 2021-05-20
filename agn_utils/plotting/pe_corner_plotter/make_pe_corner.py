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
    # 'mass_1_source': dict(latex_label='$m_1^{\\mathrm{source}}$', range=(0,200)),
    # 'mass_2_source': dict(latex_label='$m_2^{\\mathrm{source}}$', range=(0,200)),
    "cos_tilt_1": dict(latex_label="$\\cos \\mathrm{tilt}_1$", range=(-1, 1)),
    # 'cos_tilt_2': dict(latex_label='$\\cos \\mathrm{tilt}_2$', range=(-1,1)),
    "cos_theta_12": dict(latex_label="$\\cos \\theta_{12}$", range=(-1, 1)),
    "chi_p": dict(latex_label="$\\chi_p$", range=(0, 1)),
    "chi_eff": dict(latex_label="$\\chi_{\\rm{eff}}$", range=(-1, 1)),
    # 'luminosity_distance': dict(latex_label='$d_L$', range=(50,20000)),
}
PE_PRIOR = PriorDict(
    dictionary=dict(
        mass_1=Constraint(name="mass_1", minimum=10, maximum=200),
        mass_2=Constraint(name="mass_2", minimum=10, maximum=200),
        mass_ratio=Uniform(
            name="mass_ratio", minimum=0.125, maximum=1, latex_label="$q$"
        ),
        chirp_mass=Uniform(
            name="chirp_mass", minimum=5, maximum=200, latex_label="$M_{c}$"
        ),
        a_1=Uniform(name="a_1", minimum=0, maximum=0.99),
        a_2=Uniform(name="a_2", minimum=0, maximum=0.99),
        tilt_1=Sine(name="tilt_1"),
        tilt_2=Sine(name="tilt_2"),
        phi_12=Uniform(
            name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
        ),
        phi_jl=Uniform(
            name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic"
        ),
        luminosity_distance=PowerLaw(
            alpha=2,
            name="luminosity_distance",
            minimum=50,
            maximum=20000,
            unit="Mpc",
            latex_label="$d_L$",
        ),
        dec=Cosine(name="dec"),
        ra=Uniform(
            name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
        ),
        theta_jn=Sine(name="theta_jn"),
        psi=Uniform(name="psi", minimum=0, maximum=np.pi, boundary="periodic"),
        phase=Uniform(
            name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
        ),
    )
)


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-dag", help="Make dag", action="store_true")
    parser.add_argument(
        "--outdir", help="outdir for plot", type=str, default="."
    )
    parser.add_argument(
        "--event-name", help="path", type=str, default="GW150914"
    )

    args = parser.parse_args()
    return args


def make_plotter_dag(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    paths = glob.glob("../bilby_pipe_jobs/out*/result/*result.json")
    event_names = [get_event_name(p) for p in paths]
    args = [{"event-name": n, "outdir": outdir} for n in event_names]
    create_python_script_jobs(
        main_job_name="corner_plotter",
        run_dir="",
        python_script=os.path.abspath(__file__),
        job_args_list=args,
        job_names_list=event_names,
    )


def main():
    args = create_parser_and_read_args()
    if args.make_dag:
        make_plotter_dag(args.outdir)
    else:
        plot_event(
            event_name=args.event_name,
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
    if "redshift" not in df:
        df["redshift"] = get_redshift(df["luminosity_distance"])
    for ii in [1, 2]:
        df[f"mass_{ii}_source"] = df[f"mass_{ii}"] / (1 + df["redshift"])
    df = conversion.generate_spin_parameters(df)
    df = conversion.generate_mass_parameters(df)
    df = add_cos_theta_12_from_component_spins(df)
    if isinstance(df, pd.DataFrame):
        df = df[params]
    else:
        df = {p: df[p] for p in params}
    return df


def generate_prior_samples(params):
    df = pd.DataFrame(PE_PRIOR.sample(20000))
    return truncate_samples(df, params)


def load_true_values(params):
    injection_dat = "../bilby_pipe_jobs/injection_samples_all_params.dat"
    true_vals = pd.read_csv(injection_dat, sep=" ")
    true_vals = truncate_samples(true_vals, params).to_dict("records")
    return {f"inj{i}": true_vals[i] for i in range(len(true_vals))}


def load_res(event_path, params):
    result = CBCResult.from_json(event_path)
    pprint(result.injection_parameters)
    truths = truncate_samples(result.injection_parameters, params)
    posterior = truncate_samples(result.posterior, params)
    return posterior, truths


def get_redshift(dl):
    z_array = np.expm1(np.linspace(np.log(1), np.log(11), 1000))
    distance_array = Planck15.luminosity_distance(z_array).to(units.Mpc).value
    z_of_d = interp1d(distance_array, z_array)
    return z_of_d(dl)


def plot_event(event_name, outdir, params):
    event_path = glob.glob(
        f"../bilby_pipe_jobs/out*/result/{event_name}*result.json"
    )[0]
    print("Loading res")
    res_samples, true_sample = load_res(event_path, params)
    print("Loading prior")
    prior_samples = generate_prior_samples(params)

    fname = os.path.join(
        outdir, os.path.basename(event_path).replace(".json", ".png")
    )
    print("Plotting corner")
    overlaid_corner(
        [prior_samples, res_samples],
        ["Prior", f"Posterior", "Truth"],
        params=params,
        samples_colors=["lightgray", "tab:blue", "tab:orange"],
        fname=fname,
        truths=true_sample,
    )
    print(f"Saved {fname}")


def get_event_name(fname):
    name = re.findall(r"(\w*\d{6}[a-pred_z]*)", fname)
    if len(name) == 0:
        name = re.findall(r"inj\d+", fname)
    if len(name) == 0:
        name = re.findall(r"data\d+", fname)
    if len(name) == 0:
        name = os.path.basename(fname).split(".")
    return name[0]


if __name__ == "__main__":
    main()
