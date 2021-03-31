from __future__ import print_function

import glob
import os
import shutil
import warnings
from typing import List, Optional

import bilby
import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner

MIXED = "../result_files/mix.dat"
AGN = "../result_files/agn.dat"
LVC = "../result_files/lvc.json"
SIMULATED = "../result_files/sim.dat"
HIGH_MASS = "../result_files/high_mass_agn.dat"

COLS = dict(
    mix='seagreen',
    lvc='mediumpurple',
    sim='dodgerblue',
    agn='orangered',
    truths='crimson'
)


def read_agn_data():
    df = pd.read_csv(AGN, sep=' ')
    df['xi_spin'] = 1
    return df


def read_mixture_data():
    """
    params:
    ['alpha' 'beta' 'mmax' 'mmin' 'lam' 'mpp' 'sigpp' 'delta_m'
    'mu_chi' 'sigma_chi' 'sigma_1' 'sigma_12']"""
    df = pd.read_csv(MIXED, sep=' ')
    return df


def read_lvc_data():
    """
    params:
    ['alpha' 'beta' 'mmax' 'mmin' 'lam' 'mpp' 'sigpp' 'delta_m'
    'mu_chi' 'sigma_chi' 'xi_spin' 'sigma_spin' 'amax' 'lamb']"""
    df = bilby.result.read_in_result(LVC).posterior
    df['xi_spin'] = 0
    df['sigma_12'] = 1e-4
    df['sigma_1'] = df['sigma_spin']
    df['sigma_2'] = df['sigma_spin']
    trash = ['rate', 'log_10_rate', 'surveyed_hypervolume', 'n_effective', 'selection',
             'log_prior', 'log_likelihood']
    df = df.drop(trash, axis=1)
    return df


def read_simulated_pop_data():
    df = pd.read_csv(SIMULATED, sep=' ')
    df['xi_spin'] = 1
    return df


def read_high_mass_data():
    df = pd.read_csv(HIGH_MASS, sep=' ')
    df['xi_spin'] = 1
    return df


def main():
    print("Plotting...")

    agn_data = read_agn_data()
    mix_data = read_mixture_data()
    sim_data = read_simulated_pop_data()
    lvc_data = read_lvc_data()
    high_mass_data = read_high_mass_data()

    plot_params = ['sigma_1', "sigma_12", "xi_spin"]

    overlaid_corner(
        samples_list=[agn_data, mix_data],
        sample_labels=["AGN", "Mixture Model"],
        params=plot_params,
        samples_colors=[COLS['agn'], COLS['mix']],
        fname="mix_and_agn.png"
    )

    plot_params = ['sigma_1', "sigma_12"]

    overlaid_corner(
        samples_list=[mix_data],
        sample_labels=["Mix"],
        params=plot_params,
        samples_colors=[COLS['mix']],
        fname="only_mix.png",
    )

    overlaid_corner(
        samples_list=[agn_data],
        sample_labels=["AGN"],
        params=plot_params,
        samples_colors=[COLS['agn']],
        fname="only_agn.png"
    )

    overlaid_corner(
        samples_list=[high_mass_data],
        sample_labels=["High-mass"],
        params=plot_params,
        samples_colors=[COLS['agn']],
        fname="only_highmass_agn.png"
    )

    overlaid_corner(
        samples_list=[sim_data],
        sample_labels=["Sim", "Truths"],
        params=plot_params,
        truths=SIMULATED_TRUTHS,
        samples_colors=[COLS['sim'], COLS['truths']],
        fname="only_simulated.png"
    )

    overlaid_corner(
        samples_list=[lvc_data, sim_data],
        sample_labels=["LVC", "Sim", "Truths"],
        params=plot_params,
        truths=SIMULATED_TRUTHS,
        samples_colors=[COLS['lvc'], COLS['sim'], COLS['truths']],
        fname="simulated_and_lvc.png"
    )

    plot_params = sorted(list(
        set(sim_data.columns.values).intersection(set(lvc_data.columns.values))))
    plot_params.remove("xi_spin")
    plot_params.remove("sigma_12")

    overlaid_corner(
        samples_list=[lvc_data, sim_data],
        sample_labels=["LVC", "Sim", "Truths"],
        params=plot_params,
        truths=SIMULATED_TRUTHS,
        samples_colors=[COLS['lvc'], COLS['sim'], COLS['truths']],
        fname="simulated_and_lvc_all.png"
    )

    overlaid_corner(
        samples_list=[lvc_data, agn_data],
        sample_labels=["LVC", "AGN"],
        params=plot_params,
        samples_colors=[COLS['lvc'], COLS['agn']],
        fname="lvc_and_agn.png"
    )

    overlaid_corner(
        samples_list=[lvc_data, high_mass_data],
        sample_labels=["LVC", "High-Mass"],
        params=plot_params,
        samples_colors=[COLS['lvc'], COLS['agn']],
        fname="lvc_and_highmass_agn.png"
    )

    overlaid_corner(
        samples_list=[lvc_data],
        sample_labels=["LVC", "Truths"],
        params=plot_params,
        samples_colors=[COLS['lvc'], COLS['truths']],
        truths=SIMULATED_TRUTHS,
        fname="lvc_with_my_injection.png"
    )

    copy_to_outdir("", "/home/avi.vajpeyi/public_html/agn_pop/pop_inf/")

    print("done!")


def copy_to_outdir(start_dir, end_dir):
    if not os.path.isdir(end_dir):
        os.makedirs(end_dir, exist_ok=True)
    for f in glob.glob(os.path.join(start_dir, "*.png")):
        shutil.copy(f, end_dir)

if __name__ == '__main__':
    main()
