# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

warnings.filterwarnings("ignore")

rcParams["font.size"] = 20
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 20
rcParams["font.size"] = 30
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 10
rcParams['axes.linewidth'] = 2.5
rcParams['axes.edgecolor'] = 'black'
rcParams['xtick.labelsize'] = 25
rcParams['xtick.major.size'] = 10.0
rcParams['xtick.minor.size'] = 5.0
rcParams['ytick.labelsize'] = 25
rcParams['ytick.major.size'] = 10.0
rcParams['ytick.minor.size'] = 5.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

HYPER_PARAM_VALS = {
    "alpha": 2.62,
    "beta": 1.26,
    "mmax": 86.73,
    "mmin": 4.5,
    "lam": 0.12,
    "mpp": 33.5,
    "sigpp": 5.09,
    "delta_m": 4.88,
    "mu_chi": 0.25,
    "sigma_chi": 0.03,
    "sigma_1": 0.5,
    "sigma_12": 2,
    "amax": 1.0,
    "lamb": 0.0,
}


def load_posteriors(run_dir, data_label):
    data_dir = os.path.join(run_dir, "data")
    posterior_file = os.path.join(data_dir, f"{data_label}.pkl")
    event_name_file = os.path.join(data_dir, f"{data_label}_posterior_files.txt")
    posteriors = pd.read_pickle(posterior_file)
    print(f"Loaded {len(posteriors)} posteriors")
    event_ids = list()
    with open(event_name_file, "r", ) as ff:
        for line in ff.readlines():
            event_ids.append(line.split(":")[0])
    return {event_id: posterior for event_id, posterior in zip(event_ids, posteriors)}


def load_true_values(injection_dat):
    true_vals = pd.read_csv(injection_dat, sep=" ").to_dict('records')
    return {f"inj{i}": true_vals[i] for i in range(len(true_vals))}


def plot_masses(posteriors, events, truths):
    print(f"Making box plot for {len(posteriors)} posteriors")
    mass_data = [post["mass_1"] for post in posteriors]
    mass_truths = [[t["mass_1_source"]] for t in truths]
    spin_data = [post["cos_theta_12"] for post in posteriors]
    spin_truths = [[t["cos_theta_12"]] for t in truths]
    snr_truths = [t["network_snr"] for t in truths]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(len(posteriors), 12))
    axs[0].violinplot(mass_data)
    axs[0].violinplot(mass_truths)
    axs[1].violinplot(spin_data)
    axs[1].violinplot(spin_truths)
    axs[2].bar(events, snr_truths)
    axs[0].hlines(y=HYPER_PARAM_VALS['mmax'], xmin=0, xmax=len(events) + 1)
    axs[0].hlines(y=HYPER_PARAM_VALS['mmin'], xmin=0, xmax=len(events) + 1)
    axs[0].set_ylabel("mass 1 source")
    axs[1].set_ylabel("cos theta 12")
    axs[2].set_ylabel("snr")
    axs[2].set_xticks(np.arange(1, len(events) + 1), events, rotation=90)
    axs[2].set_xlim(0, len(events) + 1)
    plt.tight_layout()
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    plt.savefig("pe_posteriors.png")
    plt.close(fig)


def get_data():
    posterior_dict = load_posteriors(run_dir="simulated_pop_inf_outdir/",
                                     data_label="posteriors")
    true_val_dict = load_true_values(
        injection_dat="bilby_pipe_jobs/injection_samples_all_params.dat")
    events, posteriors, truths = [], [], []
    for i in range(200):
        event_key = f"inj{i}"
        if event_key in posterior_dict:
            events.append(event_key)
            posteriors.append(posterior_dict[event_key])
            truths.append(true_val_dict[event_key])
    return posteriors, events, truths


def main():
    plot_masses(*get_data())


if __name__ == "__main__":
    main()
