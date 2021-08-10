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
from matplotlib.colors import to_rgba


warnings.filterwarnings("ignore")


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
    "sigma_1_list": 0.5,
    "sigma_12_list": 2,
    "amax": 1.0,
    "lamb": 0.0,
}


def load_posteriors(run_dir, data_label):
    data_dir = os.path.join(run_dir, "posteriors_list")
    posterior_file = os.path.join(data_dir, f"{data_label}.pkl")
    event_name_file = os.path.join(
        data_dir, f"{data_label}_posterior_files.txt"
    )
    posteriors = pd.read_pickle(posterior_file)
    print(f"Loaded {len(posteriors)} posteriors")
    event_ids = list()
    with open(
        event_name_file,
        "r",
    ) as ff:
        for line in ff.readlines():
            event_ids.append(line.split(":")[0])
    return {
        event_id: posterior
        for event_id, posterior in zip(event_ids, posteriors)
    }


def load_true_values(injection_dat):
    true_vals = pd.read_csv(injection_dat, sep=" ").to_dict("records")
    return {f"inj{i}": true_vals[i] for i in range(len(true_vals))}


def plot_masses(posteriors, events, truths):
    print(f"Making box plot_posterior_predictive_check for {len(posteriors)} posteriors")
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
    axs[0].hlines(y=HYPER_PARAM_VALS["mmax"], xmin=0, xmax=len(events) + 1)
    axs[0].hlines(y=HYPER_PARAM_VALS["mmin"], xmin=0, xmax=len(events) + 1)
    axs[0].set_ylabel("mass 1 source")
    axs[1].set_ylabel("cos tilt 12")
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
    posterior_dict = load_posteriors(
        run_dir="simulated_pop_inf_outdir/", data_label="posteriors"
    )
    true_val_dict = load_true_values(
        injection_dat="bilby_pipe_jobs/injection_samples_all_params.dat"
    )
    events, posteriors, truths = [], [], []
    for i in range(200):
        event_key = f"inj{i}"
        if event_key in posterior_dict:
            events.append(event_key)
            posteriors.append(posterior_dict[event_key])
            truths.append(true_val_dict[event_key])
    return posteriors, events, truths


def change_violin_col(violin_part, col, idx):
    violin_part['bodies'][idx].set_facecolor(col)
    violin_part['bodies'][idx].set_edgecolor(col)
    violin_part['cquantiles'].set_color('k')
    for d in ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians']:
        if d in violin_part:
            current_colors = violin_part[d].get_colors()
            if len(current_colors)==1:
                current_col = current_colors[0]
                num_cols  = len(violin_part['bodies'])
                current_colors = [current_col for _ in range(num_cols)]
            current_colors[idx] = to_rgba(col)
            violin_part[d].set_colors(current_colors)



def simple_violin_plotter(dat, fname, dat_labs=['cos_tilt_1', 'cos_theta_12'], labels=[r"$\cos\theta_1$", r"$\cos\theta_{12}$"]):
    """dat: {posteriors:dict(label:lists of posteriors), trues:dict(label:list of trues), labels: list of labels"""
    num_events = len(dat['labels'])
    quantiles = [ [0.16, 0.84] for _ in range(num_events)]
    fig, axs = plt.subplots(3,1, sharex=True, figsize=(16, 3*len(labels)))

    for ax, dat_lab, label in zip(axs, dat_labs, labels):
        posteriors = list(dat["posteriors"][dat_lab])
        trues =  [[i] for i in dat["trues"][dat_lab]]
        trues_in_quants = [true_in_quant(p, t, quantiles[0]) for p,t in zip(posteriors, trues)]
        violin_pts = ax.violinplot(posteriors, quantiles=quantiles)
        for i, in_quant in enumerate(trues_in_quants):
            if not in_quant:
                change_violin_col(violin_pts, "tab:red", i)

        true_vpts = ax.violinplot(trues, widths=[0.9 for _ in range(num_events)])
        for b in true_vpts['bodies']:
            b.set_facecolor('orange')
        for d in ['cmaxes','cmins','cbars']:
            true_vpts[d].set_color('orange')
            # true_vpts[d].set_lw(2)
        ax.set_ylabel(label)
        # ax.set_ylim(-1,1)

    tick_labels = [i.replace('pop a highsnr ', '') for i in dat['labels']]
    format_x_axes(axs, tick_labels)

    plt.suptitle(fname.replace(".png", "").replace("_", " "))
    plt.tight_layout()
    plt.savefig(fname)

def format_x_axes(axs, labels):
    for ax in axs:
        ax.xaxis.set_tick_params(direction='inout')
        ax.xaxis.set_ticks_position('both')
        ax.minorticks_off()
    bottom = len(axs)-1
    axs[bottom].set_xticks(np.arange(1, len(labels) + 1))
    axs[bottom].set_xlabel("Events")
    axs[bottom].set_xticklabels(labels, rotation=70)
    axs[bottom].set_xlim(0.25, len(labels) + 0.75)

def long_substr(data):
    substrs = lambda x: {x[i:i+j] for i in range(len(x)) for j in range(len(x) - i + 1)}
    s = substrs(data[0])
    for val in data[1:]:
        s.intersection_update(substrs(val))
    return max(s, key=len)


def true_in_quant(post, true, quant ):
    quantiles = np.quantile(post, quant)
    return quantiles[0] <= true <= quantiles[1]


def main():
    plot_masses(*get_data())


### is quant above min
def is_quant_above_mmin(masses, quantiles, mmin):
    return min(np.quantile(masses, q=quantiles)) > mmin


def plot_masses(posteriors, events, ignore_list, mmin):
    print(f"Making box plot_posterior_predictive_check for {len(posteriors)} posteriors")
    data = [post["mass_1"] for post in posteriors]
    fig = plt.figure(figsize=(len(posteriors), 5))
    violin_parts = plt.violinplot(
        data, quantiles=[QUANTILES for _ in range(len(posteriors))]
    )
    for idx, event in enumerate(events):
        if event in ignore_list:
            adjust_colors_for_violin(violin_parts, idx, color="red")

    plt.ylim(0, 100)
    plt.hlines(
        y=mmin,
        xmin=0,
        xmax=len(events) + 1,
        colors="gray",
        linestyles="dashed",
    )
    plt.ylabel("mass 1 source")
    plt.xticks(np.arange(1, len(events) + 1), events, rotation=90)
    plt.xlim(0, len(events) + 1)
    plt.tight_layout()
    plt.grid()
    plt.suptitle(f"Mass Min = {mmin} (Solar mass)")
    plt.savefig(f"mass_posteriors_above_{mmin}.png")
    plt.close(fig)


def adjust_colors_for_violin(violin_parts, idx, color):
    violin_parts["bodies"][idx].set_facecolor(color)
    violin_parts["bodies"][idx].set_edgecolor(color)
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians', 'bodies'):
    #     violin_parts[partname][idx].set_edgecolor("red")


def get_data(mmin):
    posteriors, events = load_posteriors(
        psterior_pkl="/home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_pop_outdir/posteriors_list/posteriors.pkl",
        posterior_fname_file="/home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_pop_outdir/posteriors_list/posteriors_posterior_files.txt",
    )
    ignore_list = []
    for event, post in zip(events, posteriors):
        valid = is_quant_above_mmin(post.mass_1, QUANTILES, mmin)
        if not valid:
            ignore_list.append(event)
    print(f"MMIN={mmin}")
    print(f"Ignore {len(ignore_list)}/{len(events)}: {ignore_list}")
    print(f"Keep {len(events)-len(ignore_list)}/{len(events)}")
    return posteriors, events, ignore_list


def main():
    for mmin in [45, 40, 35]:
        posteriors, events, ignore_list = get_data(mmin)
        plot_masses(posteriors, events, ignore_list, mmin)


if __name__ == "__main__":
    main()
