# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from corner import quantile
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

MIN_MASS = 45
QUANTILES = [.16, .84]

import pandas as pd


def load_posteriors(psterior_pkl, posterior_fname_file):
    posteriors = pd.read_pickle(psterior_pkl)
    print(f"Loaded {len(posteriors)} posteriors")
    event_ids = list()
    with open(posterior_fname_file, "r") as ff:
        for line in ff.readlines():
            event_ids.append(line.split(":")[0])
    return posteriors, event_ids


def is_quant_above_mmin(masses, quantiles):
    return min(quantile(masses, q=quantiles)) > MIN_MASS


def plot_masses(posteriors, events, ignore_list):
    print(f"Making box plot for {len(posteriors)} posterimgact ../iors")
    data = [post["mass_1"] for post in posteriors]
    fig = plt.figure(figsize=(len(posteriors), 5))
    violin_parts = plt.violinplot(data,
                                  quantiles=[QUANTILES for _ in range(len(posteriors))])
    for pc, event in zip(violin_parts['bodies'], events):
        if event in ignore_list:
            pc.set_facecolor('red')
            pc.set_edgecolor('red')
    plt.ylim(0, 100)
    plt.hlines(y=MIN_MASS, xmin=0, xmax=len(events) + 1, colors="gray", linestyles="dashed")
    plt.ylabel("mass 1 source")
    plt.xticks(np.arange(1, len(events) + 1), events, rotation=90)
    plt.xlim(0, len(events) + 1)
    plt.tight_layout()
    plt.grid()
    plt.savefig("mass_posteriors_above_mmin_threshold.png")
    plt.close(fig)


def get_data():
    posteriors, events = load_posteriors(
        psterior_pkl="/home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_pop_outdir/data/posteriors.pkl",
        posterior_fname_file="/home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_pop_outdir/data/posteriors_posterior_files.txt")
    ignore_list = []
    for event, post in zip(events, posteriors):
        valid = is_quant_above_mmin(post.mass_1, QUANTILES)
        if not valid:
            ignore_list.append(event)
    print(f"Ignore {len(ignore_list)}/{len(events)}: {ignore_list}")
    return posteriors, events, ignore_list


def main():
    plot_masses(*get_data())


if __name__ == "__main__":
    main()
