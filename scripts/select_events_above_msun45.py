# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from configargparse import Namespace
from corner import quantile
from gwpopulation_pipe.data_collection import get_event_name, load_o3a_events
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


def load_posteriors(run_dir=".", o3a_samples_regex=""):
    """
    :return {even_name: samples df}
    """
    posts = load_o3a_events(args=Namespace(
        o3a_samples_regex="/home/shanika.galaudage/O3/population/o3a_pe_samples_release/S*.h5",
        run_dir=""
    ))
    return {get_event_name(f): samples for f, samples in posts.items()}


def is_quant_above_mmin(masses, quantiles):
    return min(quantile(masses, q=quantiles)) > MIN_MASS


def plot_masses(posteriors, events, ignore_list):
    print(f"Making box plot for {len(posteriors)} posteriors")
    data = [post["mass_1"] for post in posteriors]
    fig = plt.figure(figsize=(len(posteriors), 5))
    violin_parts = plt.violinplot(data, quantiles=QUANTILES)
    for pc, event in zip(violin_parts['bodies'], events):
        if event in ignore_list:
            pc.set_facecolor('red')

    plt.hlines(y=MIN_MASS, xmin=0, xmax=len(events) + 1)
    plt.ylabel("mass 1 source")
    plt.xticks(np.arange(1, len(events) + 1), events, rotation=90)
    plt.xlim(0, len(events) + 1)
    plt.tight_layout()
    plt.grid()
    plt.savefig("mass_posteriors_above_mmin_threshold.png")
    plt.close(fig)


def get_data():
    posterior_dict = load_posteriors()

    events, posteriors, ignore_list = [], [], []
    for event, post in posterior_dict.items():
        valid = is_quant_above_mmin(post.mass_1, QUANTILES)
        if not valid:
            ignore_list.append(event)
        events.append(event)
        posteriors.append(post)
    print(f"Ignore: {ignore_list}")
    return posteriors, events, ignore_list


def main():
    plot_masses(*get_data())


if __name__ == "__main__":
    main()
