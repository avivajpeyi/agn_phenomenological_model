# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import TruncatedNormal


def update_style():
    plt.style.use(
        "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/55fab35b1c811dc3c0fdc6d169d67607e2c3edfc/publication.mplstyle")


def get_data(hyper_param=0, number_events=10):
    trunc_norm = TruncatedNormal(mu=1, sigma=hyper_param, minimum=-1, maximum=1)
    data = np.array([trunc_norm.sample(1000) for _ in range(number_events)])
    return dict(cos_theta_1=data, cos_theta_12=data)


def add_cdf_percentiles_to_ax(data, ax, label):
    cumulative_prob = np.linspace(0, 1, len(data[:, 0]))
    data_05_percentile = np.quantile(np.sort(data, axis=0), 0.05, axis=1)
    data_95_percentile = np.quantile(np.sort(data, axis=0), 0.95, axis=1)
    ax.fill_betweenx(
        y=cumulative_prob,
        x1=data_05_percentile,
        x2=data_95_percentile,
        alpha=0.6, label=label
    )
    ax.plot(data_05_percentile, cumulative_prob, color='black', lw=0.5, alpha=0.5)
    ax.plot(data_95_percentile, cumulative_prob, color='black', lw=0.5, alpha=0.5)


def plot(data_sets, sigma1_vals, sigma12_vals):
    """Plots CDF plot."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for i, data in enumerate(data_sets):
        add_cdf_percentiles_to_ax(data['cos_theta_1'], axes[0], label=r"$\sigma_1=" + f"{sigma1_vals[i]}$")
        add_cdf_percentiles_to_ax(data['cos_theta_12'], axes[1], label=r"$\sigma_{12}=" + f"{sigma12_vals[i]}$")

    for i, ax in enumerate(axes):
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 1])

        if (i == 0):
            ax.set_xlabel(r"$\cos\ \theta_1$")
            ax.set_ylabel("Cumulative Probability")
        else:
            ax.set_xlabel(r"$\cos\ \theta_{12}$")
            ax.set_yticklabels([])
        ax.legend(fontsize='small')


import glob
import pandas as pd
from agn_utils.plotting.pe_corner_plotter.make_pe_corner import add_cos_theta_12_from_component_spins, conversion
from tqdm.auto import tqdm


def load_bilby_results():
    res = []
    files = glob.glob(
        "/home/avi.vajpeyi/projects/agn_phenomenological_model/simulated_events/simulated_event_samples/*.dat")
    for f in tqdm(files):
        df = pd.read_csv(f, ' ')
        df = add_cos_theta_12_from_component_spins(df)
        df = conversion.generate_spin_parameters(df)
        res.append(df)
    return res


def convert_bilby_res_to_usable_format(res):
    cos_theta_1_list = []
    cos_theta_12_list = []
    for r in res:
        cos_theta_1_list.append(r['cos_theta_1'])
        cos_theta_12_list.append(r['cos_theta_12'])
    return dict(cos_theta_1=cos_theta_1_list, cos_theta_12_list=cos_theta_12_list)


def bilby_pe_main():
    res = load_bilby_results()
    data = convert_bilby_res_to_usable_format(res)
    update_style()
    plt.close('all')
    plot([data], [0.5], [2])
    plt.suptitle(f"{200} BBH events", y=0.99)
    plt.tight_layout()
    plt.savefig(f"{200}_events.png")


def fake_events_plots():
    sigma_vals = [0.1, 0.5, 4]
    update_style()
    for n in [50, 100, 500]:
        plt.close('all')
        plot([get_data(i, n) for i in sigma_vals], sigma_vals, sigma_vals)
        plt.suptitle(f"{n} BBH events", y=0.99)
        plt.tight_layout()
        plt.savefig(f"{n}_events.png")


if __name__ == "__main__":
    try:
        bilby_pe_main()
    except Exception as e:
        print(e)
        fake_events_plots()
