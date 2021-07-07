# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import TruncatedNormal
from bilby.gw.result import CBCResult
from tqdm.auto import tqdm


def update_style():
    plt.style.use(
        "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/55fab35b1c811dc3c0fdc6d169d67607e2c3edfc/publication.mplstyle")


def get_synthetic_bbh_posteriors_from_population_param(sig1=0, sig12=0, number_events=10):
    cos_1_pop = TruncatedNormal(mu=1, sigma=sig1, minimum=-1, maximum=1)
    cos_2_pop = TruncatedNormal(mu=1, sigma=sig12, minimum=-1, maximum=1)
    return dict(cos_theta_1=np.array([cos_1_pop.sample(1000) for _ in range(number_events)]),
                cos_theta_12=np.array([cos_2_pop.sample(1000) for _ in range(number_events)]))


def add_cdf_percentiles_to_ax(posteriors_list, ax, label=""):
    cumulative_prob = np.linspace(0, 1, len(posteriors_list[:, 0]))  # get len of all posteriors
    sorted_posterior = np.sort(posteriors_list, axis=0)  #
    data_05_percentile = np.quantile(sorted_posterior, 0.05, axis=1)  # get 0.05 CI from sorted_posterior
    data_95_percentile = np.quantile(sorted_posterior, 0.95, axis=1)  # get 0.95 CI from sorted_posterior
    ax.fill_betweenx(
        y=cumulative_prob,
        x1=data_05_percentile,
        x2=data_95_percentile,
        alpha=0.6, label=label
    )
    ax.plot(data_05_percentile, cumulative_prob, color='black', lw=0.5, alpha=0.5)
    ax.plot(data_95_percentile, cumulative_prob, color='black', lw=0.5, alpha=0.5)


def plot_posterior_predictive_check(data_sets, rhs_ax_labels):
    """Plots CDF plot_posterior_predictive_check."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for i, data in enumerate(data_sets):
        add_cdf_percentiles_to_ax(data['cos_theta_1'], axes[0])
        add_cdf_percentiles_to_ax(data['cos_theta_12'], axes[1], label=rhs_ax_labels[i])

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


def add_cos_thetas_from_component_spins(df):
    df["s1x"], df["s1y"], df["s1z"] = (
        df["spin_1x"].ravel(),
        df["spin_1y"].ravel(),
        df["spin_1z"].ravel(),
    )
    df["s2x"], df["s2y"], df["s2z"] = (
        df["spin_2x"],
        df["spin_2y"],
        df["spin_2z"],
    )
    df["s1_dot_s2"] = (
            (df["s1x"] * df["s2x"])
            + (df["s1y"] * df["s2y"])
            + (df["s1z"] * df["s2z"])
    )
    df["s1_mag"] = np.sqrt(
        (df["s1x"] * df["s1x"])
        + (df["s1y"] * df["s1y"])
        + (df["s1z"] * df["s1z"])
    )
    df["s2_mag"] = np.sqrt(
        (df["s2x"] * df["s2x"])
        + (df["s2y"] * df["s2y"])
        + (df["s2z"] * df["s2z"])
    )
    df["cos_theta_12"] = df["s1_dot_s2"] / (df["s1_mag"] * df["s2_mag"])
    df["cos_theta_1"] = np.cos(df["spin_1z"])
    return df


def load_dat_results():
    res = []
    files = glob.glob(
        "/home/avi.vajpeyi/projects/agn_phenomenological_model/simulated_events/simulated_event_samples/*.dat")
    for f in tqdm(files):
        df = pd.read_csv(f, ' ')
        df = add_cos_thetas_from_component_spins(df)
        res.append(df)
    return res


def load_bilby_results(regex):
    res = []
    files = glob.glob(regex)
    for f in tqdm(files):
        r = CBCResult.from_json(filename=f)
        pos = r.posterior.tail(1000)
        for i in [1, 2]:
            pos = pos.astype({f'spin_{i}x': 'float64', f'spin_{i}y': 'float64', f'spin_{i}z': 'float64'})
        df = add_cos_thetas_from_component_spins(pos)
        res.append(df)
    return res


def convert_bilby_res_to_usable_format(res):
    cos_theta_1_list = []
    cos_theta_12_list = []
    for r in res:
        cos_theta_1_list.append(r['cos_theta_1'].ravel())
        cos_theta_12_list.append(r['cos_theta_12'].ravel())
    print(len(cos_theta_12_list) == len(res))
    return dict(cos_theta_1=np.array(cos_theta_1_list), cos_theta_12=np.array(cos_theta_12_list))


def bilby_pe_main(POPS, TRUE, fname="posterior_predictive_check.png", title="Population A"):
    samps, labels = [], []
    for pop_name, regex in POPS.items():
        res = load_bilby_results(regex)
        posteriors_list = convert_bilby_res_to_usable_format(res)
        samps.append(posteriors_list)
        labels.append(pop_name)

        n = 100
        for s in TRUE.values():
            samps.append(get_synthetic_bbh_posteriors_from_population_param(s[0], s[1], number_events=n))

    update_style()
    plt.close('all')
    plot_posterior_predictive_check(samps, ["Observed", "Expected"])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname)


if __name__ == "__main__":
    bilby_pe_main(POPS=dict(
        popA="fixed_snr/*_a/pop_a*.json",
        # popB="fixed_snr/bp_pop_b/*.json"
    ),
        TRUE=dict(
            popA=[0.5, 3],
            # popB=[1, 0.25]
        ), title="Population A", fname="pop_a_cdf.png"
    )
    bilby_pe_main(POPS=dict(
        popB="fixed_snr/*_b/pop_b*.json"
    ),
        TRUE=dict(

            popB=[1, 0.25]
        ), title="Population B", fname="pop_b_cdf.png"
    )
