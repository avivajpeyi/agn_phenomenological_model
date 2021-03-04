from __future__ import print_function

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
import os
import shutil

import glob
MIXED = "../result_files/mix.dat"
AGN = "../result_files/agn.dat"
LVC = "../result_files/lvc.json"
SIMULATED = "../result_files/sim.dat"

warnings.filterwarnings("ignore")

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

CORNER_KWARGS = dict(
    smooth=0.99,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    quantiles=(0.16, 0.84),
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)

SIMULATED_TRUTHS = {
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

PARAMS = {
    'alpha': dict(minimum=-4, maximum=12, latex_label='$\\alpha$'),
    'beta': dict(minimum=-4, maximum=12, latex_label='$\\beta_{q}$'),
    'mmax': dict(minimum=30, maximum=100, latex_label='$m_{\\max}$'),
    'mmin': dict(minimum=2, maximum=10, latex_label='$m_{\\min}$'),
    'lam': dict(minimum=0, maximum=1, latex_label='$\\lambda_{m}$'),
    'mpp': dict(minimum=20, maximum=50, latex_label='$\\mu_{m}$'),
    'sigpp': dict(minimum=1, maximum=10, latex_label='$\\sigma_{m}$'),
    'delta_m': dict(minimum=0, maximum=10, latex_label='$\\delta_{m}$'),
    'amax': dict(minimum=1, maximum=1, latex_label="$\\a_{max}$"),
    'mu_chi': dict(minimum=0, maximum=1, latex_label='$\\mu_{\\chi}$'),
    'sigma_chi': dict(minimum=0, maximum=0.25, latex_label='$\\sigma^{2}_{\\chi}$'),
    'alpha_chi': dict(minimum=1, maximum=100000, latex_label='$\\alpha_{\\chi}$'),
    'beta_chi': dict(minimum=1, maximum=100000, latex_label='$\\beta{\\chi}$'),
    'sigma_1': dict(minimum=0.01, maximum=4, latex_label='$\\sigma_{1}$'),
    'sigma_2': dict(minimum=0.01, maximum=4, latex_label='$\\sigma_{2}$'),
    'sigma_12': dict(minimum=0.0001, maximum=4, latex_label='$\\sigma_{12}$'),
    'xi_spin': dict(minimum=0, maximum=1, latex_label='$\\xi$')
}


def get_colors(num_colors: int, alpha: Optional[float] = 1) -> List[List[float]]:
    """Get a list of colorblind samples_colors,
    :param num_colors: Number of samples_colors.
    :param alpha: The transparency
    :return: List of samples_colors. Each color is a list of [r, g, b, alpha].
    """
    palettes = ['colorblind', "ch:start=.2,rot=-.3"]
    cs = sns.color_palette(palettes[0], n_colors=num_colors)
    cs = [list(c) for c in cs]
    for i in range(len(cs)):
        cs[i].append(alpha)
    return cs


COLS = dict(
    mix='seagreen',
    lvc='mediumpurple',
    sim='dodgerblue',
    agn='orangered',
    truths='crimson'
)


# COLS = {label: c for label, c in
#         zip(['mix', 'lvc', 'sim', 'agn', 'truths'], get_colors(5))}
# COLS['truths'] = 'black'


def overlaid_corner(samples_list, sample_labels, params,
                    samples_colors, fname="", title=None, truths={}):
    """Plots multiple corners on top of each other"""
    print(f"plotting {fname}")
    # sort the sample columns
    samples_list = [s[params] for s in samples_list]

    # get plot range, latex labels, colors and truths
    plot_range = [(PARAMS[p]['minimum'], PARAMS[p]['maximum']) for p in params]

    axis_labels = [PARAMS[p]['latex_label'] for p in params]

    if len(truths) == 0:
        truths = None
    else:
        truths = [truths[k] for k in params]


    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = max([len(s) for s in samples_list])

    CORNER_KWARGS.update(
        range=plot_range,
        labels=axis_labels,
        truths=truths,
        truth_color=COLS['truths'],
    )

    fig = corner.corner(
        samples_list[0],
        color=samples_colors[0],
        **CORNER_KWARGS,
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), min_len),
            color=samples_colors[idx],
            **CORNER_KWARGS
        )

    plt.legend(
        handles=[
            mlines.Line2D([], [], color=samples_colors[i], label=sample_labels[i])
            for i in range(len(sample_labels))
        ],
        fontsize=20, frameon=False,
        bbox_to_anchor=(1, ndim), loc="upper right"
    )
    if title:
        fig.suptitle(title, y=0.97)
        fig.subplots_adjust(top=0.75)
    fig.savefig(fname)
    plt.close(fig)


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)


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


def main():
    print("Plotting...")

    agn_data = read_agn_data()
    mix_data = read_mixture_data()
    sim_data = read_simulated_pop_data()
    lvc_data = read_lvc_data()

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
        samples_list=[lvc_data],
        sample_labels=["LVC", "Truths"],
        params=plot_params,
        samples_colors=[COLS['lvc'], COLS['truths']],
        truths=SIMULATED_TRUTHS,
        fname="lvc_with_my_injection.png"
    )

    dest_folder = "/home/avi.vajpeyi/public_html/agn_pop/pop_inf/"
    if os.path.isdir(dest_folder):
        for f in glob.glob("*.png"):
            shutil.copy(f, dest_folder)

    print("done!")


if __name__ == '__main__':
    main()
