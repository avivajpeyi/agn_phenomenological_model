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

MIXED = '/home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/mixed_pop_outdir/result/mixed_pop_mass_c_iid_mag_afm_tilt_powerlaw_redshift_samples.dat'
AGN = '/home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_pop_outdir/result/agn_pop_mass_c_iid_mag_agn_tilt_powerlaw_redshift_samples.dat'
LVC = '/home/avi.vajpeyi/projects/agn_phenomenological_model/data/lvc_popinf/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json'
SIMULATED = '/home/avi.vajpeyi/projects/agn_phenomenological_model/simulated_events/simulated_pop_inf_outdir/result/simulated_pop_mass_c_iid_mag_agn_tilt_powerlaw_redshift_samples.dat'

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
    truth_color='tab:orange',
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


def get_colors(num_colors: int, alpha: Optional[float] = 0) -> List[List[float]]:
    """Get a list of colorblind samples_colors,
    :param num_colors: Number of samples_colors.
    :param alpha: The transparency
    :return: List of samples_colors. Each color is a list of [r, g, b, alpha].
    """
    palettes = ['colorblind', "ch:start=.2,rot=-.3"]
    cs = sns.color_palette(palettes[1], n_colors=num_colors)
    cs = [list(c) for c in cs]
    for i in range(len(cs)):
        cs[i].append(alpha)
    return cs


def overlaid_corner(samples_list, sample_labels, params,
                    samples_colors=[], fname="", title=None, truths={}):
    """Plots multiple corners on top of each other"""

    # sort the sample columns
    samples_list = [s[params] for s in samples_list]

    # get plot range, latex labels, colors and truths
    plot_range = [(PARAMS[p]['minimum'], PARAMS[p]['maximum']) for p in params]

    axis_labels = [PARAMS[p]['latex_label'] for p in params]

    if len(truths) == 0:
        truths = None
    else:
        truths = {k: truths[k] for k in params}

    if len(samples_colors) == 0:
        samples_colors = get_colors(len(samples_list))

    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = min([len(s) for s in samples_list])

    CORNER_KWARGS.update(
        range=plot_range,
        labels=axis_labels,
        truths=truths
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
            for i in range(n)
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


COLS = {label: c for label, c in zip(['agn', 'mix', 'lvc', 'sim'], get_colors(4))}


def read_agn_data():
    df = pd.read_csv(AGN, sep=' ')
    df['xi_spin'] = 1
    return df


def read_mixture_data():
    df = pd.read_csv(MIXED, sep=' ')
    return df


def read_lvc_data():
    df = bilby.result.read_in_result(LVC).posterior
    df['xi_spin'] = 0
    df['sigma_12'] = 0
    return df


def read_simulated_pop_data():
    df = pd.read_csv(SIMULATED, sep=' ')
    df['xi_spin'] = 1
    return df


def main():
    print("Plotting...")

    plot_params = ['sigma_1', "sigma_12", "xi_spin"]

    overlaid_corner(
        samples_list=[read_agn_data(), read_mixture_data()],
        sample_labels=["AGN", "Mixture Model"],
        params=plot_params,
        samples_colors=[COLS['agn'], COLS['mix']],
        fname="mix_and_agn.png"
    )

    plot_params = ['sigma_1', "sigma_12"]

    overlaid_corner(
        samples_list=[read_mixture_data()],
        sample_labels=["Mixture Model"],
        params=plot_params,
        samples_colors=[COLS['mix']],
        fname="only_mix.png",
    )

    overlaid_corner(
        samples_list=[read_agn_data()],
        sample_labels=["AGN"],
        params=plot_params,
        samples_colors=[COLS['agn']],
        fname="only_agn.png"
    )

    overlaid_corner(
        samples_list=[read_simulated_pop_data()],
        sample_labels=["PI", "Truths"],
        params=plot_params,
        truths=SIMULATED_TRUTHS,
        samples_colors=[COLS['sim'], "tab:orange"],
        fname="only_simulated.png"
    )

    overlaid_corner(
        samples_list=[read_lvc_data(), read_simulated_pop_data()],
        sample_labels=["LVC", "sim", "sim-truths"],
        params=plot_params,
        truths=SIMULATED_TRUTHS,
        samples_colors=[COLS['lvc'], COLS['sim'], "tab:orange"],
        fname="simulated_and_lvc.png"
    )
    print("done!")


if __name__ == '__main__':
    main()
