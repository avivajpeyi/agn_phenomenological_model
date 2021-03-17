from __future__ import print_function

import warnings
from typing import List, Optional

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

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

PARAMS = {
    'mass_1_source': dict(latex='$m_1^{\\mathrm{source}}$', range=(0, 100)),
    # 'mass_2_source': dict(latex='$m_2^{\\mathrm{source}}$', range=(0, 100)),
    # 'a_1':dict(latex=$a_1$', range=(0,100),
    'cos_tilt_1': dict(latex='$\\cos \\mathrm{tilt}_1$', range=(-1, 1)),
    # 'cos_tilt_2':dict(latex=$\\cos \\mathrm{tilt}_2$', range=(0,100),
    'cos_theta_12': dict(latex='$\\cos \\theta_{12}$', range=(-1, 1)),
    # 'phi_12':dict(latex=$\\phi_{12}$', range=(0,100),
    # 'phi_jl':dict(latex=$\\phi_{JL}$', range=(0,100),
    'luminosity_distance': dict(latex='$d_L$', range=(0, 20000)),
    # 'network_snr':dict(latex=$\\rho$, range=(0,100)'
    'log_snr': dict(latex='$\\rm{log}_{10}\ \\rho$)', range=(-1, 3)),
}


def get_colors(num_colors: int, alpha: Optional[float] = 1) -> List[List[float]]:
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


def overlaid_corner(samples_list, sample_labels, axis_labels, axis_range, colors):
    """Plots multiple corners on top of each other"""
    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = max([len(s) for s in samples_list])

    CORNER_KWARGS.update(labels=axis_labels, range=axis_range)

    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        weights=get_normalisation_weight(len(samples_list[0]), min_len),
        **CORNER_KWARGS,
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), min_len),
            color=colors[idx],
            **CORNER_KWARGS
        )

    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=20, frameon=False,
        bbox_to_anchor=(1, ndim), loc="upper right"
    )
    return fig


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)


FULL_SET = "samples.dat"
DOWNSAMPLED_SET = "injection_samples_all_params.dat"


def load_injection_file(dat, param):
    df = pd.read_csv(dat, sep=" ")
    df['mass_2_source'] = df["mass_1_source"] * df["mass_ratio"]
    df["log_snr"] = np.log10(df['network_snr'])
    return df[param]


param = [p for p in PARAMS.keys()]
full = load_injection_file(FULL_SET, param)
agn = load_injection_file(DOWNSAMPLED_SET, param)

fig = overlaid_corner(
    samples_list=[full],
    sample_labels=["Full set"],
    axis_labels=[PARAMS[p]['latex'] for p in param],
    axis_range=[PARAMS[p]['range'] for p in param],
    colors=["black"]
)
fig.savefig("plots/samples.png")

fig = overlaid_corner(
    samples_list=[full, agn],
    sample_labels=["Full set", "Selected"],
    axis_labels=[PARAMS[p]['latex'] for p in param],
axis_range=[PARAMS[p]['range'] for p in param],
    colors=["black", "green"]
)
fig.savefig("plots/injection_samples_wrt_all_samples.png")

fig = overlaid_corner(
    samples_list=[agn],
    sample_labels=["Full set", "Selected"],
    axis_labels=[PARAMS[p]['latex'] for p in param],
axis_range=[PARAMS[p]['range'] for p in param],
    colors=["green"]
)
fig.savefig("plots/selected_samples.png")
