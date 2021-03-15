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

PARAMS = {
    'chirp_mass': dict(latex_label="$M_{c}$", range=(5, 200)),
    'mass_1': dict(latex_label='$m_1^{\\mathrm{source}}$', range=(0, 200)),
    'mass_2': dict(latex_label='$m_2^{\\mathrm{source}}$', range=(0, 200)),
    'cos_tilt_1': dict(latex_label='$\\cos \\mathrm{tilt}_1$', range=(-1, 1)),
    'cos_tilt_2': dict(latex_label='$\\cos \\mathrm{tilt}_2$', range=(-1, 1)),
    'cos_theta_12': dict(latex_label='$\\cos \\theta_{12}$', range=(-1, 1)),
    'chi_p': dict(latex_label='$\\chi_p$', range=(0, 1)),
    'chi_eff': dict(latex_label='$\\chi_{\\rm{eff}}$', range=(-1, 1)),
    'luminosity_distance': dict(latex_label='$d_L$', range=(50, 20000)),
}

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
    smooth=0.5,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    # quantiles=(0.16, 0.84),
    show_titles=True,
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)


TRUTHS = {
    'cos_tilt_1': 0.9328944117496468,
    'cos_theta_12': -0.3196925946941709,
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


def overlaid_corner(samples_list, sample_labels, params,
                    samples_colors, fname="", title=None, truths={}):
    """Plots multiple corners on top of each other"""
    print(f"plotting {fname}")
    print(f"Cols in samples: {samples_list[0].columns.values}")
    # sort the sample columns
    samples_list = [s[params] for s in samples_list]

    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = max([len(s) for s in samples_list])

    CORNER_KWARGS.update(
        truths=[truths.get(k, None) for k in params],
        labels=[PARAMS[k]['latex_label'] for k in params],
        range=[PARAMS[k]['range'] for k in params],
        truth_color='lightgray',
        quantiles=None,
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


def copy_to_outdir(start_dir, end_dir):
    if not os.path.isdir(end_dir):
        os.makedirs(end_dir, exist_ok=True)
    for f in glob.glob(os.path.join(start_dir, "*.png")):
        shutil.copy(f, end_dir)


import re


def get_event_name(fname):
    name = re.findall(r"(\w*\d{6}[a-z]*)", fname)
    if len(name) == 0:
        name = re.findall(r"inj\d+", fname)
    if len(name) == 0:
        name = re.findall(r"data\d+", fname)
    if len(name) == 0:
        name = os.path.basename(fname).split(".")
    return name[0]


from collections import OrderedDict


def add_agn_samples_to_df(df):
    df['s1x'], df['s1y'], df['s1z'] = df['spin_1x'], df['spin_1y'], df['spin_1z']
    df['s2x'], df['s2y'], df['s2z'] = df['spin_2x'], df['spin_2y'], df['spin_2z']
    df['s1_dot_s2'] = \
        (df['s1x'] * df['s2x']) + \
        (df['s1y'] * df['s2y']) + \
        (df['s1z'] * df['s2z'])
    df['s1_mag'] = np.sqrt(
        (df['s1x'] * df['s1x']) +
        (df['s1y'] * df['s1y']) +
        (df['s1z'] * df['s1z']))
    df['s2_mag'] = np.sqrt(
        (df['s2x'] * df['s2x']) +
        (df['s2y'] * df['s2y']) +
        (df['s2z'] * df['s2z']))
    df['cos_theta_12'] = df['s1_dot_s2'] / (df['s1_mag'] * df['s2_mag'])
    return df


def main_inj_plotter():
    res_files = glob.glob("../finding_events_with_theta12/out*/res_dats/*result.dat")
    res_dfs = {}
    for r in res_files:
        res_inj_label = get_event_name(r)
        res = pd.read_csv(r, sep=' ')

        res = add_agn_samples_to_df(res)
        res['chi_eff'] = (res['spin_1z'] +
                          res['spin_2z'] *
                          res['mass_ratio']) / \
                         (1 + res['mass_ratio'])

        res['chi_1_in_plane'] = np.sqrt(
            res['spin_1x'] ** 2 + res['spin_1y'] ** 2
        )
        res['chi_2_in_plane'] = np.sqrt(
            res['spin_2x'] ** 2 + res['spin_2y'] ** 2
        )

        res['chi_p'] = np.maximum(
            res['chi_1_in_plane'],
            (4 * res['mass_ratio'] + 3) /
            (3 * res['mass_ratio'] + 4) * res['mass_ratio'] *
            res['chi_2_in_plane'])
        res['xi_spin'] = 1
        res[
            'luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(
            res['redshift'])
        res_dfs.update({res_inj_label: res})

    res_orderd = OrderedDict()
    injection_info = pd.read_csv("SNR_info.csv").to_dict('records')
    injection_truths = []
    for i in range(len(res_dfs)):
        label = f'inj{i}'
        try:
            res_orderd.update({label: res_dfs[label]})
            injection_truths.append(injection_info[i])
        except:
            pass
    cols = get_colors(len(res_dfs))

    for res_label, df, col, inj_truth in zip(list(res_orderd.keys()), list(res_orderd.values()), cols, injection_truths):
        truths = TRUTHS.copy()
        truths['mass_1'] = inj_truth['m1']
        truths['luminosity_distance'] = inj_truth['dist']
        overlaid_corner(
            samples_list=[df],
            sample_labels=[res_label +" SNR " + f"{inj_truth['snr']:.2f}"],
            params=["cos_theta_12", "chi_p", "chi_eff", "cos_tilt_1", "mass_1",
                    "luminosity_distance"],
            samples_colors=[col],
            fname=f"different_snrs_{res_label}",
        )

    copy_to_outdir(".",
                   "/home/avi.vajpeyi/public_html/agn_pop/finding_events_with_theta12")


if __name__ == '__main__':
    main_inj_plotter()
