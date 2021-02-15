from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
from typing import List, Optional
# import seaborn as sns
# from ipywidgets import interactive, IntSlider, Layout
# import ipywidgets as widgets
import warnings
import corner
import numpy as np
import pandas as pd


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


def get_colors(num_colors: int, alpha: Optional[float]=1) -> List[List[float]]:
    """Get a list of colorblind colors,
    :param num_colors: Number of colors.
    :param alpha: The transparency
    :return: List of colors. Each color is a list of [r, g, b, alpha].
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
    quantiles=[0.5, 0.95],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
    )

def overlaid_corner(samples_list, sample_labels, axis_labels, plot_range, colors):
    """Plots multiple corners on top of each other"""
    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = min([len(s) for s in samples_list])

    CORNER_KWARGS.update(range=plot_range, labels=axis_labels)

    fig = corner.corner(
        samples_list[0],
        color=colors[0],
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



def read_agn_data():
    df= pd.read_csv(
        "agn_pop_outdir/result/agn_pop_mass_c_iid_mag_agn_tilt_powerlaw_redshift_samples.dat", 
        sep=' '
    )[['sigma_1', 'sigma_12']]
    df['xi_spin'] = 1
    return df

def read_mixture_data():
    df = pd.read_csv(
        "mixed_pop_outdir/result/mixed_pop_mass_c_iid_mag_afm_tilt_powerlaw_redshift_samples.dat", 
        sep=' '
    )[['sigma_1', 'sigma_12','xi_spin']]
    return df

print("Plotting...")
fig = overlaid_corner(
    samples_list=[read_agn_data(), read_mixture_data()], 
    sample_labels=["AGN", "Mixture Model"], 
    axis_labels=["$\\sigma_{1}$", "$\\sigma_{12}$", "$\\xi_{\\mathrm{spin}}$"], 
    plot_range=[(1e-2, 4), (1e-4, 10), (0,1)],
    colors=['blue', 'purple']
)
fig.savefig("both.png")
    
    
fig = overlaid_corner(
    samples_list=[read_mixture_data()], 
    sample_labels= ["Mixture Model"], 
    axis_labels=["$\\sigma_{1}$", "$\\sigma_{12}$", "$\\xi_{\\mathrm{spin}}$"], 
    plot_range=[(1e-2, 4), (1e-4, 10), (0,1)],
    colors=[ 'purple']
)
fig.savefig("mix.png")
    
# print("Plotting...")
# fig = overlaid_corner(
#     samples_list=[read_agn_data()[['sigma_1', 'sigma_12']], read_mixture_data()[['sigma_1', 'sigma_12']]], 
#     sample_labels=["AGN", "Mixture Model"], 
#     axis_labels=["$\\sigma_{1}$", "$\\sigma_{12}$"], 
#     plot_range=[(1e-2, 4), (1e-4, 1e1)],
#     colors=['blue', 'purple']
# )
# fig.savefig("test.png")