from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
from typing import List, Optional
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


SIMULATED_POP_SAMPLES = "/home/avi.vajpeyi/projects/agn_phenomenological_model/simulated_events/simulated_pop_inf_outdir/result/simulated_pop_mass_c_iid_mag_agn_tilt_powerlaw_redshift_samples.dat"

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
    "sigma_1": 0.5,
    "sigma_12": 2,
    "amax": 1.0,
    "lamb": 0.0,
}

LATEX ={
    "alpha": "$\\alpha$",
    "beta": "$\\beta$",
    "mmax": "$m_{\\mathrm{max}}$",
    "mmin": "$m_{\\mathrm{min}}$",
    "sigma_1": "$\\sigma_{1}$",
    "sigma_12": "$\\sigma_{12}$",
}


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
    color="green"
    )

def plot_simulated_population_corner(params):

    samples = read_simulate_pop_samples()[params]
    fig = corner.corner(
        samples,
        labels=[latex for param, latex in LATEX.items() if param in params],
        truths=[t for param, t in HYPER_PARAM_VALS.items() if param in params],
        **CORNER_KWARGS,
    )

    plt.legend(
        handles=[
            mlines.Line2D([], [], color=CORNER_KWARGS['color'], label="PI"),
            mlines.Line2D([], [], color="tab:orange", label="True"),
        ],
        fontsize=20, frameon=False,
        bbox_to_anchor=(1, len(params)), loc="upper right"
    )
    title=True
    if title:
        fig.suptitle("Population Inference\nwith simulated population", y=0.97)
        fig.subplots_adjust(top=0.75) 
    fig.savefig("simulated_population_inference_corner.png")



def read_simulate_pop_samples():
    df= pd.read_csv(SIMULATED_POP_SAMPLES,sep=' ')
    return df



print("Plotting...")
plot_simulated_population_corner(params=[ "sigma_1", "sigma_12"])

