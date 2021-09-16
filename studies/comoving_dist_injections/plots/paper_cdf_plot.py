from agn_utils.pe_postprocessing.posterior_reweighter import rejection_sample_population
from agn_utils.plotting.cdf_plotter.sigma_cdf_difference_check import pe_cdf
from agn_utils.pe_postprocessing.jsons_to_numpy import load_posteriors_and_trues
from agn_utils.plotting.posterior_violin_plotter import simple_violin_plotter
import numpy as np
import matplotlib.pyplot as plt
from agn_utils.data_formetter import ld_to_dl, dl_to_ld
from agn_utils.bbh_population_generators.posterior_simulator import simulate_exact_population_posteriors
from agn_utils.plotting.posterior_predictive_plot import plot_posterior_predictive_check, plot_trues, update_style
from matplotlib import rcParams
import matplotlib
import os
import pandas as pd
from bilby.core.result import Result
from bilby.core.prior import PriorDict, Uniform
from tqdm.auto import tqdm

np.random.seed(0)

update_style()



true_a = "navy"
true_b = "firebrick"
dat_a = "cornflowerblue"
dat_b = "lightcoral"

COLORS = [true_a, true_b, dat_a, dat_b]


def downsample_dat(num_samples, dat):
    len_dat = len(dat['labels'])
    ids = np.random.choice([i for i in range(0, len_dat)], replace=False, size=num_samples)
    new_dat = {
        k: {} for k in dat.keys()
    }
    new_dat['labels'] = list(np.array(dat['labels'])[ids])
    new_dat['trues'] = {
        k: list(np.array(v)[ids]) for k, v in dat['trues'].items()
    }
    new_dat['posteriors'] = {
        k: list(np.array(v)[ids]) for k, v in dat['posteriors'].items()
    }
    return new_dat

def plot_cdf_row(pops_dat_dicts, true_pop_params, colors, cdf_axes, legend=False):

    samps, labels = [], []

    for sim_name, sim_true_val in true_pop_params.items():
        exact_pop = simulate_exact_population_posteriors(sig1=sim_true_val[0], sig12=sim_true_val[1], number_events=1)['posteriors']
        samps.append(exact_pop)
        if legend:
            labels.append(sim_name)

    for pop_name, dat in pops_dat_dicts.items():
        samps.append(dat['posteriors'])

    plot_posterior_predictive_check(samps, labels, colors=colors, axes=cdf_axes)


def main_plotter(dat_a, dat_b):

    fig, axes = plt.subplots(nrows=3, ncols=2, sharex='col', sharey='row', figsize=(6, 7.5))

    for i, n in enumerate([15, 30, 45]):
        down_dat_a = downsample_dat(n, dat_a)
        down_dat_b = downsample_dat(n, dat_b)

        plot_cdf_row(
            pops_dat_dicts=dict(
                pop_a=down_dat_a,
                pop_b=down_dat_b
            ),
            true_pop_params={
                "Population A":[0.5, 3.0],
                "Population B":[1, 0.25]
        },
            colors=COLORS,
            cdf_axes=axes[i, :],
            legend=False
        )

        for j in range(2):
            axes[i, j].text(0.05,0.85,f"{n} Events", transform=axes[i, j].transAxes)
            axes[i,j].tick_params(zorder=100)

    for ax in axes.flatten():
        ax.set_axisbelow(False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, top=0.95, bottom=0.1)
    fig.savefig("cdfs.png")

def plot_version_2(dat_a, dat_b):
    # plt.locator_params(axis='y', nbins=3)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6, 7.5))
    for i, n in enumerate([15, 30, 45]):
        down_dat_a = downsample_dat(n, dat_a)
        down_dat_b = downsample_dat(n, dat_b)

        plot_cdf_row(
            pops_dat_dicts=dict(
                pop_a=down_dat_a,
                pop_b=down_dat_b
            ),
            true_pop_params={
                "Population A":[0.5, 3.0],
                "Population B":[1, 0.25]
            },
            colors=COLORS,
            cdf_axes=axes[i, :],
            legend=False
        )

        axes[i, 0].text(0.05,0.85,f"{n} Events", transform=axes[i, 0].transAxes, fontsize='large')
        axes[i, 0].set_xticks([-1, 0, 0.8])
        axes[i, 1].set_xticks([-0.8, 0, 1])

        if i != 2:
            for j in range(2):
                axes[i, j].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off'  # labels along the bottom edge are off)
                )
                axes[i, j].set_xlabel("")
                axes[i, j].set_xticklabels([])



        for j in range(2):
            axes[i,j].tick_params(zorder=100)
            axes[i,j].yaxis.set_major_locator(plt.MaxNLocator(3))


    for ax in axes.flatten():
        ax.set_axisbelow(False)

    fig.text(0.04, 0.5, 'Cumulative Probability', va='center', rotation='vertical', fontsize='x-large')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.05, top=0.95, bottom=0.1, left=0.15)
    fig.savefig("cdfs_v2.png")

def convert_to_bilby_res(dat):
    posteriors = dl_to_ld(dat['posteriors'])
    trues = dl_to_ld(dat['trues'])
    results = []
    p = PriorDict(dict(cos_theta_12=Uniform(-1,1), cos_tilt_1=Uniform(-1,1), weights=Uniform(0,1)))
    for i in tqdm(range(len(posteriors)), desc="Converting to Results"):
        r = Result()
        r.search_parameter_keys = ['cos_tilt_1', 'cos_theta_12', 'weights']
        r.injection_parameters = trues[i]
        r.priors = p
        r.label = dat['labels'][i]
        r.outdir = "plots"
        r.posterior = pd.DataFrame(posteriors[i])
        results.append(r)
    return results


def plot_all(dat):
    os.makedirs('plots', exist_ok=True)
    # rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["xkcd:orange", "xkcd:green", "xkcd:green"])
    rcParams['axes.grid'] = False
    rcParams["axes.axisbelow"] = False
    res = convert_to_bilby_res(dat)
    for r in tqdm(res, total=len(res), desc="Plotting"):
        fig = r.plot_corner(
            color="C0",
            label_kwargs=dict(fontsize=35, labelpad=12), labelpad=0.05,
            title_kwargs=dict(fontsize=25, pad=12), save=False,
            plot_datapoints=False, smooth=1.2, bins=20, hist_bin_factor=2
        )
        fig.savefig(f'plots/{r.label}.png', bbox_inches='tight', pad_inches=0.1)



if __name__ == '__main__':
    pop_a_pkl = 'pop_a.pkl'
    pop_b_pkl = 'pop_b.pkl'

    dat_a = load_posteriors_and_trues(pop_a_pkl)
    dat_b = load_posteriors_and_trues(pop_b_pkl)

    dat_a["posteriors"] = rejection_sample_population(
        dat_a["posteriors"],
        true_population_param=dict(sigma_1=0.5, sigma_12=3)
    )
    dat_b["posteriors"] = rejection_sample_population(
        dat_b["posteriors"],
        true_population_param=dict(sigma_1=1, sigma_12=0.25)
    )
    main_plotter(dat_a, dat_b)
    plot_version_2(dat_a, dat_b)

    plot_all(dat_a)

