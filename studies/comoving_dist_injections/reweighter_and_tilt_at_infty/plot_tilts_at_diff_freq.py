import pandas as pd
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from agn_utils.bbh_population_generators.posterior_simulator import simulate_exact_population_posteriors
from agn_utils.pe_postprocessing.jsons_to_numpy import get_bilby_results
from agn_utils.pe_postprocessing.posterior_reweighter import rejection_sample_population
from agn_utils.plotting.posterior_predictive_plot import plot_posterior_predictive_check, plot_trues, update_style
from agn_utils.plotting.posterior_violin_plotter import simple_violin_plotter
import matplotlib.pyplot as plt
import numpy as np


true_a = "navy"
true_b = "firebrick"
dat_a = "cornflowerblue"
dat_b = "lightcoral"

COLORS = [true_a, true_b, dat_a, dat_b]



def load_dat(fname):
    interesting_param = ['cos_tilt_1', 'tilt_1_inf', 'cos_theta_12']
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        print(data['posteriors'].keys())
        data['posteriors'] = {k:v for k,v in data['posteriors'].items() if k in interesting_param}
        data['posteriors']['cos_tilt_1_inf'] = []
        for tilt_1_inf in data['posteriors']['tilt_1_inf']:
            data['posteriors']['cos_tilt_1_inf'].append(np.cos(tilt_1_inf))
        return data


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





def main():
    dat_b = load_dat('pop_b.pkl')
    dat_b['posteriors']['cos_tilt_1'] = dat_b['posteriors']['cos_tilt_1_inf']
    dat_a = load_dat('pop_a.pkl')
    dat_a['posteriors']['cos_tilt_1'] = dat_a['posteriors']['cos_tilt_1_inf']
    fig, ax = plt.subplots(1,2, figsize=(7, 4))
    plot_cdf_row(
        pops_dat_dicts=dict(
            pop_a=dat_a,
            pop_b=dat_b,
        ),
        true_pop_params=dict(
            # popA=[0.5, 3],
            # popb=[1, 0.25]
        ),
        colors=[COLORS[2],COLORS[3]],
        cdf_axes=ax,
    )
    ax[0].set_title("freq AGN")
    plt.tight_layout()
    fig.savefig("fagn.png")

    plt.close('all')
    dat_b = load_dat('pop_b.pkl')
    dat_a = load_dat('pop_a.pkl')
    fig, ax = plt.subplots(1,2, figsize=(7, 4))
    plot_cdf_row(
        pops_dat_dicts=dict(
            pop_a=dat_a,
            pop_b=dat_b,
        ),
        true_pop_params=dict(
            popA=[0.5, 3],
            popb=[1, 0.25]
        ),
        colors=COLORS,
        cdf_axes=ax,
    )
    ax[0].set_title("freq REF")
    plt.tight_layout()
    fig.savefig("fref.png")

if __name__ == '__main__':
    main()