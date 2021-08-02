# -*- coding: utf-8 -*-
"""Module one liner

"""

import matplotlib.pyplot as plt

from agn_utils.bbh_population_generators.posterior_simulator import simulate_population_posteriors, \
    simulate_exact_population_posteriors
from agn_utils.pe_postprocessing.jsons_to_numpy import get_bilby_results
from agn_utils.pe_postprocessing.posterior_reweighter import rejection_sample_population
from agn_utils.plotting.posterior_predictive_plot import plot_posterior_predictive_check, plot_trues, update_style
from agn_utils.plotting.posterior_violin_plotter import simple_violin_plotter

POP_A_REGEX = "/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/multiple_agn_populations/output_ozstar/pop_a/*.json"
POP_B_REGEX = "/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/multiple_agn_populations/output_ozstar/pop_b/*.json"
POP_A_PKL = "pop_a.pkl"
POP_B_PKL = "pop_b.pkl"


def plot_simulated_cdfs(true_pop_params, fname="posterior_predictive_check.png",
                        num_simulated_events=10):
    plt.close('all')
    update_style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    cdf_axes = axes[0, :]
    pdf_axes = axes[1, :]

    samps, trues, labels = [], [], []
    n = num_simulated_events
    for sim_name, sim_true_val in true_pop_params.items():
        exact_pop = simulate_exact_population_posteriors(sig1=sim_true_val[0], sig12=sim_true_val[1], number_events=n)[
            'posteriors']
        samps.append(exact_pop)
        labels.append("Population " + str(sim_true_val))

        simulated_pop = simulate_population_posteriors(sig1=sim_true_val[0], sig12=sim_true_val[1], number_events=n)
        posteriors = simulated_pop['posteriors']
        true = simulated_pop['trues']
        posteriors = rejection_sample_population(posteriors, dict(sigma_1=sim_true_val[0], sigma_12=sim_true_val[1]))
        samps.append(posteriors)
        trues.append(true)
        labels.append("90\% CI Simulated Posteriors")


    TruePopA = "cornflowerblue"
    TruePopB = "bisque"
    DataA = "midnightblue"
    DataB = "darkorange"

    plot_posterior_predictive_check(samps, labels, colors=[TruePopA, DataA, DataB, TruePopB ],  axes=cdf_axes)
    plot_trues(trues, true_pop_params, labels, axes=pdf_axes, colors=[TruePopA, TruePopB, DataA, DataB])
    plt.suptitle(f"Simulated Events: {num_simulated_events}")
    plt.tight_layout()
    plt.savefig(fname)


def pe_cdf(pops_regexs, true_pop_params, fname="posterior_predictive_check.png", title="Population A",
           num_simulated_events=10, colors1=[], colors2=[]):
    plt.close('all')
    update_style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    cdf_axes = axes[0, :]
    pdf_axes = axes[1, :]

    samps, trues, labels, cols = [], [], [], []

    n = num_simulated_events
    for sim_name, sim_true_val in true_pop_params.items():
        exact_pop = simulate_exact_population_posteriors(sig1=sim_true_val[0], sig12=sim_true_val[1], number_events=n)[
            'posteriors']
        samps.append(exact_pop)
        labels.append("Population " + str(sim_true_val))

    for pop_name, regex in pops_regexs.items():
        res = get_bilby_results(regex, pop_name + ".pkl", params=["cos_theta_1", "cos_theta_12"])
        samps.append(res['posteriors'])
        trues.append(res['trues'])
        labels.append("90\% CI PE Posteriors")

    plot_posterior_predictive_check(samps, labels, colors=colors1,  axes=cdf_axes)
    plot_trues(trues, true_pop_params, labels, axes=pdf_axes, colors=colors2)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname)


if __name__ == "__main__":
    # dat_a = get_bilby_results(POP_A_REGEX, POP_A_PKL, ["cos_theta_1", "cos_theta_12"])
    # update_style()
    # simple_violin_plotter(dat_a, "pop_a_pe.png")
    # dat_b = get_bilby_results(POP_B_REGEX, POP_B_PKL, ["cos_theta_1", "cos_theta_12"])
    update_style()
    # simple_violin_plotter(dat_b, "pop_b_pe.png")

    simulated_pop = simulate_population_posteriors(sig1=0.5, sig12=3, number_events=20)
    simple_violin_plotter(simulated_pop, "simulated_posteriors.png")


    TruePopA = "cornflowerblue"
    TruePopB = "bisque"
    DataA = "midnightblue"
    DataB = "darkorange"

    # pe_cdf(
    #     pops_regexs=dict(
    #         pop_a="/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/multiple_agn_populations/output_ozstar/pop_a/*.json",
    #         pop_b="/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/multiple_agn_populations/output_ozstar/pop_b/*.json"
    #     ),
    #     true_pop_params=dict(
    #         popA=[0.5, 3],
    #         popB=[1, 0.25]),
    #     title="Population Comparision", fname="pop_compare_cdf.png",
    #     colors1=[DataA, DataB],
    #     colors2=[TruePopA, TruePopB, DataA, DataB]
    # )


    # pe_cdf(
    #     pops_regexs=dict(
    #         pop_a="/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/multiple_agn_populations/output_ozstar/pop_a/*.json",
    #         # pop_b="/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/multiple_agn_populations/output_ozstar/pop_b/*.json"
    #     ),
    #     true_pop_params=dict(
    #         popA=[0.5, 3]),
    #         # popB=[1, 0.25]),
    #     title="Population A", fname="pop_a_cdf.png",
    #     colors1=[TruePopA, DataA],
    #     colors2=[TruePopA, DataA]
    #
    # )
    #
    # pe_cdf(
    #     pops_regexs=dict(
    #         # pop_a="/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/multiple_agn_populations/output_ozstar/pop_a/*.json",
    #         pop_b="/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/multiple_agn_populations/output_ozstar/pop_b/*.json"
    #     ),
    #     true_pop_params=dict(
    #         # popA=[0.5, 3],
    #         popB=[1, 0.25]),
    #         title="Population B", fname="pop_b_cdf.png",
    #     colors1=[TruePopB, DataB],
    #     colors2=[TruePopB, DataB],
    #     )

    for n in [2, 10, 30, 50, 100]:
        plot_simulated_cdfs(
            true_pop_params=dict(
                popA=[0.5, 3],
                popB=[1, 0.25]),
            fname=f"pop_comparision_exp_{n:03}_cdf.png",
            num_simulated_events=n
        )
