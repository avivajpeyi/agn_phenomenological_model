import matplotlib.pyplot as plt

from agn_utils.bbh_population_generators.posterior_simulator import simulate_exact_population_posteriors
from agn_utils.pe_postprocessing.jsons_to_numpy import get_bilby_results
from agn_utils.plotting.posterior_predictive_plot import plot_posterior_predictive_check, plot_trues, update_style
from agn_utils.plotting.posterior_violin_plotter import simple_violin_plotter
from bilby.gw.result import CBCResult
from bilby.core.prior import TruncatedNormal

import argparse

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
        exact_pop = simulate_exact_population_posteriors(sig1=sim_true_val[0], sig12=sim_true_val[1], number_events=1)[
            'posteriors']
        samps.append(exact_pop)
        labels.append("Population " + str(sim_true_val))

    for pop_name, regex in pops_regexs.items():
        res = get_bilby_results(regex, pop_name + ".pkl", params=["cos_tilt_1", "cos_theta_12"])
        samps.append(res['posteriors'])
        trues.append(res['trues'])
        labels.append("90\% CI PE Posteriors")

    plot_posterior_predictive_check(samps, labels, colors=colors1, axes=cdf_axes,)
    plot_trues(trues, true_pop_params, labels, axes=pdf_axes, colors=colors2)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname)


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-a-regex",type=str, default="")
    parser.add_argument( "--pop-b-regex",type=str, default=""  )
    args = parser.parse_args()
    return args


def plotter(pop_a_regex, pop_b_regex):
    update_style()

    TruePopA = "cornflowerblue"
    TruePopB = "bisque"
    DataA = "midnightblue"
    DataB = "darkorange"

    if pop_a_regex:
        pop_a_pkl = "pop_a.pkl"
        dat_a = get_bilby_results(pop_a_regex, pop_a_pkl, ["cos_tilt_1", "cos_theta_12"])
        simple_violin_plotter(dat_a, "pop_a_pe.png")
        pe_cdf(
            pops_regexs=dict(
                pop_a=pop_a_regex,
            ),
            true_pop_params=dict(
                popA=[0.5, 3]),
            title="Population A", fname="pop_a_cdf.png",
            colors1=[TruePopA, DataA],
            colors2=[TruePopA, DataA]

        )

    if pop_b_regex:
        pop_b_pkl = "pop_b.pkl"
        dat_b = get_bilby_results(pop_b_regex, pop_b_pkl, ["cos_tilt_1", "cos_theta_12"])
        # simple_violin_plotter(dat_b, "pop_b_pe.png")
        pe_cdf(
            pops_regexs=dict(
                pop_b=pop_b_regex
            ),
            true_pop_params=dict(
                popB=[1, 0.25]),
            title="Population B", fname="pop_b_cdf.png",
            colors1=[TruePopB, DataB],
            colors2=[TruePopB, DataB],
        )

    if pop_a_regex and pop_b_regex:
        pe_cdf(
            pops_regexs=dict(
                pop_a=pop_a_regex,
                pop_b=pop_b_regex
            ),
            true_pop_params=dict(
                popA=[0.5, 3],
                popB=[1, 0.25]),
            title="Population Comparision", fname="pop_compare_cdf.png",
            colors1=[TruePopA, TruePopB, DataA,  DataB],
            colors2=[TruePopA, TruePopB, DataA, DataB]
        )

def main():
    args = create_parser_and_read_args()
    plotter(pop_a_regex=args.pop_a_regex, pop_b_regex=args.pop_b_regex)


if __name__ == "__main__":
    main()
