from agn_utils.pe_postprocessing.posterior_reweighter import rejection_sample_population
from agn_utils.plotting.cdf_plotter.sigma_cdf_difference_check import pe_cdf
from agn_utils.pe_postprocessing.jsons_to_numpy import load_posteriors_and_trues
from agn_utils.plotting.posterior_violin_plotter import simple_violin_plotter
import numpy as np

np.random.seed(0)

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

simple_violin_plotter(dat_a, "pop_a_reweighted.png")
simple_violin_plotter(dat_a, "pop_b_reweighted.png")

TruePopA = "cornflowerblue"
TruePopB = "bisque"
DataA = "midnightblue"
DataB = "darkorange"


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


for n in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    down_dat_a = downsample_dat(n, dat_a)
    down_dat_b = downsample_dat(n, dat_b)

    pe_cdf(
        pops_dat_dicts=dict(
            pop_a=down_dat_a,
            pop_b=down_dat_b
        ),
        true_pop_params=dict(
            popA=[0.5, 3.0],
            popB=[1, 0.25]
        ),
        title=f"Num event {n:02}", fname=f"pop_compare_cdf_{n:02}.png",
        colors1=[DataA, DataB, TruePopA, TruePopB],
        colors2=[DataA, DataB, TruePopA, TruePopB]
    )
