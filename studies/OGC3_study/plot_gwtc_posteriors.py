import glob
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
import h5py
import numpy as np
import pandas as pd
from agn_utils.bbh_population_generators.spin_conversions import make_spin_vector, \
    calculate_relative_spins_from_component_spins
from agn_utils.data_formetter import ld_to_dl, dl_to_ld
from agn_utils.plotting.cdf_plotter.sigma_cdf_difference_check import pe_cdf
from agn_utils.pe_postprocessing.jsons_to_numpy import load_posteriors_and_trues, save_posteriors_and_trues
from agn_utils.plotting.posterior_violin_plotter import simple_violin_plotter
from tqdm.auto import tqdm
from bilby.core.result import Result
from bilby.core.prior import PriorDict, Uniform
import pandas as pd

plt.style.use(
    "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle")


def load_gwtc_posteriors():
    dat = load_posteriors_and_trues("gwtc.pkl")
    posteriors = dl_to_ld(dat['posteriors'])
    pri = get_prior()
    results = []
    for i in tqdm(range(len(posteriors)), desc="Loading Results"):
        r = Result()
        r.search_parameter_keys = list(pri.keys())
        r.label = get_event_name(dat['labels'][i])
        r.outdir = "plots"
        r.priors = pri
        r.posterior = pd.DataFrame(posteriors[i])
        results.append(r)
    return results

def get_event_name(s):
    return os.path.basename(s).split("-")[0].replace(" ", "_")

def get_prior():
    return PriorDict(dict(
        cos_tilt_1=Uniform(-1,1,"cos_tilt_1",r"$\cos\theta_{1}$"),
        cos_theta_12=Uniform(-1,1,"cos_theta_12",r"$\cos\theta_{12}$")
    ))



def plot_all():
    os.makedirs('plots', exist_ok=True)
    rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["xkcd:orange", "xkcd:green", "xkcd:green"])
    rcParams['axes.grid'] = False
    rcParams["axes.axisbelow"] = False
    res = load_gwtc_posteriors()
    for r in tqdm(res, total=len(res), desc="Plotting"):
        fig = r.plot_corner(
            priors=True, color="C0",
            label_kwargs=dict(fontsize=35, labelpad=12), labelpad=0.05,
            title_kwargs=dict(fontsize=25, pad=12), save=False,
            plot_datapoints=False, smooth=1.2, bins=20, hist_bin_factor=2
        )
        fig.savefig(f'plots/{r.label}.png', bbox_inches='tight', pad_inches=0.1)






def main():
    rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["xkcd:orange", "xkcd:green", "xkcd:green"])
    rcParams['axes.grid'] = False
    rcParams["axes.axisbelow"] = False
    plot()


if __name__ == "__main__":
    main()
