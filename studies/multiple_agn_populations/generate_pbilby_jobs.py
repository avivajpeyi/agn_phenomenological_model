import os

import bilby
import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import add_cos_theta_12_from_component_spins
from agn_utils.pe_setup.setup_multiple_pbilby_injections import pbilby_jobs_generator
from agn_utils.plotting.overlaid_corner_plotter import CORNER_KWARGS
from bilby.gw.conversion import generate_all_bbh_parameters
from bilby.gw.prior import BBHPriorDict

plt.style.use(
    "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle")

plt.rcParams['text.usetex'] = False

PRI_PATH = "data/bbh.prior"
PRIOR = BBHPriorDict(filename=PRI_PATH)


def plot_samples(samp_path):
    n = 1000
    high_snr_df = pd.read_csv(samp_path, sep=' ')
    pri_samp = pd.DataFrame(generate_all_bbh_parameters(PRIOR.conversion_function(PRIOR.sample(n))))
    pri_samp = add_cos_theta_12_from_component_spins(pri_samp)
    check_if_injections_in_prior(high_snr_df, PRI_PATH)
    high_snr_df = add_cos_theta_12_from_component_spins(pri_samp)
    params = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'cos_tilt_1', 'cos_tilt_2', 'cos_theta_12']
    labels = ['q', 'Mc', 'dl', 'cos t1', 'cos t2', 'cos t12']
    fig = corner.corner(pri_samp[params], labels=labels, **CORNER_KWARGS, color="tab:green",
                        hist_kwargs=dict(density=True), alpha=0.3)
    pts = high_snr_df[params]
    corner.overplot_points(fig=fig, xs=pts, color="tab:orange")
    for index, row in pts.iterrows():
        data = row.to_list()
        corner.overplot_lines(fig=fig, xs=data, color="tab:orange", alpha=0.2)

    plt.suptitle(samp_path.replace(".dat", " ").replace("_", " "))
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=c, label=l)
            for c, l in zip(['tab:orange', 'tab:green'], ['High SNR', '4S prior'])
        ],
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(1, len(params)),
        loc="upper right",
    )
    plt.savefig(samp_path.replace(".dat", ".png"))
    plt.close('all')


def check_if_injections_in_prior(injection_df: pd.DataFrame, prior_path: str):
    priors = bilby.prior.PriorDict(filename=prior_path)
    if not set(priors.keys()).issubset(set(injection_df.columns)):
        injection_df = injection_df.copy()
        injection_df = generate_all_bbh_parameters(injection_df)
    in_prior = pd.DataFrame(index=injection_df.index)
    for prior in priors.values():
        inj_param = injection_df[prior.name].values
        in_prior[f"in_{prior.name}_prior"] = (prior.minimum <= inj_param) & (inj_param <= prior.maximum)
    not_in_prior = in_prior[in_prior.isin([False]).any(axis=1)]  # get all rows where inj_param outside pri range
    if len(not_in_prior) > 0:
        print(f"The following injection id(s) have parameters outside your prior range: {list(not_in_prior.T.columns)}")
    return list(list(not_in_prior.T.columns))


POPS = ['data/pop_a_highsnr.dat', 'data/pop_b_highsnr.dat']


def main_job_gen():
    for p in POPS:
        try:
            plot_samples(p)
        except Exception:
            pass
        # pbilby_jobs_generator(
        #     injection_file=p,
        #     label=os.path.basename(p).replace(".dat", ""),
        #     prior_file="data/bbh.prior",
        #     psd_file="posteriors_list/aLIGO_late_psd.txt",
        #     waveform="IMRPhenomXPHM"
        # )


if __name__ == "__main__":
    main_job_gen()
