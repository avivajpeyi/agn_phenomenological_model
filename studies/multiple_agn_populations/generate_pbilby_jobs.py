import os

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from agn_utils.pe_setup.setup_multiple_pbilby_injections import pbilby_jobs_generator, create_ini
from agn_utils.plotting.overlaid_corner_plotter import CORNER_KWARGS

plt.style.use(
    "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle")


def plot_samples(all_samp):
    all_samp_df = pd.read_csv(all_samp, sep=' ')
    high_snr_df = all_samp_df[all_samp_df['network_snr'] >= 60]
    params = ['cos_tilt_1', 'cos_tilt_2', 'cos_theta_12']
    labels = [p.replace("_", " ") for p in params]
    more_kwargs = {}  # dict(pcolor_kwargs=zor, contourf_kwargs=zor, contour_kwargs=zor)
    fig = corner.corner(all_samp_df[params], labels=labels, **CORNER_KWARGS, color="tab:blue", **more_kwargs,
                        hist_kwargs=dict(cumulative=False, density=True))
    corner.corner(high_snr_df[params], labels=labels, color="tab:orange", plot_contours=False, plot_datapoints=True,
                  plot_density=False, data_kwargs=dict(alpha=1, ms=5, zorder=2),
                  hist_kwargs=dict(cumulative=False, density=True), fig=fig)
    plt.suptitle(all_samp.replace(".dat", " ").replace("_", " "))
    plt.legend(
        handles=[
            mlines.Line2D(
                [], [], color=c, label=l
            )
            for c, l in zip(['tab:blue', 'tab:orange'], ['Population', 'High SNR'])
        ],
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(1, len(params)),
        loc="upper right",
    )
    plt.savefig(all_samp.replace(".dat", ".png"))
    plt.close('all')


POPS = ['posteriors_list/pop_a.dat', 'posteriors_list/pop_b.dat']


def main_job_gen():
    for p in POPS:
        try:
            plot_samples(p)
        except Exception:
            pass
        pbilby_jobs_generator(
            injection_file=p.replace(".dat", "_highsnr.dat"),
            label=os.path.basename(p).replace(".dat", ""),
            prior_file="4s",
            psd_file="posteriors_list/aLIGO_late_psd.txt",
            waveform="IMRPhenomXPHM"
        )


def main_sampler_setting_study():
    p =  os.path.abspath('./data/pop_a_highsnr.dat')
    default_args = dict(injection_idx=7, injection_file=p, prior_file=os.path.abspath('./data/4s.prior'),
                        psd_file=os.path.abspath("./data/aLIGO_late_psd.txt"), waveform="IMRPhenomXPHM", nodes=12,tasks=12)
    for nlive, nact, label in zip([2000, 2500, 1500],[20, 20, 25], ["run_b", "run_c", "run_d"]):
        create_ini(**default_args, label=label, nlive=nlive, nact=nact)

if __name__ == "__main__":
    main_sampler_setting_study()
