import os

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from agn_utils.pe_setup.setup_multiple_pbilby_injections import main_generator
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
                        hist_kwargs=dict(cumulative=True, density=True))
    corner.corner(high_snr_df[params], labels=labels, color="tab:orange", plot_contours=False, plot_datapoints=True,
                  plot_density=False, data_kwargs=dict(alpha=1, ms=5, zorder=2),
                  hist_kwargs=dict(cumulative=True, density=True), fig=fig)
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


POPS = ['pop_a.dat', 'pop_b.dat']


def main():
    for p in POPS:
        plot_samples(p)
        main_generator(
            injection_file=p.replace(".dat", "_highsnr.dat"),
            label=os.path.basename(p).replace(".dat", ""),
            prior_file="4s",
            psd_file="data/aLIGO_late_psd.txt",
            waveform="IMRPhenomPv2"
        )


if __name__ == "__main__":
    main()
