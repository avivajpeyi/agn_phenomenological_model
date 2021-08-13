# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import add_snr, add_cos_theta_12_from_component_spins
from bilby.core.prior import TruncatedNormal
from bilby.gw.conversion import generate_spin_parameters, generate_mass_parameters, \
    convert_to_lal_binary_black_hole_parameters
from agn_utils.bbh_population_generators.spin_conversions import  calculate_relative_spins_from_component_spins

# POPULATION_A = dict(sigma_1=0.5, sigma_12=3)
#
# POPULATION_B = dict(sigma_1=1,sigma_12=0.25)


def process_samples(s, rf=20):
    s['reference_frequency'] = rf
    s, _ = convert_to_lal_binary_black_hole_parameters(s)
    s = generate_mass_parameters(s)
    s = generate_spin_parameters(s)
    s = add_cos_theta_12_from_component_spins(s)


    # s = add_snr(s)
    # s['snr'] = s['network_snr']
    return s


def plot_pdf(pop_name, pop_val, pop_file, full_pop_file, outdir='.'):
    """Plots a scatter plot."""
    try:
        plt.style.use(
        "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle")
    except Exception:
        pass

    plt.close('all')
    all = pd.read_csv(full_pop_file, sep=" ")
    all['cos_theta_1'] = all['cos_tilt_1']
    all = process_samples(all)
    sub = pd.read_csv(pop_file, sep=" ")
    sub = process_samples(sub)
    sub['cos_theta_1'] = sub['cos_tilt_1']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, l in zip(axes, ["cos_theta_1", "cos_theta_12"]):
        ax.hist(all[l], density=True, histtype='step', color="tab:blue", label="ALL", lw=2, alpha=0.8)
        ax.hist(sub[l], density=True, histtype='step', color="tab:purple", label="HIGH SNR", lw=2, alpha=0.6)

    x = np.linspace(-1, 1, 100)
    y1 = TruncatedNormal(mu=1, sigma=pop_val[0], minimum=-1, maximum=1).prob(x)
    y2 = TruncatedNormal(mu=1, sigma=pop_val[1], minimum=-1, maximum=1).prob(x)
    axes[1].plot(x, y2, color='tab:gray', zorder=-10, lw=3, label="TRUE")
    axes[0].plot(x, y1, color='tab:gray', zorder=-10, lw=3)

    for i in range(len(axes)):
        if (i == 0):
            axes[i].set_xlabel(r"$\cos\ \theta_1$")
            axes[i].set_ylabel("PDF")
        else:
            axes[i].set_xlabel(r"$\cos\ \theta_{12}$")
            axes[i].set_yticklabels([])
            axes[i].legend()
        axes[i].grid(False)
        axes[i].set_xlim(-1, 1)

    plt.suptitle(f"POP {pop_name}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/pop_trues_{pop_name}.png")


def plot_snr(pop_a_file, pop_b_file, outdir='.'):
    plt.close()
    a_snr = pd.read_csv(pop_a_file, sep=" ")
    b_snr = pd.read_csv(pop_b_file, sep=" ")
    plt.hist(a_snr.network_snr, density=False, label="PopA",histtype="step")
    plt.hist(b_snr.network_snr, density=False, label="PopB",histtype="step")
    plt.ylabel("Count")
    plt.xlabel("SNR")
    # plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/snr.png")

def plot_snr_dist(pop_a_file, pop_b_file, outdir='.'):
    plt.close()
    a_snr = pd.read_csv(pop_a_file, sep=" ")
    b_snr = pd.read_csv(pop_b_file, sep=" ")
    plt.scatter(a_snr.luminosity_distance, a_snr.network_snr, label="PopA")
    plt.scatter(b_snr.luminosity_distance, b_snr.network_snr, label="PopB")
    plt.ylabel("SNR")
    plt.xlabel("Dl")
    # plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/snr_dl.png")


def main_plotter(pop_a_file, pop_b_file, full_pop_a_file, full_pop_b_file, outdir, pop_a_vals, pop_b_vals):
    plot_snr(pop_a_file, pop_b_file, outdir)
    plot_snr_dist(pop_a_file, pop_b_file, outdir)
    plot_pdf('a', pop_a_vals, pop_a_file, full_pop_a_file, outdir)
    plot_pdf('b', pop_b_vals, pop_b_file, full_pop_b_file, outdir)


if __name__ == "__main__":
    main()
