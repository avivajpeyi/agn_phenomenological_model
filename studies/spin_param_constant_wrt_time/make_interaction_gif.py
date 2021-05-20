import glob
import logging
import multiprocessing
import os
import shutil
import time

from matplotlib.pyplot import imshow
import bilby
import corner
import lalsimulation
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from bilby.core.prior import Cosine, Interped, Uniform

from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import iid_spin_magnitude_beta, truncnorm
from matplotlib import rcParams
from numpy import cos, sin
from tqdm import tqdm


logging.getLogger("bilby").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.INFO)

REFERENCE_FREQ = 10 ** -3
NUM = 10000


def sample_from_agn_population(sigma_1, sigma_12, n=NUM):
    samples = get_samples(
        num_samples=n,
        population_params=dict(sigma_1=sigma_1, sigma_12=sigma_12),
    )
    return samples[[k for k in PARAMS.keys()]]


def overlaid_corner(
    samples_list,
    sample_labels,
    samples_colors,
    params,
    fname,
    title,
    show_legend=True,
):
    """Plots multiple corners on top of each other"""
    # print(f"plotting {fname}")
    # sort the sample columns
    samples_list = [s[params] for s in samples_list]

    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = min([len(s) for s in samples_list])

    CORNER_KWARGS.update(
        range=[PARAMS[p]["r"] for p in params],
        labels=[PARAMS[p]["l"] for p in params],
    )

    fig = corner.corner(
        samples_list[0],
        color=samples_colors[0],
        **CORNER_KWARGS,
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), min_len),
            color=samples_colors[idx],
            **CORNER_KWARGS,
        )
    if show_legend:
        plt.legend(
            handles=[
                mlines.Line2D(
                    [], [], color=samples_colors[i], label=sample_labels[i]
                )
                for i in range(n)
            ],
            fontsize=20,
            frameon=False,
            bbox_to_anchor=(1, ndim),
            loc="upper right",
        )
    plt.suptitle(title)
    fig.savefig(fname)
    plt.close(fig)


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (
        len_of_longest_samples / len_current_samples
    )


def plot_overlaid_corners(
    sigma_1_list,
    sigma_12_list,
    pltdir,
    n=NUM,
    show_aligned=False,
    show_isotropic=False,
    show_agn=True,
):
    isotropic_samples = get_samples_from_prior(
        num_samples=n, prior_fname="isotropic.prior"
    )
    aligned_samples = get_samples_from_prior(
        num_samples=n, prior_fname="aligned.prior"
    )

    if os.path.isdir(pltdir):
        assert pltdir != "."
        shutil.rmtree(pltdir)
    os.makedirs(pltdir, exist_ok=False)

    i = 0
    for i_sig1, i_sig12 in tqdm(
        zip(sigma_1_list, sigma_12_list),
        total=len(sigma_1_list),
        desc="Hyper-Param settings",
    ):
        f_theta = (
            f"{pltdir}/{i:02}_sig12-{i_sig12:.1f}_sig1-{i_sig1:.1f}_theta.png"
        )
        f_chi = (
            f"{pltdir}/{i:02}_sig12-{i_sig12:.1f}_sig1-{i_sig1:.1f}_chi.png"
        )
        f_combo = (
            f"{pltdir}/{i:02}_sig12-{i_sig12:.1f}_sig1-{i_sig1:.1f}_combo.png"
        )
        agn_label = (
            r"$\sigma_{1}="
            + f"{i_sig1:.1f}"
            + r", \sigma_{12}="
            + f"{i_sig12:.1f}$"
        )

        samples_list, samples_colors, samples_labels = [], [], []
        if show_aligned:
            samples_list.append(aligned_samples)
            samples_colors.append("tab:green")
            samples_labels.append("Field")
        if show_isotropic:
            samples_list.append(isotropic_samples)
            samples_colors.append("tab:orange")
            samples_labels.append("Dynamic")
        if show_agn:
            agn_samples = sample_from_agn_population(
                sigma_1=i_sig1, sigma_12=i_sig12, n=n
            )
            samples_list.append(agn_samples)
            samples_colors.append("tab:blue")
            samples_labels.append("AGN " + agn_label)

        overlaid_corner(
            samples_list,
            samples_labels,
            samples_colors,
            params=["chi_eff", "chi_p"],
            fname=f_chi,
            title="",
            show_legend=False,
        )
        overlaid_corner(
            samples_list,
            samples_labels,
            samples_colors,
            params=["cos_tilt_1", "cos_tilt_2", "cos_theta_12"],
            fname=f_theta,
            title="",
        )

        images = [[Image.open(f_chi), Image.open(f_theta)]]

        combined_image = join_images(
            *images, bg_color="white", alignment=(0.5, 0.5)
        )
        combined_image.save(f_combo)
        i += 1


ISO_SAMPLES = get_samples_from_prior(
    num_samples=1000, prior_fname="isotropic.prior"
)
ALIGNED_SAMPLES = get_samples_from_prior(
    num_samples=1000, prior_fname="aligned.prior"
)


def plot_one_corner(
    show_aligned, show_isotropic, show_agn, n=NUM, sig1=0, sig12=0
):
    plt.close("all")
    f_theta = f"tmp_theta.png"
    f_chi = f"tmp_chi.png"
    agn_label = (
        r"$\sigma_{1}=" + f"{sig1:.1f}" + r", \sigma_{12}=" + f"{sig12:.1f}$"
    )
    samples_list, samples_colors, samples_labels = [], [], []
    if show_aligned:
        samples_list.append(ALIGNED_SAMPLES)
        samples_colors.append("tab:green")
        samples_labels.append("Field")
    if show_isotropic:
        samples_list.append(ISO_SAMPLES)
        samples_colors.append("tab:orange")
        samples_labels.append("Dynamic")
    if show_agn:
        agn_samples = sample_from_agn_population(
            sigma_1=sig1, sigma_12=sig12, n=n
        )
        samples_list.append(agn_samples)
        samples_colors.append("tab:blue")
        samples_labels.append("AGN " + agn_label)

    overlaid_corner(
        samples_list,
        samples_labels,
        samples_colors,
        params=["chi_eff", "chi_p"],
        fname=f_chi,
        title="",
        show_legend=False,
    )
    overlaid_corner(
        samples_list,
        samples_labels,
        samples_colors,
        params=["cos_tilt_1", "cos_tilt_2", "cos_theta_12"],
        fname=f_theta,
        title="",
    )

    images = [[Image.open(f_chi), Image.open(f_theta)]]

    combined_image = join_images(
        *images, bg_color="white", alignment=(0.5, 0.5)
    )
    plt.axis("off")
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.asarray(combined_image), aspect="equal")


def gif_main():
    sigma12_vals = list(np.arange(0.3, 3, 0.3))
    sigma1_vals = [1 for _ in range(len(sigma12_vals))]

    outdir = "vary_sigma12"
    plot_overlaid_corners(
        sigma_1_list=sigma1_vals, sigma_12_list=sigma12_vals, pltdir=outdir
    )
    # save_gif("vary_12.gif", outdir=outdir, loop=True, regex="*_combo.png")

    sigma1_vals = list(np.arange(0.3, 3, 0.3))
    sigma12_vals = [0.1 for _ in range(len(sigma1_vals))]

    outdir = "vary_sigma1"
    plot_overlaid_corners(
        sigma_1_list=sigma1_vals, sigma_12_list=sigma12_vals, pltdir=outdir
    )
    # save_gif("vary_1.gif", outdir=outdir, loop=True, regex="*_combo.png")


def test_main():
    # plot_one_corner(sigma_1=0.01, sigma_12=0.01)
    gif_main()


if __name__ == "__main__":
    test_main()
