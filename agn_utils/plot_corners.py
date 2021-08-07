import glob
import os

import bilby
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import add_cos_theta_12_from_component_spins, add_snr, result_post_processing
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner
from bilby.gw.conversion import generate_spin_parameters, generate_mass_parameters, convert_to_lal_binary_black_hole_parameters


import argparse

from fpdf import FPDF
from tqdm.auto import tqdm

plt.rcParams['font.size'] = 10


def generate_corner(r,  plot_params, bilby_corner=True, labels=[]):
    if bilby_corner:
        fig = r.plot_corner(
            truths=True,
            parameters={k: r.injection_parameters[k] for k in plot_params},
            priors=True,
            save=False,
            dpi=150,
            label_kwargs=dict(fontsize=30),
            labels=labels
        )
    else:
        # prior_samples = pd.DataFrame(priors.sample(10000))
        # prior_samples = process_samples(prior_samples, r.reference_frequency)
        fig = overlaid_corner(
            [r.posterior],
            ["Posterior", "Truth"],
            params=plot_params,
            samples_colors=["lightgray", "tab:blue", "tab:orange"],
            truths={k: r.injection_parameters[k] for k in plot_params},
            # ranges=ranges,
            quants=False
        )
    return fig


def make_plots(regex, outdir):
    files = glob.glob(regex)
    plot_dir = outdir
    os.makedirs(plot_dir, exist_ok=True)
    image_paths = []

    for f in tqdm(files, desc="Plotting corners"):
        r = bilby.gw.result.CBCResult.from_json(f)
        r = result_post_processing(r)
        plt.rcParams["text.usetex"] = False
        # r.plot_marginals(priors=True,outdir=outdir, dpi=60)
        fname = os.path.basename(f).replace(".json", ".png")
        fpath = os.path.join(plot_dir, fname)
        plot_params = ['cos_tilt_1', 'cos_tilt_2', 'cos_theta_12', 'chirp_mass', "luminosity_distance"]
        labels = ['cos t1', 'cos t2', 'cos t12', 'Mc', 'dl']
        # priors = bilby.prior.PriorDict(filename=PRIORS)
        fig = generate_corner(r,  plot_params, labels=labels)
        dl, m , s = r.injection_parameters['luminosity_distance'], r.injection_parameters['mass_1'], r.injection_parameters['snr']
        plt.suptitle(f"$dl={dl:.2f}$\n$m={m:.2f}$\n$SNR={s:.2f}$")
        # plt.tight_layout()
        fig.savefig(fpath)
        plt.close('all')
        image_paths.append(fpath)


    make_pdf(pdf_fname=f"{plot_dir}/corners.pdf", image_path_list=image_paths)


def make_pdf(pdf_fname, image_path_list):
    cover = Image.open(image_path_list[0])
    width, height = cover.size

    pdf = FPDF(unit="pt", format=[width, height])

    for page in tqdm(image_path_list, desc="Compiling PDF"):
        pdf.add_page()
        pdf.image(page, 0, 0)

    pdf.output(pdf_fname, "F")


def create_and_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex",type=str)
    parser.add_argument("--outdir",type=str)
    return args


def main():
    args = create_and_parse_args()
    make_plots(regex=args.regex, outdir=args.outdir)
    print("Complete! :)")


if __name__ == "__main__":
    main()
