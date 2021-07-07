import glob
import os

import bilby
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import add_cos_theta_12_from_component_spins, add_snr
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner
from bilby.gw.conversion import generate_spin_parameters, generate_mass_parameters, convert_to_lal_binary_black_hole_parameters

from fpdf import FPDF
from tqdm.auto import tqdm



def process_samples(s, rf):
    s['reference_frequency'] = rf
    s, _ = convert_to_lal_binary_black_hole_parameters(s)
    s = generate_mass_parameters(s)
    s = generate_spin_parameters(s)
    s = add_cos_theta_12_from_component_spins(s)
    s = add_snr(s)
    s['snr'] = s['network_snr']
    return s


def result_post_processing(r):
    r.posterior = add_cos_theta_12_from_component_spins(r.posterior)
    r.injection_parameters = process_samples(r.injection_parameters, r.reference_frequency)
    return r


def generate_corner(r,  plot_params, bilby_corner=True, ):
    # ranges=[(-1, 1), (-1, 1), (-1, 1), (14, 40), (100, 10000)]
    if bilby_corner:
        fig = r.plot_corner(
            truths=True,
            parameters={k: r.injection_parameters[k] for k in plot_params},
            # range=ranges,ranges
            # priors=priors,
            save=False
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
        r.plot_marginals(priors=True,outdir=outdir, dpi=60)
        fname = os.path.basename(f).replace(".json", ".png")
        fpath = os.path.join(plot_dir, fname)
        plot_params = ['cos_tilt_1', 'cos_tilt_2', 'cos_theta_12', 'mass_1', "luminosity_distance"]
        # priors = bilby.prior.PriorDict(filename=PRIORS)
        fig = generate_corner(r,  plot_params)
        dl, m , s = r.injection_parameters['luminosity_distance'], r.injection_parameters['mass_1'], r.injection_parameters['snr']
        # snr = r.injection_parameters['snr']
        plt.suptitle(f"$dl={dl:.2f}$\n$m={m:.2f}$\n$SNR={s:.2f}$")
        fig.savefig(fpath)
        image_paths.append(fpath)


    # make_pdf(pdf_fname=f"{plot_dir}/corners.pdf", image_path_list=image_paths)
    margs = glob.glob(f"{outdir}/*1d/*pdf.png")
    make_pdf(pdf_fname=f"{plot_dir}/plots.pdf", image_path_list=margs)


def make_pdf(pdf_fname, image_path_list):
    cover = Image.open(image_path_list[0])
    width, height = cover.size

    pdf = FPDF(unit="pt", format=[width, height])

    for page in tqdm(image_path_list, desc="Compiling PDF"):
        pdf.add_page()
        pdf.image(page, 0, 0)

    pdf.output(pdf_fname, "F")


def main():
    make_plots("out_fixed*/*.json", outdir="fixed_dl")
    # make_plots(regex="7th_inj/*.json", outdir="7th_inj-plot")
    # make_plots(regex="bp_pop_a/*.json", outdir="bp_plot_a")
    # make_plots(regex="bp_pop_b/*.json", outdir="bp_plot_b")
    # make_plots(regex="pop_a/*.json", outdir="plot_out_a")
    # make_plots(regex="pop_b/*.json", outdir="plot_out_b")
    print("Complete! :)")


if __name__ == "__main__":
    main()
