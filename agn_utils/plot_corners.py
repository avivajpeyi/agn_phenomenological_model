import argparse
import glob
import os

import bilby
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF
from tqdm.auto import tqdm
import pickle
from agn_utils.pe_postprocessing.ln_likelihood_calc import get_lnL
from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import result_post_processing    , process_samples
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner
from bilby.gw.conversion import luminosity_distance_to_redshift
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = 10

LABELS = dict(
    a_1='a1',
    a_2='a2',
    cos_tilt_1='cos t1',
    cos_tilt_2='cos t2',
    cos_theta_12='cos t2',
    phi_12='phi_12',
    phi_jl='phi_jl',
    theta_jn='theta_jn',
    mass_1=r'$m1_{\mathrm{lab}}$',
    mass_2=r'$m1_{\mathrm{lab}}$',
    chirp_mass=r'$\mathcal{M}_{\mathrm{lab}}$',
    mass_1_source=r'$m1_{\mathrm{source}}$',
    mass_2_source=r'$m1_{\mathrm{source}}$',
    chirp_mass_source=r'$\mathcal{M}_{\mathrm{source}}$',
    luminosity_distance='dl',
    ra='ra',
    dec='dec',
    psi='psi'
)


def generate_corner(r, plot_params, bilby_corner=True):
    if bilby_corner:
        fig = r.plot_corner(
            truths=True,
            parameters={k: r.injection_parameters[k] for k in plot_params},
            priors=True,                                        
            save=False,
            dpi=150,
            label_kwargs=dict(fontsize=30),
            labels=[LABELS[p] for p in plot_params]
        )
    else:
        prior_samples = pd.DataFrame(r.priors.sample(10000))
        prior_samples = process_samples(prior_samples, r.reference_frequency)
        # ranges = [(r.priors[p].minimum, r.priors[p].maximum) for p in plot_params]
        ranges = [(min(r.posterior[p]), max(r.posterior[p])) for p in plot_params]
        fig = overlaid_corner(
            [prior_samples, r.posterior],
            ["Prior", "Posterior", "Truth"],
            params=plot_params,
            samples_colors=["lightgray", "tab:blue", "tab:orange"],
            truths={k: r.injection_parameters[k] for k in plot_params},
            ranges=ranges,
            quants=False
        )
    return fig

import corner


def make_plots(regex, outdir, compile_pdf=True):
    files = glob.glob(regex)
    plot_dir = outdir
    os.makedirs(plot_dir, exist_ok=True)
    image_paths = []

    for f in tqdm(files, desc="Plotting corners"):
        r = bilby.gw.result.CBCResult.from_json(f)
        r = result_post_processing(r)
        z = luminosity_distance_to_redshift(r.injection_parameters['luminosity_distance'])
        mc = r.injection_parameters['chirp_mass']

        r.priors['cos_theta_12'] = bilby.prior.Uniform(-1, 1)
        plt.rcParams["text.usetex"] = False
        fname = os.path.basename(f).replace(".json", ".png")
        plot_params = ['cos_tilt_1', 'cos_tilt_2', 'cos_theta_12', 'phi_12',  'phi_jl', 'theta_jn',
                       'chirp_mass', "luminosity_distance", 'ra', 'dec', 'psi']
        fig = generate_corner(r, plot_params)
        dl, m = r.injection_parameters['luminosity_distance'], r.injection_parameters['chirp_mass']
        plt.suptitle(f"${r.label.replace('_','-')}$\n$dl={dl:.2f}$\n$mc={mc:.2f}$\n$mc/(1+z)={mc_down:.2f}$",fontsize=30)
        fpath = os.path.join(plot_dir, fname)
        fig.savefig(fpath)
        plt.close('all')
        image_paths.append(fpath)

    if compile_pdf:
        make_pdf(pdf_fname=f"{plot_dir}/corners.pdf", image_path_list=image_paths)


def get_data_dump_path(res_path):
    try:
        res_dir = os.path.dirname(res_path)
        data_dump_fname = os.path.basename(res_path).replace("0_result.json", "data_dump.pickle")
        outdir = os.path.dirname(res_dir)
        data_dir = os.path.join(outdir, "data")
        data_dump_path =  os.path.join(data_dir, data_dump_fname)
        return data_dump_path
    except Exception as e:
        print(e)
        return ""

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
    parser.add_argument("--regex", type=str)
    parser.add_argument("--outdir", type=str)
    args = parser.parse_args()
    return args


def main():
    args = create_and_parse_args()
    make_plots(regex=args.regex, outdir=args.outdir)
    print("Complete! :)")


if __name__ == "__main__":
    main()
