import argparse
import glob
import os

import bilby
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF
from tqdm.auto import tqdm

from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import result_post_processing
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner
from bilby.gw.conversion import luminosity_distance_to_redshift

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

import corner


def make_plots(regex, outdir):
    files = glob.glob(regex)
    plot_dir = outdir
    os.makedirs(plot_dir, exist_ok=True)
    image_paths = []

    for f in tqdm(files, desc="Plotting corners"):
        r = bilby.gw.result.CBCResult.from_json(f)
        r = result_post_processing(r)
        z = luminosity_distance_to_redshift(r.injection_parameters['luminosity_distance'])
        mc = r.injection_parameters['chirp_mass']
        mc_up = mc * (1+z)
        mc_down = mc / (1+z)
        r.injection_parameters['chirp_mass'] = mc_down
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


        plt.hist(r.posterior['chirp_mass'], density=True, alpha=0.2, label="posterior['chirp_mass']")
        plt.hist(r.posterior['chirp_mass_source'], density=True, alpha=0.2, label="posterior['chirp_mass_source']")
        plt.axvline(mc, c='tab:orange', label=f'mc (@ z = {z:.2f})')
        plt.axvline(mc_down, c='tab:blue', label = 'mc / (1+z)')
        plt.axvline(mc_up, c='tab:green', label='mc * (1+z)')
        plt.legend()
        plt.xlabel('chirp_mass (mc)')
        plt.tight_layout()
        plt.savefig(fpath.replace('.png','_chirp.png'))
        plt.close('all')

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
