import glob
import os

import bilby
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF
from tqdm.auto import tqdm

from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import add_cos_theta_12_from_component_spins

RES_REGEX = "out*/res*/sn*.json"
PRIORS = "./datafiles/bbh.prior"


def make_plots():
    files = glob.glob(RES_REGEX)
    plot_dir = "plot_out"
    os.makedirs(plot_dir, exist_ok=True)
    image_paths = []

    for f in tqdm(files, desc="Plotting corners"):
        r = bilby.gw.result.CBCResult.from_json(f)
        r.posterior = add_cos_theta_12_from_component_spins(r.posterior)
        r.injection_parameters = add_cos_theta_12_from_component_spins(r.injection_parameters)
        fname = os.path.basename(f).replace(".json", ".png")
        fpath = os.path.join(plot_dir, fname)
        plot_params = ['cos_tilt_1', 'cos_tilt_2', 'cos_theta_12', 'mass_1', "luminosity_distance"]
        priors = bilby.prior.PriorDict(filename=PRIORS)
        fig = r.plot_corner(
            truths=True,
            parameters={k: r.injection_parameters[k] for k in plot_params},
            range=[(-1, 1), (-1, 1), (-1, 1), (14, 40), (100, 500)],
            priors=priors,
            save=False
        )
        dl, m = r.injection_parameters['luminosity_distance'], r.injection_parameters['mass_1']
        snr = r.injection_parameters['snr']
        plt.suptitle(f"$dl={dl:.2f}$\n$m={m}$\n$snr={snr:.2f}$")
        fig.savefig(fpath)
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


def main():
    make_plots()
    print("Complete! :)")

if __name__ == "__main__":
    main()
