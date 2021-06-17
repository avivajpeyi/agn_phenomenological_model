import glob
import os

import bilby
from tqdm.auto import tqdm

RES_REGEX = "out*/res*/sn*.json"
PRIORS = "/home/avi.vajpeyi/projects/agn_phenomenological_model/studies/fixed_snr/datafiles/bbh.prior"


def main():
    files = glob.glob(RES_REGEX)
    plot_dir = "plot_out"
    print(f"Plotting corners for {len(files)} files.")
    os.makedirs(plot_dir, exist_ok=True)
    prior=bilby.prior.PriorDict.from_file(PRIORS)
    for f in tqdm(files):
        r = bilby.gw.result.CBCResult.from_json(f)
        fname = os.path.basename(f).replace(".json", ".png")
        fpath = os.path.join(plot_dir, fname)
        r.plot_corner(
            filename=fpath, truths=True,
            parameters=['tilt_1', 'tilt_2', 'chirp_mass', "luminosity_distance"],
            priors=prior
        )


if __name__ == "__main__":
    main()
