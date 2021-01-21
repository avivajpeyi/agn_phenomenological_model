"""Very hacky script to grab the "PublicationSamples" and priors from the GWTC2 cat"""
import os, glob
from tqdm import tqdm
import pesummary
from pesummary.io import read
import pandas as pd
import numpy as np

MAX_NUM_SAMPLES = 5000
SOURCE = "/home/zoheyr.doctor/public_html/O3/O3aCatalog/data_release/all_posterior_samples/*.h5"

if __name__ == '__main__':
    outdir = "../data/gwtc2_samples"
    os.makedirs(outdir, exist_ok=True)
    files = glob.glob(SOURCE)
    files = [f for f in files if "comoving" not in f]
    for res_file in tqdm(files, total=len(files)):
        event_name = os.path.basename(res_file).split(".h")[0]
        samp_fname = f"{outdir}/{event_name}.csv"
        prior_fname = f"{outdir}/{event_name}_prior.csv"
        print(f"Checking fname {samp_fname}")
        if not (os.path.isfile(samp_fname)):
            print(f"Opening {res_file}")
            data = pesummary.io.read(res_file)
            if len(data.samples) > MAX_NUM_SAMPLES:
                data.downsample(MAX_NUM_SAMPLES)
            samples_dict = data.samples_dict
            samples = pd.DataFrame(samples_dict['PublicationSamples'])
            samples.to_csv(samp_fname, sep=' ', index=False)
            priors_array = np.load(res_file.replace(".h5", "_prior.npy"))
            priors = pd.DataFrame(priors_array).sample(MAX_NUM_SAMPLES)
            priors.to_csv(prior_fname, sep=' ', index=False)
