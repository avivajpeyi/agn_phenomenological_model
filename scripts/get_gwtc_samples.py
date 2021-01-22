"""Hacky script to grab the "PublicationSamples" and priors from the GWTC1+2 cat
Note: runtime ~2hrs
"""
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd
import pesummary
import tqdm
from pesummary.io import read

MAX_NUM_SAMPLES = 5000
GWTC2_SOURCE = "/home/zoheyr.doctor/public_html/O3/O3aCatalog/data_release/all_posterior_samples/*.h5"
GWTC1_SOURCE = "/home/shanika.galaudage/O3/population/GWTC-1_sample_release/*.hdf5"
OUTDIR = "../data/gwtc_samples"


def safe_run(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            tb = traceback.format_tb(exc_tb)
            tb = "||".join(tb)
            print(
                f"ERROR: {exc_type}\n" f"Traceback: [{tb}]\n" f"Message: {e}"
            )
            return None

    return func_wrapper


@safe_run
def process_gwtc2_file(res_file, outdir):
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


@safe_run
def process_gwtc1_file(res_file, outdir):
    event_name = os.path.basename(res_file).split(".h")[0].split("_")[0]
    posterior_filename = f"{outdir}/{event_name}.csv"
    prior_filename = f"{outdir}/{event_name}_prior.csv"
    print(f"Checking fname {posterior_filename}")
    if not (os.path.isfile(posterior_filename)):
        print(f"Opening {res_file}")
        data = pesummary.io.read(res_file)
        if len(data.samples) > MAX_NUM_SAMPLES:
            data.downsample(MAX_NUM_SAMPLES)
        parameters = data.parameters
        posterior_samples = data.samples
        prior_samples = pd.DataFrame(data.priors['samples']).sample(MAX_NUM_SAMPLES)
        prior_samples = prior_samples.to_numpy()
        kwargs = dict(parameters=parameters, file_format="csv")
        if not os.path.isfile(posterior_filename):
            pesummary.io.write(samples=posterior_samples,
                               filename=posterior_filename, **kwargs)
        if not os.path.isfile(prior_filename):
            pesummary.io.write(samples=prior_samples, filename=prior_filename,
                               **kwargs)


def main():
    gwtc1_files = glob.glob(GWTC1_SOURCE)
    gwtc2_files = glob.glob(GWTC2_SOURCE)
    gwtc2_files = [f for f in gwtc2_files if "comoving" not in f]
    os.makedirs(OUTDIR, exist_ok=True)
    print("Processing GWTC1")
    for f in tqdm.tqdm(gwtc1_files):
        process_gwtc1_file(f, OUTDIR)
    print("Processing GWTC2")
    for f in tqdm.tqdm(gwtc2_files):
        process_gwtc2_file(f, OUTDIR)


if __name__ == '__main__':
    main()
