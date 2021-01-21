import os, glob
from tqdm import tqdm
import pesummary
from pesummary.io import read, write
import pandas as pd



outdir = "../data/downsampled_pe_samples"
os.makedirs(outdir, exist_ok=True)
MAX_NUM_SAMPLES = 5000
files = glob.glob("/home/avi.vajpeyi/projects/jupyter_notebooks/GW190408_181802/GW190408_181802.h5")
for res_file in tqdm(files, total=len(files)):
    event_name = os.path.basename(res_file).split(".h")[0].split("_")[0]
    posterior_filename = f"{outdir}/{event_name}.csv"
    prior_filename = f"{outdir}/{event_name}_prior.csv"

    if os.path.isfile(posterior_filename) and os.path.isfile(prior_filename):
        pass
    else:
        try:
            print(f"Opening {res_file}")
            data = pesummary.io.read(res_file)
            print(data.summary)
            if len(data.samples) > MAX_NUM_SAMPLES:
                data.downsample(MAX_NUM_SAMPLES)
            
            print('Found run labels:')
            print(data.labels)
            samples_dict = data.samples_dict
            posterior_samples = samples_dict['PublicationSamples']
            parameters = sorted(list(posterior_samples.keys()))
            print(parameters)
            prior_samples = pd.DataFrame(data.priors['samples']).to_numpy()
            kwargs = dict(parameters=parameters, file_format="csv")

            if not os.path.isfile(posterior_filename):
                write(samples=posterior_samples,
                                   filename=posterior_filename, **kwargs)
            if not os.path.isfile(prior_filename):
                write(samples=prior_samples, filename=prior_filename,
                                   **kwargs)
        except Exception as e:
            print(f"ERROR {event_name}: {e}")

