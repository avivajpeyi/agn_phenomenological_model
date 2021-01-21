import os, glob
from tqdm import tqdm
import pesummary.io

import logging

# logging.getLogger("PESummary").setLevel(logging.ERROR)

outdir = "../data/downsampled_pe_samples"
os.makedirs(outdir, exist_ok=True)
MAX_NUM_SAMPLES = 5000
files = glob.glob("/home/shanika.galaudage/O3/population/o3a_pe_samples_release/*.h5")
for res_file in tqdm(files, total=len(files)):
    event_name = os.path.basename(res_file).split(".h")[0].split("_")[0]
    posterior_filename = f"{outdir}/{event_name}.csv"
    prior_filename = f"{outdir}/{event_name}_prior.csv"

    if os.path.isfile(posterior_filename) and os.path.isfile(prior_filename):
        pass
    else:
        try:
            data = pesummary.io.read(res_file)
            if len(data.samples) > MAX_NUM_SAMPLES:
                data.downsample(MAX_NUM_SAMPLES)

            parameters = data.parameters
            posterior_samples = data.samples
            prior_samples = pd.DataFrame(data.priors['samples']).to_numpy()
            kwargs = dict(parameters=parameters, file_format="csv")

            if not os.path.isfile(posterior_filename):
                pesummary.io.write(samples=posterior_samples,
                                   filename=posterior_filename, **kwargs)
            if not os.path.isfile(prior_filename):
                pesummary.io.write(samples=prior_samples, filename=prior_filename,
                                   **kwargs)
        except Exception as e:
            print(f"ERROR {event_name}: {e}")

