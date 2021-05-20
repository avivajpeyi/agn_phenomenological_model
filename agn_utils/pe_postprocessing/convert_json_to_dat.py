import glob
import os
import sys

import pandas as pd
from bilby.gw.result import CBCResult
from tqdm import tqdm

PARAM = [
    "a_1",
    "a_2",
    "cos_tilt_1",
    "mass_ratio",
    "mass_1",
    "redshift",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
]


def save_cbc_dat(posterior: pd.DataFrame, old_fn: str, new_dir: str):
    posterior["mass_1"] = posterior["mass_1_source"]
    posterior["mass_2"] = posterior["mass_2_source"]
    posterior = posterior[PARAM]
    new_fn = os.path.basename(old_fn).replace(".json", ".dat")
    new_fn = os.path.join(new_dir, new_fn)
    posterior.to_csv(new_fn, sep=" ", index=False)


def process_res(
    dat_outdir="simulated_event_samples",
    json_regex="bilby_pipe_jobs/out*/result/*result.json",
):
    print(f"Getting res from {json_regex} and sving to {dat_outdir}")
    if not os.path.isdir(dat_outdir):
        os.makedirs(dat_outdir)
    res_fnames = glob.glob(json_regex)
    print(f"Converting {len(res_fnames)} result json-->dat")
    for fn in tqdm(res_fnames, desc="Converting File", total=len(res_fnames)):
        try:
            save_cbc_dat(
                CBCResult.from_json(fn).posterior, fn, new_dir=dat_outdir
            )
        except Exception as e:
            print(f"ERROR PROCESSING {fn}: {e}. SKIPPING")
    print("COMPLETE.")


import pandas as pd

MAX_SAMPLES = 5000


def downsampler():
    fnames = []
    lens = []
    files = glob.glob("../data/gwtc_samples/*.csv")
    for f in tqdm.tqdm(files):
        try:
            lens.append(len(pd.read_csv(f, sep=" ")))
            fnames.append(f)
        except Exception as e:
            print(f"ERROR {f}: {e}")

    samples_count = pd.DataFrame(dict(fnames=fnames, lens=lens))
    samples_count = samples_count[samples_count["lens"] > MAX_SAMPLES]
    samples_count = samples_count.reset_index(drop=True)
    print(samples_count)

    for i in range(len(samples_count)):
        fname = samples_count.iloc[i]["fnames"]
        df = pd.read_csv(fname, sep=" ").sample(MAX_SAMPLES)
        df.to_csv(fname, sep=" ", index=False)
        print(f"{fname} len {samples_count.iloc[i]['lens']}-->{MAX_SAMPLES}")


def main():
    dat_outdir = sys.argv[1]
    json_regex = sys.argv[2]
    process_res(dat_outdir, json_regex)


if __name__ == "__main__":
    main()
