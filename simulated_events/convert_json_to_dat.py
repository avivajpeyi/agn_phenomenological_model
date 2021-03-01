import glob
import os

import pandas as pd
from bilby.gw.result import CBCResult
from tqdm import tqdm

DAT_OUTDIR = "simulated_event_samples"
JSON_REGEX = "outdir_agn_injections/result/*result.json"

PARAM = [
    'a_1',
    'a_2',
    'cos_tilt_1',
    'cos_theta_12',
    'mass_ratio',
    'mass_1',
    'redshift',
]


def main():
    if not os.path.isdir(DAT_OUTDIR):
        os.makedirs(DAT_OUTDIR)
    res_fnames = glob.glob(JSON_REGEX)
    print(f"Converting {len(res_fnames)} result json-->dat")
    for fn in tqdm(res_fnames, desc="Converting File", total=len(res_fnames)):
        save_cbc_dat(CBCResult.from_json(fn).posterior, fn)
    print("COMPLETE.")


def save_cbc_dat(posterior: pd.DataFrame, old_fn: str):
    posterior['mass_1'] = posterior["mass_1_source"]
    posterior["mass_2"] = posterior["mass_2_source"]
    posterior = posterior[PARAM]
    new_fn = os.path.basename(old_fn).replace(".json", ".dat")
    new_fn = os.path.join(DAT_OUTDIR, new_fn)
    posterior.to_csv(new_fn, sep=" ", index=False)

if __name__ == "__main__":
    main()
