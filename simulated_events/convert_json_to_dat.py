import glob
import os

import pandas as pd
from bilby.gw.result import CBCResult
from tqdm import tqdm

DAT_OUTDIR = "simulated_event_samples"
JSON_REGEX = "outdir_agn_injections/result/*result.json"


def main():
    if not os.path.isdir(DAT_OUTDIR):
        os.makedirs(DAT_OUTDIR)
    res_fnames = glob.glob(JSON_REGEX)
    print(f"Converting {len(res_fnames)} result json-->dat")
    for fn in tqdm(res_fnames, desc="Converting File", total=len(res_fnames)):
        save_cbc_dat(CBCResult.from_json(fn).posterior, fn)
    print("COMPLETE.")


def save_cbc_dat(posterior: pd.DataFrame, old_fn: str):
    new_fn = os.path.basename(old_fn).replace(".json", ".dat")
    new_fn = os.path.join(DAT_OUTDIR, new_fn)
    posterior.to_csv(new_fn, sep=" ", index=False)


if __name__ == "__main__":
    main()