import pandas as pd
import pickle
from agn_utils.data_formetter import dl_to_ld, ld_to_dl
import glob
from tqdm.auto import tqdm
REGEX = dict(
    pop_a="/fred/oz980/avajpeyi/projects/agn_phenomenological_model/studies/comoving_dist_injections/outdir_pop_a_validsnr_clean/out_*/result/*_reweighted.h5",
    pop_b="/fred/oz980/avajpeyi/projects/agn_phenomenological_model/studies/comoving_dist_injections/outdir_pop_b_validsnr_clean/out_*/result/*_reweighted.h5",
)

def store_posteriors_in_pkl(regex, label):
    files = glob.glob(regex)
    data, fnames = [], []
    for f in tqdm(files, desc="Extracting posteriors"):
        df = pd.read_hdf(f, 'samples')
        fnames.append(f)
        data.append(df.to_dict('list'))
    data = ld_to_dl(data)

    archive_fname = f"{label}.pkl"
    print(f"Saving {archive_fname}")

    with open(archive_fname, 'wb') as f:
        pickle.dump(dict(posteriors=data, fnames=fnames), f)

def main():
    store_posteriors_in_pkl(REGEX['pop_a'])
    store_posteriors_in_pkl(REGEX['pop_b'])

if __name__ == '__main__':
    main()