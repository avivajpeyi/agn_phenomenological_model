import pandas as pd
import glob
import tqdm
import re
import os

SAMPLE_SRC = "/home/avi.vajpeyi/projects/agn_phenomenological_model/data/gwtc_samples/"
PARAM_TO_KEEP = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2", "mass_ratio", "mass_1",
                 "mass_2", "redshift", "cos_theta_12", "chi_p", "chi_eff"]

posterior_files = [f for f in glob.glob(SAMPLE_SRC + "*.csv") if "prior" not in f]
prior_files = glob.glob(SAMPLE_SRC + "*_prior.csv")

reject_events = ["GW170817", "S190425z", "S190426c", "S190719an", "190814", "S190909w",
                 "S190930ak"]
posts = []
events = []


def get_event_name(fname):
    return re.findall(r"(\w*\d{6}[a-z]*)", fname)[0]


def add_agn_samples_to_df(df):
    df['s1x'], df['s1y'], df['s1z'] = df['spin_1x'], df['spin_1y'], df['spin_1z']
    df['s2x'], df['s2y'], df['s2z'] = df['spin_2x'], df['spin_2y'], df['spin_2z']
    df['s1_dot_s2'] = (df['s1x'] * df['s2x']) + (df['s1y'] * df['s2y']) + (
            df['s1z'] * df['s2z'])
    df['s1_mag'] = np.sqrt(
        (df['s1x'] * df['s1x']) + (df['s1y'] * df['s1y']) + (df['s1z'] * df['s1z']))
    df['s2_mag'] = np.sqrt(
        (df['s2x'] * df['s2x']) + (df['s2y'] * df['s2y']) + (df['s2z'] * df['s2z']))
    df['cos_theta_12'] = df['s1_dot_s2'] / (df['s1_mag'] * df['s2_mag'])
    return df


def read_posteriors(posterior_files):
    # posterior_files = [f for f in glob.glob(SAMPLE_SRC+"*.csv") if "prior" not in f]
    posteriors = {}
    for f in tqdm.tqdm(posterior_files, desc='Posteriors'):
        event = get_event_name(f)
        if event not in reject_events:
            d = pd.read_csv(f, sep=" ", index_col=False)
            try:
                d = add_agn_samples_to_df(d)
                posteriors.update({f: d[PARAM_TO_KEEP]})
            except Exception as e:
                print(f"Skipping {event}: {e}\n{d.columns.values.tolist()}")
    return posteriors


posteriors = read_posteriors(posterior_files)
# priors = {f:pd.read_csv(f,sep=" ", index_col=False) for f in tqdm.tqdm(prior_files, desc="Priors")}


agn_samples = "/home/avi.vajpeyi/projects/agn_phenomenological_model/data/gwtc_agn_samples/"
os.mkdir(agn_samples)

for fname, posterior in posteriors.items():
    new_f = os.path.join(agn_samples, os.path.basename(fname))
    posterior.to_csv(new_f, index=False, sep=' ')