import os
import warnings

import bilby
import pandas as pd
from agn_utils.bbh_population_generators import get_bbh_population_from_agn_params
from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import add_snr, add_signal_duration

warnings.filterwarnings('ignore')

PRIOR_PATH = "data/bbh.prior"

POPULATION_A = dict(sigma_1=0.5,
                    sigma_12=3)

POPULATION_B = dict(sigma_1=1,
                    sigma_12=0.25)

REQUIRED_PARAMS = [
    'dec',
    'ra',
    'theta_jn',
    'psi',
    'phase',
    'geocent_time',
    'a_1',
    'a_2',
    'cos_tilt_1',
    'cos_tilt_2',
    'phi_z_s12',
    'phi_jl',
    'mass_ratio',
    'chirp_mass',
    'luminosity_distance',
    'duration'
]

POPS = dict(pop_a=POPULATION_A, pop_b=POPULATION_B)


def check_if_injections_in_prior(injection_df: pd.DataFrame, prior_path: str):
    priors = bilby.prior.PriorDict(filename=prior_path)
    assert set(priors.keys()).issubset(set(injection_df.columns))
    in_prior = pd.DataFrame(index=injection_df.index)
    for prior in priors.values():
        inj_param = injection_df[prior.name].values
        in_prior[f"in_{prior.name}_prior"] = (prior.minimum <= inj_param) & (inj_param <= prior.maximum)
    not_in_prior = in_prior[in_prior.isin([False]).any(axis=1)]  # get all rows where inj_param outside pri range
    if len(not_in_prior) > 0:
        print(f"The following injection id(s) have parameters outside your prior range: {list(not_in_prior.T.columns)}")
    return list(list(not_in_prior.T.columns))


def keep_injections_inside_prior(df, prior_path):
    invalid_ids = check_if_injections_in_prior(df, prior_path)
    df = df.drop(index=invalid_ids)
    return df


def filter_undesired_injections(df, prior_path):
    df = df[df['network_snr'] >= 60]
    df = df[df['duration'] == 4]
    df = keep_injections_inside_prior(df, prior_path)
    return df

def generate_population(pop_name, pop_params):
    fname = f"data/{pop_name}.dat"
    num_high_snr = 0
    iteration = 0
    if os.path.exists(fname):
        cached_pop = pd.read_csv(fname, sep=' ')
        num_high_snr = len(cached_pop[cached_pop['network_snr'] >= 60])

    while (num_high_snr < 40):
        pop_df = get_bbh_population_from_agn_params(
            num_samples=1000,
            **pop_params
        )
        pop_df = add_snr(pop_df)
        pop_df = add_signal_duration(pop_df)

        if os.path.exists(fname):
            cached_pop = pd.read_csv(fname, sep=' ')
            pop_df = pop_df.append(cached_pop, ignore_index=True)
        num_high_snr = len(pop_df[pop_df['network_snr'] >= 60])
        print(f"it-{iteration:02}: # high SNR events: {num_high_snr} in {len(pop_df):04} BBH")
        pop_df.to_csv(fname, sep=' ', mode='w', index=False)
        iteration += 1

    cached_pop = pd.read_csv(fname, sep=' ')
    high_snr_events = filter_undesired_injections(cached_pop, PRIOR_PATH)
    high_snr_events = high_snr_events.loc[:, REQUIRED_PARAMS]
    high_snr_events = high_snr_events.reset_index(drop=True)
    high_snr_events.to_csv(fname.replace('.dat', '_highsnr.dat'), index=False, sep=' ')


if __name__ == "__main__":
    for pop_name, pop_params in POPS.items():
        generate_population(pop_name, pop_params)
