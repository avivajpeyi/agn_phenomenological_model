import os
import warnings
from typing import Dict, List

import bilby
import numpy as np
import pandas as pd
from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import (
    get_component_mass_from_mchirp_q, add_snr, add_signal_duration
)
from agn_utils.bbh_population_generators.spin_conversions import (agn_to_cartesian_coords,
                                                                  cartesian_to_spherical_coords,
                                                                  spherical_to_cartesian_coords)
from agn_utils.plotting.plot_cos_dist import main_plotter
from bilby.core.prior import TruncatedNormal, Uniform
from deepdiff import DeepDiff

import argparse

warnings.filterwarnings('ignore')


POPULATION_A = dict(sigma_1=0.5, sigma_12=3)

POPULATION_B = dict(sigma_1=1, sigma_12=0.25)

POPS = dict(pop_a=POPULATION_A, pop_b=POPULATION_B)

REF_FREQ = 20.0

REQUIRED_PARAMS = [
    "chirp_mass",
    "mass_ratio",
    "a_1",
    "a_2",
    "cos_tilt_1",
    "cos_tilt_2",
    "phi_1",
    "phi_12",
    "phi_jl",
    "luminosity_distance",
    "dec",
    "ra",
    "theta_jn",
    "psi",
    "phase",
    "incl",
    "geocent_time"
]


def ensure_no_duplicated_params(params):
    expected_params = {
        "chirp_mass", "mass_ratio",
        "a_1", "a_2", "phi_1", "cos_theta_1", "phi_z_s12", "cos_theta_12", "incl",
        "ra", "dec", "luminosity_distance", "geocent_time", "psi", "phase", "reference_frequency"
    }
    diff = DeepDiff(set(params), expected_params)

    if len(diff) != 0:
        raise ValueError(f"Params passed dont match expected params: \n{diff}")


def convert_s12_samples_to_s2_samples(s: Dict[str, List[float]]) -> Dict[str, List[float]]:
    ensure_no_duplicated_params(list(s.keys()))
    num_samples = len(s['chirp_mass'])
    mass_1, mass_2 = get_component_mass_from_mchirp_q(mchirp=s["chirp_mass"], q=s["mass_ratio"])
    reference_frequency = [s['reference_frequency'] for _ in range(num_samples)]
    incl, s1x, s1y, s1z, s2x, s2y, s2z = agn_to_cartesian_coords(
        incl=s["incl"],
        phi_1=s["phi_1"],
        tilt_1=np.arccos(s["cos_theta_1"]),
        theta_12=np.arccos(s["cos_theta_12"]),
        phi_z_s12=s["phi_z_s12"],
        a_1=s["a_1"],
        a_2=s["a_2"],
        mass_1=mass_1,
        mass_2=mass_2,
        reference_frequency=reference_frequency,
        phase=s["phase"]
    )
    theta_jn, phi_jl, tilt_1, tilt_2, phi_1, phi_2, a_1, a_2, phi_12, theta_12, phi_z_s12 = cartesian_to_spherical_coords(
        incl,
        s1x, s1y, s1z, s2x, s2y, s2z,
        mass_1, mass_2, s["phase"], reference_frequency
    )
    initial_spin_vector = dict(tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, a_1=a_1, a_2=a_2, phi_jl=phi_jl,
                               theta_jn=theta_jn)

    # roundtrip check
    incl_2, s1x_2, s1y_2, s1z_2, s2x_2, s2y_2, s2z_2 = spherical_to_cartesian_coords(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, reference_frequency, s["phase"])
    theta_jn_2, phi_jl_2, tilt_1_2, tilt_2_2, phi_1_2, phi_2_2, a_1_2, a_2_2, phi_12_2, theta_12_2, phi_z_s12_2 = cartesian_to_spherical_coords(
        incl_2,
        s1x_2, s1y_2, s1z_2, s2x_2, s2y_2, s2z_2,
        mass_1, mass_2, s["phase"], reference_frequency)
    final_spin_vector = dict(tilt_1=tilt_1_2, tilt_2=tilt_2_2, phi_12=phi_12_2, a_1=a_1_2, a_2=a_2_2, phi_jl=phi_jl_2,
                             theta_jn=theta_jn_2)

    diff = DeepDiff(initial_spin_vector, final_spin_vector, math_epsilon=0.001)
    if len(diff) != 0:
        ValueError(f"ERROR: roundrip conversion not working: {diff}")

    data = dict(
        mass_1=mass_1, mass_2=mass_2,
        tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, a_1=a_1, a_2=a_2, phi_jl=phi_jl, theta_jn=theta_jn,
        ra=s["ra"], dec=s["dec"], luminosity_distance=s["luminosity_distance"], geocent_time=s["geocent_time"],
        psi=s["psi"],
        phase=s["phase"], reference_frequency=reference_frequency
    )

    return data


def create_population_prior(pop_parameters, prior_path):
    """Remove tilt angle priors, replace with population priors"""
    prior = bilby.prior.PriorDict(filename=prior_path)
    for param in ['cos_tilt_1', 'cos_tilt_2', 'phi_12', 'theta_jn', 'phi_jl']:
        prior.pop(param)
    for i in [1, 12]:
        kwargs = dict(mu=1, minimum=-1, maximum=1)
        prior[f'cos_theta_{i}'] = TruncatedNormal(sigma=pop_parameters[f'sigma_{i}'], **kwargs)
    prior[f'phi_1'] = Uniform(name='phi_1', minimum=0, maximum=2 * np.pi, boundary='periodic')
    prior[f'phi_z_s12'] = Uniform(name='phi_z_s12', minimum=0, maximum=2 * np.pi, boundary='periodic')
    prior[f'incl'] = Uniform(name='incl', minimum=0, maximum=2 * np.pi, boundary='periodic')
    return prior


def check_if_injections_in_prior(injection_df: pd.DataFrame, prior_path: str):
    priors = bilby.prior.PriorDict(filename=prior_path)
    injection_df['mass_ratio'] = injection_df['mass_2'] / injection_df['mass_1']
    missing_params = set(priors.keys()) - set(injection_df.columns)
    assert len(missing_params) == 0, f"Injection DF mising {missing_params}"
    in_prior = pd.DataFrame(index=injection_df.index)
    for prior in priors.values():
        if prior.name is not None:
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


def filter_undesired_injections(df, prior_path, number_high_snr_events=80):
    df = df[df['network_snr'] >= 50]
    df = df[df['network_snr'] <= 100]
    df = df[df['duration'] == 4]
    df = keep_injections_inside_prior(df, prior_path)
    return df


def generate_population(pop_name, pop_pri, outdir, prior_path):
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/{pop_name}.dat"

    pop_samp = pop_pri.sample(200)
    pop_samp['reference_frequency'] = REF_FREQ
    pop_samp = convert_s12_samples_to_s2_samples(pop_samp)
    pop_samp = add_snr(pop_samp)
    pop_samp = add_signal_duration(pop_samp)
    pop_samp['cos_tilt_2'] = np.cos(pop_samp['tilt_2'])
    pop_samp['cos_tilt_1'] = np.cos(pop_samp['tilt_1'])


    df = pd.DataFrame(pop_samp)
    num_high_snr = len(df[(df['network_snr'] >= 50) & (df['network_snr'] <= 100)])


    pop_df = pd.DataFrame(pop_samp)

    pop_df.to_csv(fname, sep=' ', mode='w', index=False)
    print(f"# high SNR events: {num_high_snr} in {len(pop_df)} BBH")

    cached_pop = pd.read_csv(fname, sep=' ')
    high_snr_events = filter_undesired_injections(cached_pop, prior_path)

    high_snr_events = high_snr_events.reset_index(drop=True)
    if len(high_snr_events) > 40:
        high_snr_events = high_snr_events.sample(40)
    print(f"Saving {len(high_snr_events)} SNR events")
    high_snr_events.to_csv(fname.replace('.dat', '_highsnr.dat'), index=False, sep=' ')


def main_generator(prior_path, outdir='data'):
    os.makedirs(outdir, exist_ok=True)
    for pop_name, pop_params in POPS.items():
        pop_pri = create_population_prior(pop_parameters=pop_params, prior_path=prior_path)
        generate_population(pop_name, pop_pri, outdir, prior_path)
    main_plotter(
        pop_a_file=f"{outdir}/{list(POPS.keys())[0]}_highsnr.dat",
        pop_b_file=f"{outdir}/{list(POPS.keys())[1]}_highsnr.dat",
        full_pop_a_file=f"{outdir}/{list(POPS.keys())[0]}.dat",
        full_pop_b_file=f"{outdir}/{list(POPS.keys())[1]}.dat",
        outdir=outdir,
        pop_a_vals=list(list(POPS.values())[0].values()),
        pop_b_vals=list(list(POPS.values())[1].values()),
    )

def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--prior-file",type=str)
    args = parser.parse_args()
    return args

def main():
    args = create_parser_and_read_args()
    main_generator(prior_path=args.prior_file, outdir=args.outdir)