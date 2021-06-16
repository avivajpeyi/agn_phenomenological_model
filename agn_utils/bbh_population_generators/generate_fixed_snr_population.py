import logging
import os
import shutil
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import (
    add_snr, add_signal_duration
)
from agn_utils.regressors import TfRegressor

FRESH_BUILD = False


def get_snrs_for_lumins(lumins, constant_params):
    params = {p: [v] * len(lumins) for p, v in constant_params.items()}
    params['luminosity_distance'] = lumins
    s = add_snr(pd.DataFrame(params))
    return list(s.network_snr.values)


def get_snrs_for_masses_and_lumins(masses, lumins, constant_params):
    cached_l, cached_m, cached_snrs = [], [], []
    for m in tqdm(masses, desc="Getting SNRS for diffent mass+dl"):
        constant_params['mass_1'], constant_params['mass_2'] = m, m
        snrs_for_current_mass = get_snrs_for_lumins(lumins, constant_params=constant_params)
        cached_snrs.extend(snrs_for_current_mass)
        cached_m.extend([m] * len(lumins))
        cached_l.extend(list(lumins))
    data = dict(lumin=cached_l, mass=cached_m, snr=cached_snrs)
    return pd.DataFrame(data)


def get_lumin_from_snrs_and_masses(snrs, masses, constant_params, model_path):
    logging.info("Using regression model to get Lumin for desired SNR")
    snr_and_mass_to_lumin_func = get_lumin_from_snr_and_mass_regressor(constant_params, model_path)
    best_guess_lumin = snr_and_mass_to_lumin_func(pd.DataFrame(dict(snr=snrs, mass=masses)))
    return best_guess_lumin


def get_lumin_from_snr_and_mass_regressor(constant_params, model_path) -> Callable:
    if FRESH_BUILD:
        shutil.rmtree(model_path)

    regressor_kwargs = dict(
        input_parameters=["mass", "snr"],
        output_parameters=["lumin"],
        outdir=model_path,
    )
    r = TfRegressor(**regressor_kwargs)

    if os.path.exists(r.savepath):
        r.load()
    else:
        masses = np.arange(15, 38, step=1)
        lumins = np.geomspace(100, 5000, num=250)
        training_data = get_snrs_for_masses_and_lumins(masses, lumins, constant_params)

        r.train(data=training_data)
        r.test(
            data=training_data[regressor_kwargs["input_parameters"]],
            labels=training_data[regressor_kwargs["output_parameters"]],
        )
        r.save()
    return r.predict


def main():
    masses = np.arange(16, 38, step=2)
    required_snr = 60

    constant_params = dict(
        dec=0, ra=0,  # event on the horizon
        theta_jn=np.pi / 2,  # Zenith angle between the total angular momentum and the line of sight
        psi=0,  # polarisation angle
        phase=0,
        geocent_time=0.0,
        a_1=0.7,
        a_2=0.7,
        tilt_1=np.arccos(0.7),
        tilt_2=np.arccos(0.7),
        phi_12=np.pi / 2,
        phi_jl=0.2
    )

    lumin = get_lumin_from_snrs_and_masses(
        snrs=[required_snr] * len(masses), masses=masses,
        constant_params=constant_params,
        model_path="lumin_from_snr_regressor"
    )
    params = {p: [v] * len(masses) for p, v in constant_params.items()}
    params['luminosity_distance'] = lumin
    params['mass_1'], params['mass_2'] = masses, masses
    bbh_df = pd.DataFrame(params)
    bbh_df = add_snr(bbh_df)
    bbh_df['snr'] = bbh_df['network_snr']
    bbh_df = bbh_df.drop(['h1_snr', 'l1_snr', 'network_snr'], axis=1)
    bbh_df = add_signal_duration(bbh_df)
    bbh_df.to_csv("same_snr_injections.csv")


if __name__ == '__main__':
    main()
