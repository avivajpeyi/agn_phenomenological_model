import numpy as np
import pandas as pd
import tqdm
from agn_utils.batch_processing import run_function_with_multiprocess
from agn_utils.bbh_population_generators import (
    get_bbh_population_from_agn_params,
)

from .agn_logger import logger

DATA_KEY = "training_data"


def save_agn_samples_for_population_instance(fname, sigma_1, sigma_12):
    """

    :param fname:
    :type fname:
    :param sigma_1:
    :type sigma_1:
    :param sigma_12:
    :type sigma_12:
    :return:
    :rtype:
    """
    s = get_bbh_population_from_agn_params(
        num_samples=5000, sigma_1=sigma_1, sigma_12=sigma_12
    )

    df = pd.DataFrame(
        dict(
            sigma_1=sigma_1,
            sigma_12=sigma_12,
            chi_p=s["chi_p"],
            chi_eff=s["chi_eff"],
            cos_tilt_1=s["cos_tilt_1"],
            cos_theta_12=s["cos_theta_12"],
            mass_ratio=s["mass_ratio"],
            p_mass_ratio=s["p_mass_ratio"],
            p_cos_tilt_1=s["p_cos_tilt_1"],
            p_cos_theta_12=s["p_cos_theta_12"],
        )
    )
    store = pd.HDFStore(fname)
    store.append(key=DATA_KEY, value=df, format="t", data_columns=True)
    store.close()


def save_agn_samples_for_many_populations(num, fname, num_processors=1):
    logger.info(
        f"Generating {num ** 2} AGN populations and their distributions"
    )
    vals = np.linspace(0.05, 4, num=num)
    sig1, sig12 = vals, vals
    kwargs = []
    for i_sig1 in sig1:
        for i_sig12 in sig12:
            kwargs.append(dict(fname=fname, sigma_1=i_sig1, sigma_12=i_sig12))

    kwargs_lists = np.array_split(kwargs, len(kwargs) / num_processors)

    for k in tqdm.tqdm(kwargs_lists, desc="Processing populations"):
        run_function_with_multiprocess(
            num_multiprocesses=num_processors,
            target_function=save_agn_samples_for_population_instance,
            kwargs=k,
        )


def load_training_data(fname) -> pd.DataFrame:
    df = pd.read_hdf(fname, key=DATA_KEY)
    return df
