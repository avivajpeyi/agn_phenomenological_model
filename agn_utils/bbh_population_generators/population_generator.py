import bilby
import pandas as pd

from .agn_prior import get_bbh_population_from_agn_prior
from .calculate_extra_bbh_parameters import (add_cos_theta_12_from_component_spins,
                                             add_signal_duration)
from ..agn_logger import logger

def get_bbh_population_from_prior(num_samples, prior_fname) -> pd.DataFrame:
    prior = bilby.gw.prior.PriorDict(filename=prior_fname)
    samples = prior.sample(num_samples)
    samples = bilby.gw.conversion.generate_all_bbh_parameters(samples)
    samples = add_cos_theta_12_from_component_spins(samples)
    samples = add_signal_duration(samples)
    return pd.DataFrame(samples)


def get_bbh_population_from_agn_params(num_samples, sigma_1, sigma_12) -> pd.DataFrame:
    pop_params = dict(sigma_1=sigma_1, sigma_12=sigma_12)
    logger.info(f"Drawing samples with {pop_params}")
    samples = get_bbh_population_from_agn_prior(
        num_samples=num_samples,
        population_params=pop_params
    )
    return samples
