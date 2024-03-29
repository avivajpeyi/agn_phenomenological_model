"""
See
https://git.ligo.org/RatesAndPopulations/gwpopulation_pipe/-/blob/master/gwpopulation_pipe/common_format.py#L136
"""
from agn_utils.data_formetter import dl_to_ld, ld_to_dl

import matplotlib.pyplot as plt

import numpy as np
from bilby.hyper.model import Model
from scipy.special import erf, gammaln
from typing import  Optional, Dict, List


def truncnorm(xx, mu, sigma, high, low):
    if sigma <= 0:
        raise ValueError(f"Sigma must be greater than 0, sigma={sigma}")
    norm = 2 ** 0.5 / np.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = np.exp(-np.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

def agn_spin(dataset, sigma_1, sigma_12):
    """cos_theta_12: angle bw BH1 and BH2"""
    prior = truncnorm(xx=dataset["cos_tilt_1"], mu=1, sigma=sigma_1, high=1, low=-1) \
            * truncnorm(xx=dataset["cos_theta_12"], mu=1, sigma=sigma_12, high=1, low=-1)
    return prior


from tqdm import tqdm
import pandas as pd
import corner

from agn_utils.bbh_population_generators.posterior_simulator import simulate_posterior

BOUNDS = dict(
    cos_theta_1=(-1, 1),
    cos_theta_12=(-1, 1),
)

PRIOR_VOLUME = (
        (BOUNDS["cos_theta_1"][1] - BOUNDS["cos_theta_1"][0])
        * (BOUNDS["cos_theta_12"][1] - BOUNDS["cos_theta_12"][0])
)


def rejection_sample_posterior(event_samples:Dict[str, List], hyper_param:dict, n_draws=2000)->pd.DataFrame:
    # event_samples['cos_tilt_1'] = event_samples['cos_theta_1']
    model = Model(model_functions=[agn_spin])
    model.parameters.update(hyper_param)
    weights = np.array(model.prob(event_samples) / PRIOR_VOLUME)
    weights = weights / np.linalg.norm(weights)
    event_samples['weights'] = weights
    event_samples = pd.DataFrame(event_samples)
    event_samples = event_samples.sample(n_draws, weights=weights)
    event_samples = event_samples.to_dict('list')
    return event_samples


def rejection_sample_population(posteriors, true_population_param):
    """
    :param posteriors: dict(label:posteior list for each event)
    :param true_population_param: dict of hyper_param
    """
    posteriors_ld = dl_to_ld(posteriors)
    posteriors_ld = [rejection_sample_posterior(p, true_population_param) for p in posteriors_ld]
    posteriors = ld_to_dl(posteriors_ld)
    posteriors = {k:np.array(v) for k,v in posteriors.items()}
    return posteriors


def plot_weights(samples, weights):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(samples['cos_theta_1'], weights)
    axes[1].scatter(samples['cos_theta_12'], weights)
    plt.savefig("weights.png")




if __name__ == '__main__':
    event_samples = simulate_posterior(dict(cos_theta_1=-0.8, cos_theta_12=-0.2), n_samples=10000)
    p = rejection_sample_posterior(event_samples, hyper_param=dict(sigma_1=0.5, sigma_12=3))
    fig, ax = plt.subplots()
    ax.hist(event_samples['cos_theta_1'], density=True, histtype='step')
    ax.hist(p['cos_theta_1'], histtype='step', density=True)
    plt.savefig('test.png')
