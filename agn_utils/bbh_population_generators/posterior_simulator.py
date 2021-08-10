import numpy as np
import pandas as pd
from bilby.core.prior import TruncatedNormal, PriorDict

from agn_utils.data_formetter import ld_to_dl

BOUNDS = dict(
    cos_theta_1=(-1, 1),
    cos_theta_12=(-1, 1),
)

PRIOR_VOLUME = (
        (BOUNDS["cos_theta_1"][1] - BOUNDS["cos_theta_1"][0])
        * (BOUNDS["cos_theta_12"][1] - BOUNDS["cos_theta_12"][0])
)


def simulate_posterior(sample, fractional_sigma=0.1, n_samples=1000):
    posterior = pd.DataFrame()
    for key in sample:
        if key in BOUNDS:
            bound = BOUNDS[key]
        else:
            bound = (-np.inf, np.inf)
        sigma = sample[key] * fractional_sigma
        new_true = TruncatedNormal(
            mu=sample[key], sigma=sigma, minimum=bound[0], maximum=bound[1]
        ).sample()
        posterior[key] = TruncatedNormal(
            mu=new_true, sigma=sigma, minimum=bound[0], maximum=bound[1]
        ).sample(n_samples)

    posterior["prior"] = 1 / PRIOR_VOLUME
    return posterior


def simulate_population_posteriors(sig1=5, sig12=5, number_events=10, n_samp=50000, fractional_sigma=1):
    pop_prior = PriorDict(dict(
        cos_theta_1=TruncatedNormal(mu=1, sigma=sig1, minimum=-1, maximum=1),
        cos_theta_12=TruncatedNormal(mu=1, sigma=sig12, minimum=-1, maximum=1)
    ))
    params = pop_prior.keys()
    posteriors = {p: [] for p in params}
    trues = {p: [] for p in params}
    for i in range(number_events):
        true = pop_prior.sample()
        posterior = simulate_posterior(true, n_samples=n_samp, fractional_sigma=1)
        for p in params:
            posteriors[p].append(posterior[p].values)
            trues[p].append(true[p])

    for p in params:
        posteriors[p] = np.array(posteriors[p])
        trues[p] = np.array(trues[p])

    return dict(
        trues=trues,
        posteriors=posteriors
    )


def simulate_exact_population_posteriors(sig1=5, sig12=5, number_events=10, n_samp=10000):
    pop_prior = PriorDict(dict(
        cos_tilt_1=TruncatedNormal(mu=1, sigma=sig1, minimum=-1, maximum=1),
        cos_theta_12=TruncatedNormal(mu=1, sigma=sig12, minimum=-1, maximum=1)
    ))
    posteriors = [pop_prior.sample(n_samp) for _ in range(number_events)]
    posteriors = ld_to_dl(posteriors)
    posteriors = {k:np.array(v) for k,v in posteriors.items()}

    return dict(
        trues=[],
        posteriors=posteriors
    )


if __name__ == '__main__':
    num_events = 5
    dat = simulate_population_posteriors(sig1=0.5, sig12=3, number_events=num_events)

    pass
