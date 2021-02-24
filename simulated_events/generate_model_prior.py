import warnings

import bilby
import numpy as np
import pandas as pd
from bilby.core.prior import Cosine, Interped, Uniform
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import iid_spin_magnitude_beta, truncnorm

warnings.filterwarnings("ignore")

NUM = 200

POP_MODEL_VALS = {
    "alpha": 2.62,
    "beta": 1.26,
    "mmax": 86.73,
    "mmin": 4.5,
    "lam": 0.12,
    "mpp": 33.5,
    "sigpp": 5.09,
    "delta_m": 4.88,
    "mu_chi": 0.25,
    "sigma_chi": 0.03,
    "sigma_1": 0.5,
    "sigma_12": 2,
    "amax": 1.0,
    "lamb": 0.0,
}


def generate_prior(p):
    p, _ = convert_to_beta_parameters(p)
    # make x-vals
    mass = np.linspace(5, 100, num=NUM)
    q = np.linspace(0, 1, num=NUM)
    cos_vals = np.linspace(-1, 1, num=NUM)
    a = np.linspace(0, 1, num=NUM)
    z = np.linspace(0, 2.3, num=NUM)

    # calcualte probabilites
    mass_model = SinglePeakSmoothedMassDistribution()
    p_mass = mass_model.p_m1(
        dataset=pd.DataFrame(dict(mass_1=mass)),
        alpha=p['alpha'], mmin=p['mmin'], mmax=p['mmax'], lam=p['lam'], mpp=p['mpp'],
        sigpp=p['sigpp'], delta_m=p['delta_m'])
    p_q = mass_model.p_q(
        dataset=pd.DataFrame(dict(mass_ratio=q, mass_1=mass)), beta=p["beta"],
        mmin=p["mmin"], delta_m=p["delta_m"])
    p_costheta12 = truncnorm(xx=cos_vals, mu=1, sigma=p['sigma_12'], high=1, low=-1)
    p_costilt1 = truncnorm(xx=cos_vals, mu=1, sigma=p['sigma_1'], high=1, low=-1)
    p_a = iid_spin_magnitude_beta(
        dataset=pd.DataFrame(dict(a_1=a, a_2=a)),
        amax=p['amax'], alpha_chi=p['alpha_chi'],
        beta_chi=p['beta_chi'])
    p_z = PowerLawRedshift(z_max=2.3).probability(
        dataset=pd.DataFrame(dict(redshift=z)), lamb=p['lamb'])

    # after generating prior, generate samples, then convert the samples to BBH params
    priors = bilby.prior.PriorDict(dict(
        a_1=Interped(a, p_a, minimum=0, maximum=1, name='a_1', latex_label="$a_1$"),
        a_2=Interped(a, p_a, minimum=0, maximum=1, name='a_2', latex_label="$a_2$"),
        redshift=Interped(z, p_z, minimum=0, maximum=2.3, name='redshift',
                          latex_label="$z$"),
        cos_tilt_1=Interped(cos_vals, p_costilt1, minimum=-1, maximum=1,
                            name='cos_tilt_1', latex_label="$\\cos\ \\mathrm{tilt}_1$"),
        cos_theta_12=Interped(cos_vals, p_costheta12, minimum=-1, maximum=1,
                              name='cos_theta_12', latex_label="$\\cos\ \\theta_{12}$"),
        mass_1_source=Interped(mass, p_mass, minimum=5, maximum=100,
                               name='mass_1_source', latex_label="$m_{1}$"),
        mass_ratio=Interped(q, p_q, minimum=0, maximum=1, name='mass_ratio',
                            latex_label="$q$"),
        # Detection
        dec=Cosine(name='dec'),
        ra=Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        psi=Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
        geocent_time=Uniform(minimum=-0.1, maximum=0.1, name="geocent_time",
                             latex_label="$t_c$", unit="$s$")
    ))
    return priors


def main():
    prior = generate_prior(POP_MODEL_VALS)
    label = "population_model_samples"
    pd.DataFrame(prior.sample(size=10000)).to_csv(f"{label}.csv", sep=" ")
    print(f"Saved model samples: {label}.csv")


if __name__ == "__main__":
    main()
