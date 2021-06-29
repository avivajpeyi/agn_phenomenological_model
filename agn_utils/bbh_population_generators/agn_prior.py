import logging

import bilby
import numpy as np
import pandas as pd
from bilby.core.prior import Cosine, Interped, Uniform
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import iid_spin_magnitude_beta, truncnorm

from .calculate_extra_bbh_parameters import (
    get_chi_eff,
    get_chi_p,
    get_component_mass_from_source_mass_and_z,
    get_t2_from_t1_and_t12_spins,
    make_spin_vector,
    scale_vector,
    transform_component_spins,
)

logging.getLogger("bilby").setLevel(logging.ERROR)

DEFAULT_POPULATION_PARAMS = {
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
    "amax": 1.0,
    "lamb": 0.0,
}

A_1 = "a_1"
A_2 = "a_2"
CHI_EFF = "chi_eff"
CHI_P = "chi_p"
PHI_1 = "phi_1"
PHI_2 = "phi_2"
PHI_12 = "phi_12"
MASS_1 = "mass_1"
MASS_2 = "mass_2"
LUMINOSITY_DISTANCE = "luminosity_distance"
TILT_1 = "tilt_1"
TILT_2 = "tilt_2"
COS_TILT_1 = "cos_tilt_1"
COS_TILT_2 = "cos_tilt_2"
COS_THETA_12 = "cos_theta_12"
THETA_JN = "theta_jn"
PHI_JL = "phi_jl"
INCL = "incl"
PHASE = "phase"
Q = "mass_ratio"
Z = "redshift"
SOURCE_M1 = "mass_1_source"


def generate_population_prior(population_params):
    params = DEFAULT_POPULATION_PARAMS.copy()
    params.update(population_params)
    p, _ = convert_to_beta_parameters(params)

    # make grid_x-vals
    num_x = 10000
    mass = np.linspace(5, 100, num=num_x)
    q = np.linspace(0, 1, num=num_x)
    cos_vals = np.linspace(-1, 1, num=num_x)
    a = np.linspace(0, 1, num=num_x)
    z = np.linspace(0, 2.3, num=num_x)

    # calcualte probabilites
    mass_model = SinglePeakSmoothedMassDistribution()
    p_mass = mass_model.p_m1(
        dataset=pd.DataFrame(dict(mass_1=mass)),
        alpha=p["alpha"],
        mmin=p["mmin"],
        mmax=p["mmax"],
        lam=p["lam"],
        mpp=p["mpp"],
        sigpp=p["sigpp"],
        delta_m=p["delta_m"],
    )
    p_q = mass_model.p_q(
        dataset=pd.DataFrame(dict(mass_ratio=q, mass_1=mass)),
        beta=p["beta"],
        mmin=p["mmin"],
        delta_m=p["delta_m"],
    )
    p_costheta12 = truncnorm(
        xx=cos_vals, mu=1, sigma=p["sigma_12"], high=1, low=-1
    )
    p_costilt1 = truncnorm(
        xx=cos_vals, mu=1, sigma=p["sigma_1"], high=1, low=-1
    )
    p_a = iid_spin_magnitude_beta(
        dataset=pd.DataFrame(dict(a_1=a, a_2=a)),
        amax=p["amax"],
        alpha_chi=p["alpha_chi"],
        beta_chi=p["beta_chi"],
    )
    p_z = PowerLawRedshift(z_max=2.3).probability(
        dataset=pd.DataFrame(dict(redshift=z)), lamb=p["lamb"]
    )

    # after generating prior, generate samples, then convert the samples to BBH params
    priors = bilby.prior.PriorDict(
        dict(
            a_1=Interped(
                a, p_a, minimum=0, maximum=1, name="a_1", latex_label="$a_1$"
            ),
            a_2=Interped(
                a, p_a, minimum=0, maximum=1, name="a_2", latex_label="$a_2$"
            ),
            redshift=Interped(
                z,
                p_z,
                minimum=0,
                maximum=2.3,
                name="redshift",
                latex_label="$pred_z$",
            ),
            cos_tilt_1=Interped(
                cos_vals,
                p_costilt1,
                minimum=-1,
                maximum=1,
                name="cos_tilt_1",
                latex_label="$\\cos\ \\mathrm{tilt}_1$",
            ),
            cos_theta_12=Interped(
                cos_vals,
                p_costheta12,
                minimum=-1,
                maximum=1,
                name="cos_theta_12",
                latex_label="$\\cos\ \\theta_{12}$",
            ),
            mass_1_source=Interped(
                mass,
                p_mass,
                minimum=5,
                maximum=100,
                name="mass_1_source",
                latex_label="$m_{1}$",
            ),
            mass_ratio=Interped(
                q,
                p_q,
                minimum=0,
                maximum=1,
                name="mass_ratio",
                latex_label="$q$",
            ),
            dec=Cosine(name="dec"),
            ra=Uniform(
                name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            psi=Uniform(
                name="psi", minimum=0, maximum=np.pi, boundary="periodic"
            ),
            phi_1=Uniform(
                name="phi_1",
                minimum=0,
                maximum=2 * np.pi,
                boundary="periodic",
                latex_label="$\\phi_1$",
            ),
            phi_12=Uniform(
                name="phi_12",
                minimum=0,
                maximum=2 * np.pi,
                boundary="periodic",
                latex_label="$\\phi_{12}$",
            ),
            phase=Uniform(
                name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            incl=Uniform(
                name="incl", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            geocent_time=Uniform(
                minimum=-0.1,
                maximum=0.1,
                name="geocent_time",
                latex_label="$t_c$",
                unit="$s$",
            ),
        )
    )
    return priors


def get_bbh_population_from_agn_prior(num_samples=10000, population_params={}):
    pop_model = generate_population_prior(population_params)
    s = pd.DataFrame(pop_model.sample(num_samples)).to_dict("list")
    # Unpack posteriors_list
    phi_1, phi_12 = s[PHI_1], s[PHI_12]
    z, q = s[Z], s[Q]
    incl, phase = s[INCL], s[PHASE]
    a_1, a_2 = np.array(s[A_1]), np.array(s[A_2])
    tilt_1, theta_12 = np.arccos(s[COS_TILT_1]), np.arccos(s[COS_THETA_12])
    m1, m2 = get_component_mass_from_source_mass_and_z(s[SOURCE_M1], s[Q], z)
    tilt_2, phi_2 = get_t2_from_t1_and_t12_spins(
        θ12=theta_12, ϕ12=phi_12, θ1=tilt_1, ϕ1=phi_1
    )
    s1 = scale_vector(
        a_1, np.array(make_spin_vector(tilt_1, phi_1)).transpose()
    )
    s2 = scale_vector(
        a_2, np.array(make_spin_vector(tilt_2, phi_2)).transpose()
    )
    s1x, s1y, s1z = s1[:, 0], s1[:, 1], s1[:, 2]
    s2x, s2y, s2z = s2[:, 0], s2[:, 1], s2[:, 2]
    theta_jn, phi_jl, _, _, _, _, _ = transform_component_spins(
        incl=incl,
        S1x=s1x,
        S1y=s1y,
        S1z=s1z,
        S2x=s2x,
        S2y=s2y,
        S2z=s2z,
        m1=m1 * bilby.utils.solar_mass,
        m2=m2 * bilby.utils.solar_mass,
        phase=phase,
    )
    # pack posteriors_list
    s[CHI_EFF] = get_chi_eff(s1z=s1z, s2z=s2z, q=q)
    s[CHI_P] = get_chi_p(s1x=s1x, s1y=s1y, s2x=s2x, s2y=s2y, q=q)
    s[PHI_2] = phi_2
    s[MASS_1] = m1
    s[MASS_2] = m2
    s[Q] = q
    s[f"p_{Q}"] = pop_model[Q].prob(s[Q])
    s[
        LUMINOSITY_DISTANCE
    ] = bilby.gw.conversion.redshift_to_luminosity_distance(z)
    s[TILT_1] = tilt_1
    s[TILT_2] = tilt_2
    s[COS_TILT_1] = np.cos(tilt_1)
    s[COS_TILT_2] = np.cos(tilt_2)
    s[f"p_{COS_TILT_1}"] = pop_model[COS_TILT_1].prob(s[COS_TILT_1])
    s[COS_THETA_12] = np.cos(theta_12)
    s[f"p_{COS_THETA_12}"] = pop_model[COS_THETA_12].prob(s[COS_THETA_12])
    s[THETA_JN] = theta_jn
    s[PHI_JL] = phi_jl
    return pd.DataFrame(s)
