import lalsimulation
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
import bilby
import numpy as np
import pandas as pd
from bilby.core.prior import Cosine, Interped, Uniform
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import iid_spin_magnitude_beta, truncnorm

warnings.filterwarnings("ignore")
REFERENCE_FREQ = 20
NUM = 200

CORNER_KWARGS = dict(
    smooth=0.99,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color='tab:orange',
    quantiles=(0.16, 0.84),
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)


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

import corner


def plot_samples_corner():
    samples = pd.read_csv("samples.csv")
    p = ["a_1","a_2","redshift","cos_tilt_1","cos_tilt_2","cos_theta_12","mass_1","mass_2","mass_ratio"]

    corner.corner(samples[p], **CORNER_KWARGS, labels=[
        "$a_1$", "$a_2$", "z", "$\\cos \\mathrm{tilt}_1$", "$\\cos \\mathrm{tilt}_2$", "$\\cos \\theta_{12}$", "$m_1$",
        "$m_2$", "q"
    ])
    plt.savefig("plots/samples.png")
    print("done")


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
    ))

    plt.plot(cos_vals, priors['cos_theta_12'].prob(cos_vals))
    plt.savefig('costheta12.png')
    return priors





def get_isotropic_vectors(num_points=100):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0, 2 * np.pi, size=num_points)
    costheta = np.random.uniform(-1, 1, size=num_points)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z]).transpose()


def dot_product(vec_list1, vec_list2):
    return np.einsum("ij, ij->i", vec_list1, vec_list2)


def generate_agn_spins_from_component_spins(m1_list, m2_list):
    """"""
    vec_list1 = get_isotropic_vectors(num_points=len(m1_list))
    vec_list2 = get_isotropic_vectors(num_points=len(m1_list))
    v1_dot_v2 = dot_product(vec_list1, vec_list2)
    v1_mag = np.linalg.norm(vec_list1, axis=1)
    v2_mag = np.linalg.norm(vec_list2, axis=1)
    cos_theta_12 = v1_dot_v2 / (v1_mag * v2_mag)
    incl = np.random.uniform(0, 2 * np.pi, size=len(vec_list1))
    phase = np.random.uniform(0, 2 * np.pi)
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = transform_component_spins(
        incl=incl, m1=m1_list, m2=m2_list, phase=phase,
        S1x=vec_list1[:, 0], S1y=vec_list1[:, 1], S1z=vec_list1[:, 2],
        S2x=vec_list2[:, 0], S2y=vec_list2[:, 1], S2z=vec_list2[:, 2],
    )
    return cos_theta_12, theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase


@np.vectorize
def transform_component_spins(incl=2, S1x=0, S1y=0, S1z=1, S2x=0, S2y=0, S2z=1, m1=20,
                              m2=20, phase=0):
    """
    Conversions source XLALSimInspiralTransformPrecessingWvf2PE():
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html#ga6920c640f473e7125f9ddabc4398d60a
    -   thetaJN	zenith angle between J and N (rad) [return]
    -   phiJL	azimuthal angle of L_N on its cone about J (rad) [return]
    -   theta1	zenith angle between S1 and LNhat (rad) [return]
    -   theta2	zenith angle between S2 and LNhat (rad) [return]
    -   phi12	difference in azimuthal angle btwn S1, S2 (rad) [return]
    -   chi1	dimensionless spin of body 1
    -   chi2	dimensionless spin of body 2
    -   incl	Inclination angle of L_N (input)
    -   S1x	S1 x component (input)
    -   S1y	S1 y component (input)
    -   S1z	S1 z component (input)
    -   S2x	S2 x component (input)
    -   S2y	S2 y component (input)
    -   S2z	S2 z component (input)
    -   m1	mass of body 1 (solar mass)
    -   m2	mass of body 2 (solar mass)
    -   fRef	reference GW frequency (Hz)
    -   phiRef	reference orbital phase
    """
    thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2 = (
        lalsimulation.SimInspiralTransformPrecessingWvf2PE(
            incl=incl, S1x=S1x, S1y=S1y, S1z=S1z, S2x=S2x, S2y=S2y, S2z=S2z, m1=m1,
            m2=m2, fRef=REFERENCE_FREQ, phiRef=phase
        ))
    return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2


def get_mass_samples(m1_source, q, z):
    m1 = m1_source * (1 + z)
    m2 = m1 * q
    return m1, m2


import matplotlib.pyplot as plt

def get_samples_with_rejection_sampling():
    pop_model = generate_prior(POP_MODEL_VALS)
    samples = pd.DataFrame(pop_model.sample(10000))
    mass_1, mass_2 = get_mass_samples(
        samples.mass_1_source, samples.mass_ratio, samples.redshift)
    cos_theta_12, theta_jn, phi_jl, tilt_1, tilt_2, phi_12, _, _, phase = (
        generate_agn_spins_from_component_spins(mass_1, mass_2))
    lumin_dist = bilby.gw.conversion.redshift_to_luminosity_distance(samples.redshift)
    samples['mass_1'] = mass_1
    samples['mass_2'] = mass_2
    samples['luminosity_distance'] = lumin_dist
    samples['cos_theta_12'] = cos_theta_12
    samples['theta_jn'] = theta_jn
    samples['phi_jl'] = phi_jl
    samples['cos_tilt_1'] = np.cos(tilt_1)
    samples['cos_tilt_2'] = np.cos(tilt_2)
    samples['phi_12'] = phi_12
    print(samples[['cos_theta_12', "cos_tilt_1"]].describe())
    fig, axs = plt.subplots(2)
    axs[0].hist(samples['cos_tilt_1'], histtype='step', density=True, color='r')
    axs[0].hist(samples['cos_theta_12'], histtype='step',density=True, color='b')
    axs[0].set_xlim(-1,1)
    samples_subset = samples[[i.name for i in pop_model.values()]]
    prior_prob = [pop_model.prob(s) for s in tqdm(samples_subset.to_dict('records'), desc="Calc prior prob")]
    downsampled = samples.sample(n=1000, weights=prior_prob)
    print(downsampled[['cos_theta_12', "cos_tilt_1"]].describe())
    axs[1].hist(downsampled['cos_tilt_1'], histtype='step', color='r',density=True, label="after")
    axs[1].hist(downsampled['cos_theta_12'], histtype='step',density=True, color='b')
    axs[1].set_xlim(-1, 1)
    plt.savefig("compare.png")
    plt.close(fig)
    return downsampled


if __name__ == '__main__':
    s = get_samples_with_rejection_sampling()
    s.to_csv("samples.csv")