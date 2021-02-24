from pprint import pprint as print_p
import warnings

import bilby
import corner
import lalsimulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Cosine, Interped, Uniform
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import iid_spin_magnitude_beta, truncnorm
from numpy import cos, sin

warnings.filterwarnings("ignore")
REFERENCE_FREQ = 20
NUM = 200

SAMPLES = "samples.dat"

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


def plot_samples_corner():
    samples = pd.read_csv(SAMPLES, sep=" ")
    p = ["mass_1", "mass_2", "a_1", "cos_tilt_1", "cos_tilt_2","cos_theta_12", "phi_12", "phi_jl"]
    corner.corner(samples[p], **CORNER_KWARGS,
                  labels=["$m_1$", "$m_2$",
                      "$a_1$", "$\\cos \\mathrm{tilt}_1$","$\\cos \\mathrm{tilt}_2$","$\\cos \\theta_{12}$", "$\\phi_{12}$", "$\\phi_{JL}$"
                  ],
                  # range=[
                  #     (-1,1), (-1,1), (-1,1),
                  # ]
                  )
    plt.savefig("plots/samples.png")
    plt.close()
    # pprint.pprint(samples.to_dict('records')[0])
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
        dec=Cosine(name='dec'),
        ra=Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        psi=Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
        phi_1=Uniform(name="phi_1", minimum=0, maximum=2 * np.pi, boundary='periodic',
                      latex_label="$\\phi_1$"),
        phi_12=Uniform(name="phi_12", minimum=0, maximum=2 * np.pi, boundary='periodic',
                       latex_label="$\\phi_{12}$"),
        phase=Uniform(name="phase", minimum=0, maximum=2 * np.pi, boundary='periodic'),
        incl=Uniform(name="incl", minimum=0, maximum=2 * np.pi, boundary='periodic'),
        geocent_time=Uniform(minimum=-0.1, maximum=0.1, name="geocent_time",
                             latex_label="$t_c$", unit="$s$")
    ))
    return priors


def generate_s1_to_z_rotation_matrix(theta, phi):
    return np.array((
        (cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)),
        (-sin(phi), cos(phi), 0),
        (cos(phi)*sin(theta), sin(theta)*sin(phi), cos(theta))
    ))


@np.vectorize
def generate_agn_spins(theta_12, phi_12, theta_1, phi_1):
    """
    S1 = {sin(theta_1) cos(phi_1),sin(theta_1)*sin(phi_1), cos(theta_1)}
    S2* = {sin(theta_{12}) cos(phi_{12}),sin(theta_{12})*sin(phi_{12}), cos(theta_{12})}

    Here S1 is defined wrt z, and S2* is defined wrt S1_z.
    We need to rotate S2* to S2

    We can define the rotation matrix to rotate a vector from L to S1 with:
    R = R_z(phi_1) R_x(theta_1) ,

    and the inverse of this,
    R^{-1} = R_x^dagger(theta_1) R_z^dagger(phi_1) ,

    and now,
    S2 = R^{-1} S2*

    # TO GENERATE  ROTATION MATRIX:
    S[\[Theta]_, \[Phi]_]:= {Sin[\[Theta]] * Cos[\[Phi]], Sin[\[Theta]] * Sin[\[Phi]], Cos[\[Theta]]}
    Rx[\[Theta]_]:=RotationMatrix[\[Theta], {1, 0, 0}]
    Ry[\[Theta]_]:= RotationMatrix[\[Theta], {0, 1, 0}] ;
    Rz[\[Theta]_]:= RotationMatrix[\[Theta], {0, 0, 1}] ;
    Rot[\[Theta]_, \[Phi]_]:= Simplify[Rz[\[Phi]].Ry[\[Theta]]];
    Rinv [\[Theta]_, \[Phi]_]:=Simplify[ Inverse[Rot[\[Theta], \[Phi]]]];
    Rinv[\[Theta], \[Phi]] //MatrixForm

    # TEST
    Rinv.R == Unit Matrix
    Rinv = {
    {cos[phi],Sin[phi],0},
    {-cos[theta]sin[phi],cos[theta] cos[phi],Sin[theta]},
    {Sin[theta]sin[phi],-cos[phi]sin[theta],cos[theta]}
    }

    # Test 2
    Rinv.s1 = zhat

    """
    s1 = make_spin_vector(theta_1, phi_1)
    rinv1 = generate_s1_to_z_rotation_matrix(theta_1, phi_1)
    zhat = np.dot(rinv1, s1)

    s2_p = make_spin_vector(theta_12, phi_12)
    s2_p = np.array(s2_p)
    s2 = np.dot(rinv1, s2_p)

    tilt_2 = s2[2]
    phi_2 = np.arctan2(s2[1], s2[0])
    if phi_2 < 0:
        phi_2 += 2.0 * np.pi

    return tilt_2, phi_2



def get_mass_samples(m1_source, q, z):
    m1 = m1_source * (1 + np.array(z))
    m2 = m1 * q
    return m1, m2


@np.vectorize
def transform_component_spins(incl=2, S1x=0, S1y=0, S1z=1, S2x=0, S2y=0, S2z=1, m1=20, m2=20, phase=0):
    """https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html#ga6920c640f473e7125f9ddabc4398d60a"""
    thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2 = (
        lalsimulation.SimInspiralTransformPrecessingWvf2PE(
            incl=incl, S1x=S1x, S1y=S1y, S1z=S1z, S2x=S2x, S2y=S2y, S2z=S2z, m1=m1,
            m2=m2, fRef=REFERENCE_FREQ, phiRef=phase
        ))

    # print("INPUT")
    # pprint.pprint(dict(
    #     incl=incl, S1x=S1x, S1y=S1y, S1z=S1z, S2x=S2x, S2y=S2y, S2z=S2z, m1=m1,
    #         m2=m2, fRef=REFERENCE_FREQ, phiRef=phase,
    # ))
    # print("OUTPUT")
    # pprint.pprint(dict(
    #     thetaJN=thetaJN, phiJL=phiJL, theta1=theta1, theta2=theta2, phi12=phi12, chi1=chi1, chi2=chi2
    # ))
    return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2


@np.vectorize
def make_spin_vector(theta, phi):
    return (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))


def get_total_orb_angles(s1, s2, incl, m1, m2, phase, ):
    s1x, s1y, s1z = s1[:, 0], s1[:, 1], s1[:, 2]
    s2x, s2y, s2z = s2[:, 0], s2[:, 1], s2[:, 2]
    theta_jn, phi_jl, _, _, _, _, _ = (
        transform_component_spins(
            incl=incl, S1x=s1x, S1y=s1y, S1z=s1z, S2x=s2x, S2y=s2y, S2z=s2z,
            m1=m1 * bilby.utils.solar_mass, m2=m2 * bilby.utils.solar_mass,
            phase=phase
        ))

    return theta_jn, phi_jl


def mult_magn_to_vect(mags, vecs):
    return np.array([m * v for m, v in zip(mags, vecs)])

def calculate_theta_jn_and_phi_jl(m1, m2):
    lhat = [0, 0, 1]


def get_samples(num_samples=10000):
    pop_model = generate_prior(POP_MODEL_VALS)
    s = pd.DataFrame(pop_model.sample(num_samples)).to_dict('list')
    phi_1, z = s["phi_1"], s["redshift"]
    m1, m2 = get_mass_samples(s["mass_1_source"], s["mass_ratio"], z)
    tilt_1, theta_12 = np.arccos(s["cos_tilt_1"]), np.arccos(s["cos_theta_12"])
    tilt_2, phi_2 = generate_agn_spins(theta_12=theta_12, phi_12=s["phi_12"],
                                       theta_1=tilt_1, phi_1= phi_1)
    lumin_dist = bilby.gw.conversion.redshift_to_luminosity_distance(z)
    s1 = mult_magn_to_vect(np.array(s["a_1"]),
                           np.array(make_spin_vector(tilt_1, phi_1)).transpose())
    s2 = mult_magn_to_vect(np.array(s["a_2"]),
                           np.array(make_spin_vector(tilt_2, phi_2)).transpose())
    theta_jn, phi_jl = get_total_orb_angles(s1, s2, s['incl'], m1, m2, s['phase'])
    s['phi_2'] = phi_2
    s['mass_1'] = m1
    s['mass_2'] = m2
    s['luminosity_distance'] = lumin_dist
    s['tilt_1'] = tilt_1
    s['tilt_2'] = tilt_2
    s['cos_tilt_2'] = np.cos(tilt_2)
    s['theta_jn'] = theta_jn
    s['phi_jl'] = phi_jl
    return pd.DataFrame(s)


def generation_main():
    s = get_samples()
    s.to_csv(SAMPLES, sep=" ", index=False)
    plot_samples_corner()


def plot_theta_12_dist():
    samples = pd.read_csv(SAMPLES, sep=" ")
    prior = generate_prior(POP_MODEL_VALS)
    plt.hist(samples['cos_theta_12'], density=True, histtype="step", color="blue",
             label="Event Params")
    plt.plot(prior['cos_theta_12'].xx, prior['cos_theta_12'].yy, color="purple",
             label="Prior")
    plt.legend()
    plt.xlim(-1, 1)
    plt.ylabel("$p(\\cos \\theta_{12})$")
    plt.xlabel("$\\cos \\theta_{12})$")
    plt.savefig("plots/cos_theta_12.png")




def get_one_sample():
    s = get_samples(1)
    s = s.to_dict('records')[0]
    s['reference_frequency'] = REFERENCE_FREQ
    # print_p(s)
    params = ['theta_jn',
              'phi_jl',
              'tilt_1',
              'tilt_2',
              'phi_12',
              'a_1',
              'a_2',
              'mass_1',
              'mass_2',
              'reference_frequency',
              'phase']
    s_to_pass = {k:v for k,v in s.items() if k in params}
    returned_s = bilby.gw.conversion.transform_precessing_spins(**s_to_pass)
    returned_s = {k:float(v) for k,v in zip(params, returned_s)}
    component_spins = bilby.gw.conversion.generate_component_spins(s)
    # print_p(returned_s)


def save_multipl_samples():
    s = get_samples(200)
    s.to_csv(SAMPLES, sep=" ", index=False)
    plot_samples_corner()


if __name__ == '__main__':
    get_one_sample()
    save_multipl_samples()
    plot_theta_12_dist()
