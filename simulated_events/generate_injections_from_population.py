import logging
import multiprocessing
import os
import time
import warnings
from pprint import pprint as print_p

import bilby
import corner
import lalsimulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Cosine, Interped, Uniform
from bilby_pipe.gracedb import determine_duration_and_scale_factor_from_parameters
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import iid_spin_magnitude_beta, truncnorm
from numpy import cos, sin
from matplotlib import rcParams

rcParams["font.size"] = 20
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 20
rcParams["font.size"] = 30
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 10
rcParams['axes.linewidth'] = 2.5
rcParams['axes.edgecolor'] = 'black'
rcParams['xtick.labelsize'] = 25
rcParams['xtick.major.size'] = 10.0
rcParams['xtick.minor.size'] = 5.0
rcParams['ytick.labelsize'] = 25
rcParams['ytick.major.size'] = 10.0
rcParams['ytick.minor.size'] = 5.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

logging.getLogger("bilby").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.INFO)

warnings.filterwarnings("ignore")
REFERENCE_FREQ = 20
NUM = 200

SAMPLES = "samples.dat"
INJ_SAMPLES = "injection_samples.dat"

CORNER_KWARGS = dict(
    smooth=1,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color='tab:orange',
    quantiles=(0.16, 0.84),
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=5,
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

PARAMS = {
    # 'mass_1': '$m_1$',
    # 'mass_2': '$m_2$',
    # 'a_1': '$a_1$',
    'cos_tilt_1': '$\\cos \\mathrm{tilt}_1$',
    # 'cos_tilt_2': '$\\cos \\mathrm{tilt}_2$',
    'cos_theta_12': '$\\cos \\theta_{12}$',
    # 'phi_12': '$\\phi_{12}$',
    # 'phi_jl': '$\\phi_{JL}$',
    'luminosity_distance': '$d_L$',
    # 'network_snr': '$\\rho$'
    'log_snr': '$\\rm{log}_{10} \ \\rho$'
}

INJECTION_PARAMS = ["a_1",
                    "a_2",
                    "dec",
                    "ra",
                    "psi",
                    "phi_12",
                    "phase",
                    "incl",
                    "geocent_time",
                    "mass_1",
                    "mass_2",
                    "luminosity_distance",
                    "tilt_1",
                    "tilt_2",
                    "theta_jn",
                    "phi_jl"]


def timing(function):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) / 60.0
        duration_sec = (end_time - start_time) % 60.0
        f_name = function.__name__
        logging.info(f"{f_name} took {int(duration)}min {int(duration_sec)}s")

        return result

    return wrap


def plot_samples_corner(samples=[], samples_dat=None):
    if len(samples)==0:
        samples = pd.read_csv(samples_dat, sep=" ")
    # samples = samples[samples.network_snr > 8]
    if 'network_snr' in samples:
        samples['log_snr'] = np.log10(samples.network_snr)
    p = [i for i in PARAMS.keys()]
    l = [PARAMS[i] for i in p]
    corner.corner(samples[p], **CORNER_KWARGS, labels=l)
    fname = f"plots/{os.path.basename(samples_dat).replace('.dat', '.png')}"
    plt.savefig(fname)
    plt.close()
    logging.info(f"Plotted {fname}")


@np.vectorize
def get_injection_snr(
        a_1,
        a_2,
        dec,
        ra,
        psi,
        phi_12,
        phase,
        incl,
        geocent_time,
        mass_1,
        mass_2,
        luminosity_distance,
        tilt_1,
        tilt_2,
        theta_jn,
        phi_jl, **kwargs):
    """
    :returns H1 snr, L1 snr, network SNR
    """
    injection_parameters = dict(
        a_1=a_1,
        a_2=a_2,
        dec=dec,
        ra=ra,
        psi=psi,
        phi_12=phi_12,
        phase=phase,
        incl=incl,
        geocent_time=geocent_time,
        mass_1=mass_1,
        mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        theta_jn=theta_jn,
        phi_jl=phi_jl,
    )

    chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
    duration, _ = determine_duration_and_scale_factor_from_parameters(chirp_mass)
    sampling_frequency = 2048.

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant='IMRPhenomPv2',
            reference_frequency=20., minimum_frequency=20.
        )
    )

    # Set up interferometers.
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters['geocent_time'] - 2
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters
    )

    snrs = [ifo.meta_data["optimal_SNR"] for ifo in ifos]
    network_snr = np.sqrt(np.sum([i ** 2 for i in snrs]))
    return snrs[0], snrs[1], network_snr


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
        (cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)),
        (-sin(phi), cos(phi), 0),
        (cos(phi) * sin(theta), sin(theta) * sin(phi), cos(theta))
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
def transform_component_spins(incl=2, S1x=0, S1y=0, S1z=1, S2x=0, S2y=0, S2z=1, m1=20,
                              m2=20, phase=0):
    """https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html#ga6920c640f473e7125f9ddabc4398d60a"""
    thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2 = (
        lalsimulation.SimInspiralTransformPrecessingWvf2PE(
            incl=incl, S1x=S1x, S1y=S1y, S1z=S1z, S2x=S2x, S2y=S2y, S2z=S2z, m1=m1,
            m2=m2, fRef=REFERENCE_FREQ, phiRef=phase
        ))
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


@timing
def get_samples(num_samples=10000):
    poolname = multiprocessing.current_process().name
    logging.info(f"{poolname}: Drawing samples from Population Prior")
    pop_model = generate_prior(POP_MODEL_VALS)
    s = pd.DataFrame(pop_model.sample(num_samples)).to_dict('list')
    phi_1, z = s["phi_1"], s["redshift"]
    m1, m2 = get_mass_samples(s["mass_1_source"], s["mass_ratio"], z)
    tilt_1, theta_12 = np.arccos(s["cos_tilt_1"]), np.arccos(s["cos_theta_12"])
    tilt_2, phi_2 = generate_agn_spins(theta_12=theta_12, phi_12=s["phi_12"],
                                       theta_1=tilt_1, phi_1=phi_1)
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
    logging.info(f"{poolname}: Calculating SNR")
    h1_snr, l1_snr, network_snr = get_injection_snr(**s)
    s['h1_snr'] = h1_snr
    s['l1_snr'] = l1_snr
    s['network_snr'] = network_snr

    return pd.DataFrame(s)


def plot_theta_12_dist(sample_dat, label="100 BBH"):
    samples = pd.read_csv(sample_dat, sep=" ")
    prior = generate_prior(POP_MODEL_VALS)
    plt.hist(samples['cos_theta_12'], density=True,  color="tab:blue",
             label=label,  zorder=-1)
    plt.plot(prior['cos_theta_12'].xx, prior['cos_theta_12'].yy, color="tab:orange",
             label="Prior", zorder=1)
    plt.legend()
    plt.tight_layout()
    plt.xlim(-1, 1)
    plt.ylabel("$p(\\cos \\theta_{12})$")
    plt.xlabel("$\\cos \\theta_{12})$")
    plt.savefig(f"plots/{os.path.basename(sample_dat).replace('.dat','')}_cos_theta_12.png")
    plt.close('all')


def get_one_sample():
    s = get_samples(1)
    s = s.to_dict('records')[0]
    s['reference_frequency'] = REFERENCE_FREQ
    print_p(s)
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
    s_to_pass = {k: v for k, v in s.items() if k in params}
    returned_s = bilby.gw.conversion.transform_precessing_spins(**s_to_pass)
    returned_s = {k: float(v) for k, v in zip(params, returned_s)}
    component_spins = bilby.gw.conversion.generate_component_spins(s)
    print_p(returned_s)
    print_p(component_spins)


def save_multipl_samples():
    s = get_samples(10000)
    if os.path.isfile(SAMPLES):
        logging.info("Cached samples exist, appending...")
        s.to_csv(SAMPLES, sep=" ", header=False, index=False, mode='a')
    else:
        s.to_csv(SAMPLES, sep=" ", index=False, mode='w')

def plot_snrs(full_df, truncated_df, weights):
    ligo_events = full_df[full_df.network_snr > 8]
    agn_ligo_event = ligo_events[ligo_events.network_snr > 60]

    print(f"{len(agn_ligo_event)}/{len(ligo_events)} events > SNR 60")
    fig, axs = plt.subplots(2,1, sharex=False, figsize=(7, 9))
    axs[0].hist(full_df.network_snr, bins=50,color='tab:orange', label="All Inj", zorder=-1)
    axs[0].hist([], color='tab:green', label="Subset")
    axs[0].scatter([],[], color='tab:blue', label="Weights")
    axs[1].hist(truncated_df.network_snr, color='tab:green', label="Truncated", zorder=-1)
    ax2 = axs[1].twinx()
    ax2.grid(False)
    ax2.scatter(full_df.network_snr, weights, zorder=1, color="tab:blue")
    for i in range(len(axs)):
        axs[i].set_ylabel("Counts")
        axs[i].set_xlabel("Network SNR")
    axs[0].set_yscale("log")

    axs[0].legend(frameon=False, loc='upper right', bbox_to_anchor=(1.8, 1))
    fig.subplots_adjust(top=0.75)
    plt.savefig("plots/snr.png",  bbox_inches='tight')
    plt.close('all')
    logging.info("saved SNR plot")


def downsample_samples():
    logging.info("Making SNR cuts...")
    s = pd.read_csv(SAMPLES, sep=" ")
    plot_samples_corner(s, SAMPLES)
    # probability_distribution = bilby.prior.Uniform(minimum=60, maximum=200)
    probability_distribution = bilby.prior.Uniform(minimum=60, maximum=500)
    prob_for_snr = probability_distribution.prob(s.network_snr)
    injection_samples = s.sample(100, weights=prob_for_snr)
    plot_snrs(s, injection_samples, prob_for_snr)
    logging.info(f"{len(s)}-->{len(injection_samples)} samples")
    injection_samples.to_csv("injection_samples_all_params.dat", sep=" ", index=False)
    plot_samples_corner(injection_samples, "injection_samples_all_params.dat")
    injection_samples = injection_samples[INJECTION_PARAMS]
    injection_samples.to_csv(INJ_SAMPLES, sep=" ", index=False)


def get_samples_with_multiprocesses(num_multiprocesses):
    processes = []
    for i in range(1, num_multiprocesses):
        p = multiprocessing.Process(target=save_multipl_samples, name=f"Process{i}")
        processes.append(p)
        p.start()

    for process in processes:
        process.join()


if __name__ == '__main__':
    # for i in range(25):
    #     logging.info(f"Iteration {i}")
    #     get_samples_with_multiprocesses(6)
    # save_multipl_samples()
    downsample_samples()
    plot_theta_12_dist("injection_samples_all_params.dat")
    plot_theta_12_dist(SAMPLES, "All BBH")
