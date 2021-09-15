"""
Module to help convert parameters to our AGN formalism
"""
import bilby
import lalsimulation
import numpy as np

from bilby.gw.conversion import (
    component_masses_to_chirp_mass, total_mass_and_mass_ratio_to_component_masses,
                                 chirp_mass_and_mass_ratio_to_total_mass, generate_all_bbh_parameters, generate_spin_parameters, generate_mass_parameters,
    convert_to_lal_binary_black_hole_parameters, generate_mass_parameters, generate_component_spins
)
from bilby_pipe.gracedb import (
    determine_duration_and_scale_factor_from_parameters,
)
from .spin_conversions import calculate_relative_spins_from_component_spins
from numpy import cos, sin
from ..pe_postprocessing.evolve_spins_back import get_tilts_at_inf

REFERENCE_FREQ = 20



def add_kick(df):
    from bbh_simulator.calculate_kick_vel_from_samples import Samples

    samples = Samples(posterior=df)
    samples.calculate_remnant_kick_velocity()
    return samples.posterior


def add_signal_duration(df):
    df["chirp_mass"] = component_masses_to_chirp_mass(df['mass_1'], df['mass_2'])
    duration, roq_scale_factor = np.vectorize(
        determine_duration_and_scale_factor_from_parameters
    )(chirp_mass=df["chirp_mass"])
    df["duration"] = duration
    long_signals = [
        f"data{i}" for i in range(len(duration)) if duration[i] > 4
    ]
    # print(f"long_signals= " + str(long_signals).replace("'", ""))
    return df


def add_snr(df):
    required_params = [
        "dec",
        "ra",
        "theta_jn",
        "geocent_time",
        "luminosity_distance",
        "psi",
        "phase",
        "mass_1",
        "mass_2",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
    ]
    df_cols = list(df.keys())
    missing_params = set(required_params) - set(df_cols)
    if len(missing_params) != 0:
        raise ValueError(f"Params missing for SNR calculation: {missing_params}")

    h1_snr, l1_snr, network_snr = _get_injection_snr(**df)
    df["h1_snr"] = h1_snr
    df["l1_snr"] = l1_snr
    df["network_snr"] = network_snr
    return df


@np.vectorize
def _get_injection_snr(
        a_1,
        a_2,
        dec,
        ra,
        psi,
        phi_12,
        phase,
        geocent_time,
        mass_1,
        mass_2,
        luminosity_distance,
        tilt_1,
        tilt_2,
        theta_jn,
        phi_jl,
        **kwargs,
):
    """
    :returns H1 snr, L1 snr, network SNR
    """
    injection_parameters = dict(
        # location params
        dec=dec,
        ra=ra,
        theta_jn=theta_jn,
        luminosity_distance=luminosity_distance,
        geocent_time=geocent_time,
        # phase params
        psi=psi,
        phase=phase,
        # mass params
        mass_1=mass_1,
        mass_2=mass_2,
        # spin params
        a_1=a_1,
        a_2=a_2,
        phi_12=phi_12,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_jl=phi_jl,
    )

    chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(
        mass_1, mass_2
    )
    duration, _ = determine_duration_and_scale_factor_from_parameters(
        chirp_mass
    )
    sampling_frequency = 2048.0

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=REFERENCE_FREQ,
            minimum_frequency=20.0,
        ),
    )

    # Set up interferometers.
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    snrs = [ifo.meta_data["optimal_SNR"] for ifo in ifos]
    network_snr = np.sqrt(np.sum([i ** 2 for i in snrs]))
    return snrs[0], snrs[1], network_snr


@np.vectorize
def get_chi_eff(s1z, s2z, q):
    return (s1z + s2z * q) / (1 + q)


@np.vectorize
def get_chi_p(s1x, s1y, s2x, s2y, q):
    chi1p = np.sqrt(s1x ** 2 + s1y ** 2)
    chi2p = np.sqrt(s2x ** 2 + s2y ** 2)
    qfactor = q * ((4 * q) + 3) / (4 + (3 * q))
    return np.maximum(chi1p, chi2p * qfactor)



def get_component_mass_from_source_mass_and_z(m1_source, q, z):
    m1 = m1_source * (1 + np.array(z))
    m2 = m1 * q
    return m1, m2


def get_component_mass_from_mchirp_q(mchirp, q):
    mtot = chirp_mass_and_mass_ratio_to_total_mass(chirp_mass=mchirp, mass_ratio=q)
    m1, m2 = total_mass_and_mass_ratio_to_component_masses(mass_ratio=q, total_mass=mtot)
    return m1, m2


def scale_vector(scale, vector):
    if len(scale.shape) > 0:
        if scale.shape[0] == vector.shape[0]:
            return  np.array([m * v for m, v in zip(scale, vector)])
    else:
        v = scale * vector
        v.shape = (3,1)
        return v.T


def add_cos_theta_12_from_component_spins(s):
    _, _, _, _, _, _, _, theta_12, _ = calculate_relative_spins_from_component_spins(s["spin_1x"], s["spin_1y"], s["spin_1z"], s["spin_2x"], s["spin_2y"], s["spin_2z"])
    s['cos_theta_12'] = np.cos(theta_12)
    return s


def process_samples(s, rf):
    s['reference_frequency'] = rf
    s, _ = convert_to_lal_binary_black_hole_parameters(s)
    s = generate_mass_parameters(s)
    s = generate_spin_parameters(s)
    s = add_cos_theta_12_from_component_spins(s)
    try:
        s = add_snr(s)
        s['snr'] = s['network_snr']
    except Exception as e:
        pass
    return s


def result_post_processing(r:bilby.result.Result):
    r.posterior = add_cos_theta_12_from_component_spins(r.posterior)
    r.posterior = get_tilts_at_inf(r.posterior, fref=r.reference_frequency)
    r.injection_parameters = process_samples(r.injection_parameters, r.reference_frequency)
    return r
