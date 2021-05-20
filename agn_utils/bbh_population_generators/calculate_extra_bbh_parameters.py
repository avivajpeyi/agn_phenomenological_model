"""
Module to help convert parameters to our AGN formalism
"""
import bilby
import lalsimulation
import numpy as np

from bilby.gw.conversion import component_masses_to_chirp_mass
from bilby_pipe.gracedb import (
    determine_duration_and_scale_factor_from_parameters,
)
from numpy import cos, sin

REFERENCE_FREQ = 20


def add_cos_theta_12_from_component_spins(df):
    df["s1x"], df["s1y"], df["s1z"] = (
        df["spin_1x"],
        df["spin_1y"],
        df["spin_1z"],
    )
    df["s2x"], df["s2y"], df["s2z"] = (
        df["spin_2x"],
        df["spin_2y"],
        df["spin_2z"],
    )
    df["s1_dot_s2"] = (
        (df["s1x"] * df["s2x"])
        + (df["s1y"] * df["s2y"])
        + (df["s1z"] * df["s2z"])
    )
    df["s1_mag"] = np.sqrt(
        (df["s1x"] * df["s1x"])
        + (df["s1y"] * df["s1y"])
        + (df["s1z"] * df["s1z"])
    )
    df["s2_mag"] = np.sqrt(
        (df["s2x"] * df["s2x"])
        + (df["s2y"] * df["s2y"])
        + (df["s2z"] * df["s2z"])
    )
    df["cos_theta_12"] = df["s1_dot_s2"] / (df["s1_mag"] * df["s2_mag"])
    return df


def add_kick(df):
    from bbh_simulator.calculate_kick_vel_from_samples import Samples

    samples = Samples(posterior=df)
    samples.calculate_remnant_kick_velocity()
    return samples.posterior


def add_signal_duration(df):
    df["chirp_mass"] = component_masses_to_chirp_mass(df.mass_1, df.mass_2)
    duration, roq_scale_factor = np.vectorize(
        determine_duration_and_scale_factor_from_parameters
    )(chirp_mass=df["chirp_mass"])
    df["duration"] = duration
    long_signals = [
        f"data{i}" for i in range(len(duration)) if duration[i] > 4
    ]
    print(f"long_signals= " + str(long_signals).replace("'", ""))
    return df


def add_snr(df):
    assert [
        "a_1",
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
        "phi_jl",
    ] in df.columns.values
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
    incl,
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
            reference_frequency=20.0,
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


@np.vectorize
def get_t2_from_t1_and_t12_spins(θ12, ϕ12, θ1, ϕ1):
    """https://www.wolframcloud.com/obj/d954c5d7-c296-40c5-b776-740441099c40"""
    s2 = [
        -(cos(θ12) * sin(θ1)) + cos(θ1) * cos(ϕ1 - ϕ12) * sin(θ12),
        sin(θ12) * sin(ϕ1 - ϕ12),
        cos(θ1) * cos(θ12) + cos(ϕ1 - ϕ12) * sin(θ1) * sin(θ12),
    ]
    tilt_2 = np.arccos(s2[2])
    phi_2 = np.arctan2(s2[1], s2[0])
    if phi_2 < 0:
        phi_2 += 2.0 * np.pi

    return tilt_2, phi_2


def get_component_mass_from_source_mass_and_z(m1_source, q, z):
    m1 = m1_source * (1 + np.array(z))
    m2 = m1 * q
    return m1, m2


@np.vectorize
def transform_component_spins(
    incl=2, S1x=0, S1y=0, S1z=1, S2x=0, S2y=0, S2z=1, m1=20, m2=20, phase=0
):
    """https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html#ga6920c640f473e7125f9ddabc4398d60a"""
    (
        thetaJN,
        phiJL,
        theta1,
        theta2,
        phi12,
        chi1,
        chi2,
    ) = lalsimulation.SimInspiralTransformPrecessingWvf2PE(
        incl=incl,
        S1x=S1x,
        S1y=S1y,
        S1z=S1z,
        S2x=S2x,
        S2y=S2y,
        S2z=S2z,
        m1=m1,
        m2=m2,
        fRef=REFERENCE_FREQ,
        phiRef=phase,
    )
    return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2


@np.vectorize
def make_spin_vector(θ, ϕ):
    return (sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))


def scale_vector(scale, vector):
    return np.array([m * v for m, v in zip(scale, vector)])
