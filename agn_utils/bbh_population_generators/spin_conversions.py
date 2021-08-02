import lalsimulation
import numpy as np
import pandas as pd
from numpy import cross, eye
from numpy import sin, cos
from scipy.linalg import expm, norm

pd.reset_option('all')
pd.set_option("precision", 4)

MSUN = 1.9884099021470415e+30  # Kg
REFERENCE_FREQ = 20.0
PI = np.pi
PI2 = np.pi * 2.0
ANGLE_PARMS = ["phi_1", "phi_2", "phi_2", "tilt_1", "tilt_2", "theta_12"]


@np.vectorize
def agn_to_cartesian_coords(incl, phi_1, tilt_1, theta_12, phi_z_s12, a_1,
                            a_2, mass_1, mass_2, reference_frequency, phase, **kwargs):
    tilt_2, phi_2 = get_secondary_spins_from_primary_and_relative_spins(theta_12, phi_z_s12, tilt_1, phi_1)
    spin_1x, spin_1y, spin_1z = make_spin_vector(tilt=tilt_1, phi=phi_1)
    spin_2x, spin_2y, spin_2z = make_spin_vector(tilt=tilt_2, phi=phi_2)
    return incl, a_1 * spin_1x, a_1 * spin_1y, a_1 * spin_1z, a_2 * spin_2x, a_2 * spin_2y, a_2 * spin_2z


@np.vectorize
def spherical_to_cartesian_coords(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1,
                                  a_2, mass_1, mass_2, reference_frequency, phase, **kwargs):
    """
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html#ga6920c640f473e7125f9ddabc4398d60a
    """
    incl, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
        lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
            thetaJN=theta_jn, phiJL=phi_jl, theta1=tilt_1, theta2=tilt_2,
            phi12=phi_12, chi1=a_1, chi2=a_2,
            m1=mass_1 * MSUN, m2=mass_2 * MSUN,
            fRef=reference_frequency, phiRef=phase))
    return incl, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z


@np.vectorize
def cartesian_to_spherical_coords(incl, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
                                  spin_2z, mass_1, mass_2, phase, reference_frequency, **kwargs):
    """
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html#ga6920c640f473e7125f9ddabc4398d60a
    """
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = (
        lalsimulation.SimInspiralTransformPrecessingWvf2PE(
            incl=incl,
            S1x=spin_1x, S1y=spin_1y, S1z=spin_1z,
            S2x=spin_2x, S2y=spin_2y, S2z=spin_2z,
            m1=mass_1, m2=mass_2,
            fRef=reference_frequency, phiRef=phase
        ))
    _, phi_1, _, _, phi_2, _, phi_12, theta_12, phi_z_s12 = calculate_relative_spins_from_component_spins(
        spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z
    )
    return theta_jn, phi_jl, tilt_1, tilt_2, phi_1, phi_2, a_1, a_2, phi_12, theta_12, phi_z_s12


def angle(val):
    return np.fmod(PI2 + val, PI2)


def v_dot(spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z):
    return (
            spin_1x * spin_2x +
            spin_1y * spin_2y +
            spin_1z * spin_2z
    )


def angle_bw_vectors(spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z):
    s1_dot_s2 = v_dot(spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    s1_mag = np.sqrt(v_dot(spin_1x, spin_1y, spin_1z, spin_1x, spin_1y, spin_1z))
    s2_mag = np.sqrt(v_dot(spin_2x, spin_2y, spin_2z, spin_2x, spin_2y, spin_2z))
    return np.arccos(s1_dot_s2 / (s1_mag * s2_mag))


@np.vectorize
def calculate_relative_spins_from_component_spins(spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z):
    a_1 = np.sqrt(v_dot(spin_1x, spin_1y, spin_1z, spin_1x, spin_1y, spin_1z))
    a_2 = np.sqrt(v_dot(spin_2x, spin_2y, spin_2z, spin_2x, spin_2y, spin_2z))
    phi_1 = np.arctan2(spin_1y, spin_1x)
    phi_2 = np.arctan2(spin_2y, spin_2x)
    tilt_1 = np.arccos(spin_1z / a_1)
    tilt_2 = np.arccos(spin_2z / a_2)
    theta_12 = angle_bw_vectors(spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)
    phi_12 = phi_2 - phi_1
    if phi_12 < 0:
        phi_12 += np.pi * 2

    y_axis, z_axis = [0, 1, 0], [0, 0, 1]
    Ry = M(y_axis, tilt_1)  # to rotate z by tilt_1 --> down to xy axis
    Rz = M(z_axis, phi_1)  # to rotate z by phi_1 --> around to z axis
    R_z_to_s1 = np.dot(Rz, Ry)
    R_s1_to_z = R_z_to_s1.T
    s2 = make_spin_vector(tilt_2, phi_2)
    s12 = np.dot(R_s1_to_z, s2)
    phi_z_s12 = angle(np.arctan2(s12[1], s12[0]))

    return a_1, phi_1, tilt_1, a_2, phi_2, tilt_2, phi_12, theta_12, phi_z_s12


@np.vectorize
def make_spin_vector(tilt, phi):
    return (
        sin(tilt) * cos(phi),
        sin(tilt) * sin(phi),
        cos(tilt)
    )


def M(axis, theta):
    return expm(cross(eye(3), axis / norm(axis) * theta))



@np.vectorize
def get_secondary_spins_from_primary_and_relative_spins(theta_12, phi_z_s12, tilt_1, phi_1):
    y_axis, z_axis = [0, 1, 0], [0, 0, 1]
    s12 = make_spin_vector(theta_12, phi_z_s12)  # s1 points along zhat
    Ry = M(y_axis, tilt_1)  # to rotate s12 by tilt_1 --> down to s1xy axis
    Rz = M(z_axis, phi_1)  # to rotate s12 by phi_1 --> around to s1z axis
    RotM_s12_to_s2 = np.dot(Rz, Ry)
    s2 = np.dot(RotM_s12_to_s2, s12)

    # check1: see if s2 can be moved back to s12
    s12_check = np.dot(RotM_s12_to_s2.T, s2)
    for c, z in zip(s12_check, s12):
        np.testing.assert_almost_equal(c, z)

    tilt_2 = np.arccos(s2[2])
    phi_2 = np.arctan2(s2[1], s2[0])
    # if phi_2 < 0:
    #     phi_2 += 2.0 * np.pi

    # check2: see if s2 can be moved to zhat
    Ry = M(y_axis, tilt_2)
    Rz = M(z_axis, phi_2)
    RotM = np.dot(Rz, Ry).T
    check = np.dot(RotM, s2)
    for c, z in zip(check, z_axis):
        np.testing.assert_almost_equal(c, z)

    return tilt_2, phi_2

