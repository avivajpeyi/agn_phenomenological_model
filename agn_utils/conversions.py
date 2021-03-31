"""
Module to help convert parameters to our AGN formalism
"""
import numpy as np
from bbh_simulator.calculate_kick_vel_from_samples import Samples

from bilby.gw.conversion import component_masses_to_chirp_mass
from bilby_pipe.gracedb import determine_duration_and_scale_factor_from_parameters


def add_agn_samples_to_df(df):
    df['s1x'], df['s1y'], df['s1z'] = df['spin_1x'], df['spin_1y'], df['spin_1z']
    df['s2x'], df['s2y'], df['s2z'] = df['spin_2x'], df['spin_2y'], df['spin_2z']
    df['s1_dot_s2'] = \
        (df['s1x'] * df['s2x']) + \
        (df['s1y'] * df['s2y']) + \
        (df['s1z'] * df['s2z'])
    df['s1_mag'] = np.sqrt(
        (df['s1x'] * df['s1x']) +
        (df['s1y'] * df['s1y']) +
        (df['s1z'] * df['s1z']))
    df['s2_mag'] = np.sqrt(
        (df['s2x'] * df['s2x']) +
        (df['s2y'] * df['s2y']) +
        (df['s2z'] * df['s2z']))
    df['cos_theta_12'] = df['s1_dot_s2'] / (df['s1_mag'] * df['s2_mag'])
    return df


def add_kick(samples_filename):
    samples = Samples.from_file(samples_filename)
    samples.save_samples_with_kicks()  # sample file with `_kicks.dat`


def add_signal_durations(df):
    df['chirp_mass'] = component_masses_to_chirp_mass(df.mass_1, df.mass_2)
    duration, roq_scale_factor = np.vectorize(
        determine_duration_and_scale_factor_from_parameters)(
        chirp_mass=df['chirp_mass'])
    df['duration'] = duration
    long_signals = [f"data{i}" for i in range(len(duration)) if duration[i] > 4]
    print(f"long_signals= " + str(long_signals).replace("'", ""))
    return df
