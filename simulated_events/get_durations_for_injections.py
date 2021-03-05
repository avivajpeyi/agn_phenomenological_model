# -*- coding: utf-8 -*-
import pandas as pd
from bilby.gw.conversion import component_masses_to_chirp_mass
from bilby_pipe.gracedb import determine_duration_and_scale_factor_from_parameters
import matplotlib.pyplot as plt
import numpy as np


def plot(df):
    """Plots a scatter plot."""
    fig, ax = plt.subplots()
    ax.hist(df.duration, density=True)
    fig.savefig(f"{len(df)}_durations.png")


def load_injections(dat):
    return pd.read_csv(dat, sep=" ")


def calculate_durations(df):
    df['chirp_mass'] = component_masses_to_chirp_mass(df.mass_1, df.mass_2)
    duration, roq_scale_factor = np.vectorize(determine_duration_and_scale_factor_from_parameters)(
        chirp_mass=df['chirp_mass'])
    df['duration'] = duration
    long_signals = [f"data{i}" for i in range(len(duration)) if duration[i] > 4]
    print(f"long_signals= " + str(long_signals).replace("'", ""))
    return df

def calcualte_durations_for_injections(dat):
    df = load_injections(dat)
    df = calculate_durations(df)
    plot(df)




def main():
    calcualte_durations_for_injections("samples.dat")
    calcualte_durations_for_injections("injection_samples_all_params.dat")


if __name__ == "__main__":
    main()
