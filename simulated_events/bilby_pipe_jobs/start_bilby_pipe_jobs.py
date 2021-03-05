# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from bilby.gw.conversion import component_masses_to_chirp_mass
from bilby_pipe.gracedb import determine_duration_and_scale_factor_from_parameters


def load_injections(dat):
    return pd.read_csv(dat, sep=" ")


def calculate_durations(df):
    df['chirp_mass'] = component_masses_to_chirp_mass(df.mass_1, df.mass_2)
    duration, roq_scale_factor = np.vectorize(
        determine_duration_and_scale_factor_from_parameters)(
        chirp_mass=df['chirp_mass'])
    df['duration'] = duration
    long_signals = [f"data{i}" for i in range(len(duration)) if duration[i] > 4]
    print(f"long_signals= " + str(long_signals).replace("'", ""))
    return df


def run_bilby_pipe_jobs(dat):
    df = load_injections(dat)
    df = calculate_durations(df)
    for i in range(len(df)):
        duration = df.iloc[i]['duration']
        print(
            f"bilby_pipe injection_study_bilby_pipe.ini"
            f"--duration {duration} "
            f"--generation-seed {i} "
            f"--injection-numbers [{i}] "
            f"--label data{i} "
            f"--outdir outdir_data{i} "
        )


if __name__ == "__main__":
    run_bilby_pipe_jobs("injection_samples_all_param.dat")
