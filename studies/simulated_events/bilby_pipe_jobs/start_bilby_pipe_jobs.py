# -*- coding: utf-8 -*-
import pandas as pd

from agn_utils import add_signal_duration


def load_injections(dat):
    return pd.read_csv(dat, sep=" ")


def run_bilby_pipe_jobs(dat):
    print("Loading injections")
    df = load_injections(dat)
    df = add_signal_duration(df)
    print("Loaded data")
    f = open("run_bilby_jobs.sh", "w")
    for i in range(len(df)):
        duration = df.iloc[i]['duration']
        command = (
            f"bilby_pipe injection_study_bilby_pipe.ini "
            f"--duration {duration} "
            f"--generation-seed {i} "
            f"--injection-numbers {i} "
            f"--label inj{i} \n"
        )
        print(command)
        f.write(command)


if __name__ == "__main__":
    run_bilby_pipe_jobs("injection_samples_all_params.dat")
