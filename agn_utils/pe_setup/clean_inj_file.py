import pandas as pd
import sys

REQUIRED_PARAM = [
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

def edit_injection_file(filename):
    df = pd.read_csv(filename, sep=" ")
    df = df[REQUIRED_PARAM]
    df.to_csv(filename.replace(".dat", "_clean.dat"), sep=" ", index=False)


def main():
    print('Argument List:', str(sys.argv))
    edit_injection_file(sys.argv[1])
