# -*- coding: utf-8 -*-
import glob
import os
import pandas as pd

import logging
logging.getLogger().setLevel(logging.INFO)


SAMPLES = 5000


def main():
    files = glob.glob("../data/gwtc1_samples/*.dat")
    for samples_filename in files:
        logging.info(f"PROCESSING {samples_filename}")
        samples = pd.read_csv(samples_filename, sep=" ").sample(SAMPLES)
        samples.to_csv(f"../data/downsampled_gwtc1_samples/{os.path.basename(samples_filename)}", index=False, sep=" ")


if __name__ == "__main__":
    main()
