# -*- coding: utf-8 -*-
import glob

from bbh_simulator.calculate_kick_vel_from_samples import Samples

import logging
logging.getLogger("bbh_simulator").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

SAMPLES = 5000


def main():
    files = glob.glob("../data/downsampled_gwtc1_samples/*.dat")
    for samples_filename in files:
        logging.info(f"PROCESSING {samples_filename}")
        samples = Samples(filename=samples_filename)
        samples.save_samples_with_kicks()


if __name__ == "__main__":
    main()
