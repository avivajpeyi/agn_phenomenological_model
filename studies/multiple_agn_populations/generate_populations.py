import os
import warnings

import pandas as pd

from agn_utils.bbh_population_generators import get_bbh_population_from_agn_params
from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import add_snr, add_signal_duration

warnings.filterwarnings('ignore')

POPULATION_A = dict(sigma_1=0.5,
                    sigma_12=3)

POPULATION_B = dict(sigma_1=1,
                    sigma_12=0.25)

REQUIRED_PARAMS = [
    'dec',
    'ra',
    'theta_jn',
    'psi',
    'phase',
    'geocent_time',
    'a_1',
    'a_2',
    'tilt_1',
    'tilt_2',
    'phi_12',
    'phi_jl',
    'mass_1',
    'mass_2',
    'luminosity_distance',
    'duration'
]

POPS = dict(pop_a=POPULATION_A, pop_b=POPULATION_B)

if __name__ == "__main__":
    for pop_name, pop_params in POPS.items():
        fname = f"{pop_name}.dat"
        num_high_snr = 0
        iteration = 0
        if os.path.exists(fname):
            cached_pop = pd.read_csv(fname, sep=' ')
            num_high_snr = len(cached_pop[cached_pop['network_snr'] >= 60])

        while (num_high_snr < 40):
            pop_df = get_bbh_population_from_agn_params(
                num_samples=1000,
                **pop_params
            )
            pop_df = add_snr(pop_df)
            pop_df = add_signal_duration(pop_df)

            if os.path.exists(fname):
                cached_pop = pd.read_csv(fname, sep=' ')
                pop_df = pop_df.append(cached_pop, ignore_index=True)
            num_high_snr = len(pop_df[pop_df['network_snr'] >= 60])
            print(f"it-{iteration:02}: # high SNR events: {num_high_snr} in {len(pop_df):04} BBH")
            pop_df.to_csv(fname, sep=' ', mode='w', index=False)
            iteration += 1

        cached_pop = pd.read_csv(fname, sep=' ')
        high_snr_events = cached_pop[cached_pop['network_snr'] >= 60]
        high_snr_events = high_snr_events.loc[:, REQUIRED_PARAMS]
        high_snr_events.to_csv(fname.replace('.dat', '_highsnr.dat'), index=False, sep=' ')