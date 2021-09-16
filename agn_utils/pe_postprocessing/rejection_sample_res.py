from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import result_post_processing
from agn_utils.pe_postprocessing.evolve_spins_back import get_tilts_at_inf
from agn_utils.pe_postprocessing.posterior_reweighter import rejection_sample_posterior
import sys
import pandas as pd
import bilby

import argparse

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="path to res", type=str)
    parser.add_argument("sig1", help="sig1 tru val", type=float)
    parser.add_argument("sig12", help="sig12 tru val", type=float)
    args =  parser.parse_args()
    return args.result, dict(sigma_1=args.sig1, sigma_12=args.sig12)

def process_r(result_fn, hyper_params):
    print("Reading res")
    r = bilby.gw.result.CBCResult.from_json(filename=result_fn)
    r = result_post_processing(r)
    print("Rejection-sampling res")
    samples = pd.DataFrame(rejection_sample_posterior(r.posterior, hyper_param=hyper_params))
    print("Converting to spin at inf")
    samples = get_tilts_at_inf(samples, fref=r.reference_frequency)
    samples.to_hdf(result_fn.replace(".json", "_reweighted.h5"), "samples")
    print(f"Completed processing {result_fn}")
    log_number_nans_in_df(samples)

def log_number_nans_in_df(df):
    tot = len(df)
    num_nans = df.isna().any(axis=1).sum()
    print(f"Nans in {num_nans}/{tot} rows")

def main():
    result, hyper_params = get_cli_args()
    process_r(result, hyper_params)


def test():
    fname= "/Users/avaj0001/Documents/projects/agn_phenomenological_model/studies/determining_sampler_settings/run_d/result/run_d_0_result.json"
    hyper_param = dict(sigma_1=0.5, sigma_12=0.5)
    process_r(fname, hyper_param)


if __name__ == '__main__':
    test()