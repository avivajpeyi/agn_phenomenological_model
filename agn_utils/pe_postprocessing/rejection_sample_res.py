from agn_utils.bbh_population_generators.calculate_extra_bbh_parameters import result_post_processing
from .posterior_reweighter import rejection_sample_posterior
import sys
import bilby

import argparse

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="path to res", type=str)
    parser.add_argument("sig1", help="sig1 tru val", type=float)
    parser.add_argument("sig12", help="sig12 tru val", type=float)
    args =  parser.parse_args()
    return args.result, dict(sigma_1=args.sig1, sigma_12=args.sigma_12)


def main():
    result, hyper_params = get_cli_args()
    r = bilby.result.Result.from_json(filename=result)
    r = result_post_processing(r)
    samples = rejection_sample_posterior(r.posterior, hyper_param=hyper_params)
    samples.to_hdf(result.replace(".json", "_reweighted.h5"))
    print(f"Completed processing {result}")


