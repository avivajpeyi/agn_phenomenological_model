import shutil

import bilby
import bilby_pipe
import pandas as pd

NUM_PRIOR_POINTS = 10000


def get_prior_samples_from_settings(ini_file):
    print("Getting prior samples")
    parser = bilby_pipe.main.create_parser()
    temp_outdir = "temp"
    args_list = [ini_file, "--outdir", temp_outdir]
    args, unknown_args = parser.parse_known_args(args_list)
    inputs = bilby_pipe.main.MainInput(args, unknown_args)
    bilby_pipe.main.write_complete_config_file(parser, args, inputs)
    complete_args_list = [temp_outdir + f"/{inputs.label}_config_complete.ini"]
    complete_args, complete_unknown_args = parser.parse_known_args(complete_args_list)
    complete_inputs = bilby_pipe.main.MainInput(
        complete_args, complete_unknown_args
    )
    shutil.rmtree(temp_outdir)

    prior_samples = pd.DataFrame(complete_inputs.priors.sample(NUM_PRIOR_POINTS))
    #
    # calculate kicks for the priors before hand and save them in
    prior_samples = bilby.gw.conversion.generate_all_bbh_parameters(prior_samples)
    return prior_samples


def main():
    prior_files = ["4s", "8s", "16s", "high_mass"]
    for p in prior_files:
        prior = bilby_pipe.input.Input.get_default_prior_files()[p]
        prior_df = pd.DataFrame(bilby.prior.PriorDict(filename=prior).sample(NUM_PRIOR_POINTS))
        prior_df.to_csv(f"../data/gwtc1_samples/{p}.dat", index=False, sep=" ")


if __name__ == "__main__":
    main()
