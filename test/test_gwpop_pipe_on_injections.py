import gwpopulation_pipe
import os
from bilby.core.utils import logger
import pandas as pd

def load_args(ini="injection_study_gwpop_pipe.ini"):    
    parser = gwpopulation_pipe.main.create_parser()
    args, _ = parser.parse_known_args([ini])
    args.dat_samples_regex = "small_test_dir/*.dat"
    args.run_dir = "small_test_dir/test_out"
    args.log_dir = "small_test_dir/test_out/logs"
    
    
    complete_ini_file = f"{args.run_dir}/{args.label}_config_complete.ini"
    args.ini_file = complete_ini_file
    gwpopulation_pipe.main.make_dag(args)
    gwpopulation_pipe.main.make_submit_files(args)
    parser.write_to_file(
        filename=complete_ini_file,
        args=args,
        overwrite=True,
        include_description=False,
    )
    with open(complete_ini_file, "r") as ff:
        content = ff.readlines()
    for ii, line in enumerate(content):
        content[ii] = gwpopulation_pipe.main.strip_quotes(line)
    with open(complete_ini_file, "w") as ff:
        ff.writelines(content)
    
    logger.info(args)
    
    return args
    
def run_data_collection(args):
    posts, events = gwpopulation_pipe.data_collection.gather_posteriors(args=args)
    logger.info(f"Using {len(posts)} events, final event list is: {', '.join(events)}.")
    posterior_file = f"{args.data_label}.pkl"
    logger.info(f"Saving posteriors to {posterior_file}")
    filename = os.path.join(args.run_dir, "data", posterior_file)
    pd.to_pickle(posts, filename)

def run_data_analysis(args):
    parser = gwpopulation_pipe.data_analysis.create_parser()
    args, unknown_args = parser.parse_known_args("/home/avi.vajpeyi/projects/agn_phenomenological_model/simulated_events/small_test_dir/test_out/simulated_pop_config_complete.ini --prior /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/priors/mass_c_iid_mag_agn_tilt_powerlaw_redshift.prior --label simulated_pop_mass_c_iid_mag_agn_tilt_powerlaw_redshift --models SmoothedMassDistribution --models iid_spin_magnitude --models agn_spin_orientation --models gwpopulation.models.redshift.PowerLawRedshift --vt-models SmoothedMassDistribution --vt-models gwpopulation.models.redshift.PowerLawRedshift".split())
    posterior_file = os.path.join(args.run_dir, "data", f"{args.data_label}.pkl")
    posteriors = pd.read_pickle(posterior_file)
    for ii, post in enumerate(posteriors):
        posteriors[ii] = post[post["redshift"] < args.max_redshift]
    gwpopulation_pipe.data_analysis.vt_helper.N_EVENTS = len(posteriors)
    gwpopulation_pipe.data_analysis.vt_helper.max_redshift = args.max_redshift
    logger.info(f"Loaded {len(posteriors)} posteriors")
    event_ids = list()
    with open(
        os.path.join(args.run_dir, "data", f"{args.data_label}_posterior_files.txt"),
        "r",
    ) as ff:
        for line in ff.readlines():
            event_ids.append(line.split(":")[0])

    logger.info(f"VT Models = {args.vt_models}")
    logger.info(f"Tilt Models = {args.tilt_models}")
            
    hyper_prior = gwpopulation_pipe.data_analysis.load_prior(args)
    model = gwpopulation_pipe.data_analysis.load_model(args)
    selection = gwpopulation_pipe.data_analysis.load_vt(args)

    likelihood = gwpopulation_pipe.data_analysis.create_likelihood(args, posteriors, model, selection)
    likelihood.input_parameters.update(hyper_prior.sample())
    likelihood.log_likelihood_ratio()

    if args.injection_file is not None:
        injections = pd.read_json(args.injection_file)
        injection_parameters = dict(injections.iloc[args.injection_index])
    else:
        injection_parameters = None

    logger.info("Starting sampling")
    result = gwpopulation_pipe.data_analysis.run_sampler(
        likelihood=likelihood,
        priors=hyper_prior,
        label=args.label,
        sampler=args.sampler_name,
        outdir=os.path.join(args.run_dir, "result"),
        injection_parameters=injection_parameters,
        **gwpopulation_pipe.data_analysis.get_sampler_kwargs(args),
    )
    result.prior = args.prior
    result.models = args.models
    result.event_ids = event_ids

    logger.info("Computing rate posterior")
    compute_rate_posterior(posterior=result.posterior, selection=selection)
    result.save_to_file(extension="json", overwrite=True)
    logger.info("Resampling single event posteriors")
    resample_single_event_posteriors(likelihood, result, save=True)
    result.plot_corner(parameters=result.search_parameter_keys + ["log_10_rate"])
    

def main():
    args = load_args()
    logger.info("BEGINNING COLLECTION")
    run_data_collection(args)
    logger.info("BEGINNING ANALYSIS")
    run_data_analysis(args)
    
    
    
    
if __name__=="__main__":
    main()