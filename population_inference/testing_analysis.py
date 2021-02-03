from gwpopulation_pipe.data_analysis import *
from configargparse import Namespace

EXECUTION_STR = "gwpopulation_pipe_analysis /Users/avaj0001/Documents/projects/agn_phenomenological_model/population_inference/outdir/test_config_complete.ini " \
                "--prior /Users/avaj0001/Documents/projects/agn_phenomenological_model/population_inference/priors/mass_c_iid_mag_agn_tilt_powerlaw_redshift.prior " \
                "--label test_mass_c_iid_mag_agn_tilt_powerlaw_redshift " \
                "--models SmoothedMassDistribution " \
                "--models iid_spin_magnitude " \
                "--models ind_spin_orientation " \
                "--models gwpopulation.models.redshift.PowerLawRedshift " \
                "--vt-models SmoothedMassDistribution " \
                "--vt-models gwpopulation.models.redshift.PowerLawRedshift"

#"--models agn_spin_orientation " \

def main():
    parser = create_parser()
    args, unknown_args = parser.parse_known_args(EXECUTION_STR.split(" ")[1:])
    posterior_file = os.path.join(args.run_dir, "data", f"{args.data_label}.pkl")
    posteriors = pd.read_pickle(posterior_file)
    for ii, post in enumerate(posteriors):
        posteriors[ii] = post[post["redshift"] < args.max_redshift]
    vt_helper.N_EVENTS = len(posteriors)
    vt_helper.max_redshift = args.max_redshift
    logger.info(f"Loaded {len(posteriors)} posteriors")
    event_ids = list()
    with open(
        os.path.join(args.run_dir, "data", f"{args.data_label}_posterior_files.txt"),
        "r",
    ) as ff:
        for line in ff.readlines():
            event_ids.append(line.split(":")[0])

    hyper_prior = load_prior(args)
    model = load_model(args)
    selection = load_vt(args)

    likelihood = create_likelihood(args, posteriors, model, selection)
    likelihood.parameters.update(hyper_prior.sample())
    likelihood.log_likelihood_ratio()

    if args.injection_file is not None:
        injections = pd.read_json(args.injection_file)
        injection_parameters = dict(injections.iloc[args.injection_index])
    else:
        injection_parameters = None

    logger.debug("Starting sampling")
    result = run_sampler(
        likelihood=likelihood,
        priors=hyper_prior,
        label=args.label,
        sampler=args.sampler_name,
        outdir=os.path.join(args.run_dir, "result"),
        injection_parameters=injection_parameters,
        **get_sampler_kwargs(args),
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


if __name__ == '__main__':
    main()