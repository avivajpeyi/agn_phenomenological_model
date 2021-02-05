from gwpopulation_pipe.data_analysis import *
from configargparse import Namespace
from pprint import pprint

EXECUTION_STR = "gwpopulation_pipe_analysis /Users/avaj0001/Documents/projects/agn_phenomenological_model/population_inference/agn_outdir/agn_config_complete.ini --prior /Users/avaj0001/Documents/projects/agn_phenomenological_model/population_inference/priors/mass_c_iid_mag_agn_tilt_powerlaw_redshift.prior --label agn_mass_c_iid_mag_agn_tilt_powerlaw_redshift --models SmoothedMassDistribution --models iid_spin_magnitude --models agn_spin_orientation --models gwpopulation.models.redshift.PowerLawRedshift --vt-models SmoothedMassDistribution --vt-models gwpopulation.models.redshift.PowerLawRedshift --verbose"

#"--models agn_spin_orientation " \

def test_likelihood(likelihood, hyper_prior):
    theta = hyper_prior.sample()
    likelihood.parameters.update(theta)
    ln_l = likelihood.log_likelihood()
    ratio = likelihood.log_likelihood_ratio()
    params = {
        key: t for key, t in zip(hyper_prior.keys(), theta.values())
    }
    ln_p = hyper_prior.ln_prob(params)
    pprint(theta)
    pprint(dict(ln_likelihood=ln_l, ln_prior=ln_p, log_likelihood_ratio=ratio))

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
    rand_sample = hyper_prior.sample()
    likelihood.parameters.update(rand_sample)
    likelihood.log_likelihood_ratio()
    test_likelihood(likelihood, hyper_prior)

    logger.debug("Starting sampling")
    run_sampler(
        likelihood=likelihood,
        priors=hyper_prior,
        label=args.label,
        sampler=args.sampler_name,
        outdir=os.path.join(args.run_dir, "result"),
        **get_sampler_kwargs(args),
    )



if __name__ == '__main__':
    main()