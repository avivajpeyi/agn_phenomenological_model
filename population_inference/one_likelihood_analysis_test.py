# %%
# !/usr/bin/env python3

import inspect
import os
from importlib import import_module

import numpy as np
import pandas as pd
from bilby.core.prior import LogUniform, PriorDict
from bilby.core.utils import logger
from bilby.hyper.model import Model
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.hyperpe import HyperparameterLikelihood, RateLikelihood
from gwpopulation.models.mass import (
    BrokenPowerLawPeakSmoothedMassDistribution,
    BrokenPowerLawSmoothedMassDistribution,
    MultiPeakSmoothedMassDistribution,
    SmoothedMassDistribution,
    two_component_primary_mass_ratio,
)
from gwpopulation.models.spin import (agn_spin, iid_spin, iid_spin_magnitude_beta,
                                      iid_spin_orientation_gaussian_isotropic,
                                      independent_spin_magnitude_beta,
                                      independent_spin_orientation_gaussian_isotropic)
from gwpopulation_pipe import vt_helper
from gwpopulation_pipe.parser import create_parser as create_main_parser
from gwpopulation_pipe.utils import prior_conversion


def create_parser():
    parser = create_main_parser()
    parser.add_argument("--prior", help="Prior file readable by bilby.")
    parser.add_argument(
        "--sampler_name",
        default="dynesty",
        help="Sampler to use, allowed options are pymultinest, dynesty, nestle, "
             "cpnest, emcee.",
    )
    parser.add_argument(
        "--models",
        type=str,
        action="append",
        help="Model functions to evaluate, default is "
             "two component mass and iid spins.",
    )
    parser.add_argument(
        "--vt-models",
        type=str,
        action="append",
        help="Model functions to evaluate for selection, default is no model",
    )
    parser.add(
        "--sampler-kwargs",
        type=str,
        default="Default",
        help=(
            "Dictionary of sampler-kwargs to pass in, e.g., {nlive: 1000} OR "
            "pass pre-defined set of sampler-kwargs {Default, FastTest}"
        ),
    )
    parser.add_argument(
        "--max-samples",
        default=1e10,
        type=int,
        help="Maximum number of posterior samples per event",
    )
    parser.add_argument(
        "--rate", default=False, type=bool, help="Whether to sample in the merger rate."
    )
    parser.add_argument(
        "--max-redshift", default=2.3, type=float, help="Maximum redshift in model."
    )
    return parser


def load_prior(args):
    hyper_prior = PriorDict(filename=args.prior)
    hyper_prior.conversion_function = prior_conversion
    if args.rate:
        hyper_prior["rate"] = LogUniform(
            minimum=1e-1,
            maximum=1e3,
            name="rate",
            latex_label="$R$",
            boundary="reflective",
        )
    return hyper_prior


MODEL_MAP = {
    "two_component_primary_mass_ratio": two_component_primary_mass_ratio,
    "iid_spin": iid_spin,
    "iid_spin_magnitude": iid_spin_magnitude_beta,
    "ind_spin_magnitude": independent_spin_magnitude_beta,
    "iid_spin_orientation": iid_spin_orientation_gaussian_isotropic,
    "two_comp_iid_spin_orientation": iid_spin_orientation_gaussian_isotropic,
    "ind_spin_orientation": independent_spin_orientation_gaussian_isotropic,
    "agn_spin_orientation": agn_spin,
    "SmoothedMassDistribution": SmoothedMassDistribution,
    "BrokenPowerLawSmoothedMassDistribution": BrokenPowerLawSmoothedMassDistribution,
    "MultiPeakSmoothedMassDistribution": MultiPeakSmoothedMassDistribution,
    "BrokenPowerLawPeakSmoothedMassDistribution": BrokenPowerLawPeakSmoothedMassDistribution,
}


def load_model(args):
    if args.models is None:
        args.models = [
            "two_component_primary_mass_ratio",
            "iid_spin",
            "gwpopulation.models.redshift.PowerLawRedshift",
        ]
    logger.info(f"Loading models = {args.models}")
    model = Model([_load_model(model, args) for model in args.models])
    return model


def load_vt(args):
    if args.vt_function == "" or args.vt_file == "None":
        return vt_helper.dummy_selection
    vt_model = Model([_load_model(model, args) for model in args.vt_models])
    try:
        vt_func = getattr(vt_helper, args.vt_function)
        return vt_func(args.vt_file, model=vt_model)
    except AttributeError:
        return vt_helper.injection_resampling_vt(vt_file=args.vt_file, model=vt_model)


def _load_model(model, args):
    if "." in model:
        split_model = model.split(".")
        module = ".".join(split_model[:-1])
        function = split_model[-1]
        _model = getattr(import_module(module), function)
        logger.info(f"Using {function} from {module}.")
    elif model in MODEL_MAP:
        _model = MODEL_MAP[model]
        logger.info(f"Using {model}.")
    else:
        raise ValueError(f"Model {model} not found.")
    if inspect.isclass(_model):
        if "redshift" in model.lower():
            kwargs = dict(z_max=args.max_redshift)
        else:
            kwargs = dict()
        _model = _model(**kwargs)
    return _model


def create_likelihood(args, posteriors, model, selection):
    if args.rate:
        likelihood_class = RateLikelihood
    else:
        likelihood_class = HyperparameterLikelihood
    likelihood = likelihood_class(
        posteriors,
        model,
        conversion_function=convert_to_beta_parameters,
        selection_function=selection,
        max_samples=args.max_samples,
    )

    return likelihood


def get_sampler_kwargs(args):
    if args.sampler_kwargs == "Default":
        sampler_kwargs = dict()
    elif not isinstance(args.sampler_kwargs, dict):
        sampler_kwargs = dict()
        for arg in args.sampler_kwargs:
            key = arg.split(":")[0].strip()
            value = arg.split(":")[1].strip()
            try:
                value = eval(value)
            except NameError:
                pass
            sampler_kwargs[key] = value
    else:
        sampler_kwargs = args.sampler_args
    if args.sampler_name == "cpnest" and "seed" not in sampler_kwargs:
        sampler_kwargs["seed"] = np.random.randint(0, 1e6)
    sampler_kwargs["nlive"] = 500
    sampler_kwargs["nact"] = 2
    sampler_kwargs["walks"] = 5
    return sampler_kwargs


def get_likelihood_and_hyoperprior_from_args(cli_str):
    parser = create_parser()
    cli_args = cli_str.split(" ")[1:]
    print(cli_args)
    args, unknown_args = parser.parse_known_args(cli_args)
    posterior_file = os.path.join(args.run_dir, "data", f"{args.data_label}.pkl")
    posteriors = pd.read_pickle(posterior_file)
    for ii, post in enumerate(posteriors):
        posteriors[ii] = post[post["redshift"] < args.max_redshift]
    vt_helper.N_EVENTS = len(posteriors)
    vt_helper.max_redshift = args.max_redshift
    logger.info(f"Loaded {len(posteriors)} posteriors")
    event_ids = list()
    with open(
            os.path.join(args.run_dir, "data",
                         f"{args.data_label}_posterior_files.txt"),
            "r",
    ) as ff:
        for line in ff.readlines():
            event_ids.append(line.split(":")[0])

    hyper_prior = load_prior(args)
    model = load_model(args)
    selection = load_vt(args)

    likelihood = create_likelihood(args, posteriors, model, selection)
    return likelihood, hyper_prior


# %%


LOCAL = "gwpopulation_pipe_analysis /Users/avaj0001/Documents/projects/agn_phenomenological_model/population_inference/test_outdir/test_config_complete.ini " \
        "--prior /Users/avaj0001/Documents/projects/agn_phenomenological_model/population_inference/priors/mass_c_iid_mag_agn_tilt_powerlaw_redshift.prior " \
        "--label test_mass_c_iid_mag_agn_tilt_powerlaw_redshift " \
        "--models SmoothedMassDistribution " \
        "--models iid_spin_magnitude " \
        "--models agn_spin_orientation " \
        "--models gwpopulation.models.redshift.PowerLawRedshift " \
        "--vt-models SmoothedMassDistribution " \
        "--vt-models gwpopulation.models.redshift.PowerLawRedshift"
CIT = "gwpopulation_pipe_analysis /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_outdir/agn_config_complete.ini --prior /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/priors/mass_c_iid_mag_agn_tilt_powerlaw_redshift.prior --label agn_mass_c_iid_mag_agn_tilt_powerlaw_redshift --models SmoothedMassDistribution --models iid_spin_magnitude --models agn_spin_orientation --models gwpopulation.models.redshift.PowerLawRedshift --vt-models SmoothedMassDistribution --vt-models gwpopulation.models.redshift.PowerLawRedshift"

from pprint import pprint


def test_likelihood(likelihood, hyper_prior):
    theta = hyper_prior.sample()
    likelihood.parameters.update(theta)
    ln_l = likelihood.log_likelihood()
    ratio = likelihood.log_likelihood_ratio()
    params = {
        key: t for key, t in zip(hyper_prior.keys(), theta.values())
    }
    ln_p = hyper_prior.ln_prob(params)
    pprint(dict(ln_likelihood=ln_l, ln_prior=ln_p, log_likelihood_ratio=ratio, **theta))


if __name__ == '__main__':
    likelihood, hyperprior = get_likelihood_and_hyoperprior_from_args(CIT)
    test_likelihood(likelihood, hyperprior)
