import sys

import bilby
import pandas as pd
from bilby.core.prior import PriorDict, Uniform
from gwpopulation.hyperpe import HyperparameterLikelihood
from gwpopulation.models.spin import agn_spin

from agn_utils.data_formetter import dl_to_ld
from agn_utils.pe_postprocessing.jsons_to_numpy import load_posteriors_and_trues
from agn_utils.stats_printer import print_cpu_info, print_gpu_info
import os

BOUNDS = dict(
    cos_theta_1=(-1, 1),
    cos_theta_12=(-1, 1),
)

PRIOR_VOLUME = (
        (BOUNDS["cos_theta_1"][1] - BOUNDS["cos_theta_1"][0])
        * (BOUNDS["cos_theta_12"][1] - BOUNDS["cos_theta_12"][0])
)


def get_agn_spin_hyper_pe_prior():
    p = PriorDict()
    p["sigma_1"] = Uniform(minimum=1e-2, maximum=4, latex_label='$\\sigma_{1}$')
    p["sigma_12"] = Uniform(minimum=1e-2, maximum=4, latex_label='$\\sigma_{12}$')
    return p


def load_posteriors(pickle_fname):
    """
    posteriors: list
    An list of pandas data frames of samples sets of samples.
    Each set may have a different size.

    The following columns are needed:
    cos_tilt_1, cos_theta_12, prior

    prior` column containing the original prior

    """
    dat = load_posteriors_and_trues(pickle_fname)
    posteriors = dl_to_ld(dat['posteriors'])
    posteriors_dfs = []
    for p in posteriors:
        df = pd.DataFrame(p)
        df["prior"] = 1 / PRIOR_VOLUME
        posteriors_dfs.append(df)

    return posteriors_dfs


def get_agn_spin_likelihood(posteriors):
    return HyperparameterLikelihood(
        posteriors=posteriors,
        hyper_prior=agn_spin
    )


def main():
    print_cpu_info()
    print_gpu_info()

    # load posteriors
    fname = sys.argv[1]
    sig_1 = sys.argv[2]
    sig_12 = sys.argv[3]
    posteriors = load_posteriors(fname)
    label = os.path.basename(fname).split(".")[0]
    outdir = f"out_{label}"
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    # get hyper-prior
    prior = get_agn_spin_hyper_pe_prior()

    # get likelihood
    likelihood = get_agn_spin_likelihood(posteriors)

    likelihood.parameters.update(prior.sample())
    likelihood.log_likelihood_ratio()

    likelihood.parameters.update(dict(sigma_1=sig_1, sigma_12=sig_12))
    likelihood.log_likelihood_ratio()

    print(f"True LnL: {likelihood.log_likelihood()}")

    hyper_pe_result = bilby.run_sampler(
        likelihood=likelihood, priors=prior,
        sampler='dynesty', nlive=1000,
        outdir=outdir, label=label
    )
    hyper_pe_result.plot_corner(save=True)

    print(f"Max LnL: {likelihood.log_likelihood()}")


if __name__ == '__main__':
    main()
