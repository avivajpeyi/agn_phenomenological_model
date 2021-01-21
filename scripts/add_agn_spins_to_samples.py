import glob
import logging
import os
import shutil
from collections import namedtuple

import bilby
import bilby_pipe
import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sys

logging.getLogger('bilby_pipe').setLevel(logging.ERROR)

NUM_PRIOR_POINTS = 5000


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

    prior_label = os.path.basename(complete_inputs.prior_file).split(".prior")[0]
    prior_samples_file = glob.glob(os.path.join(os.path.dirname(ini_file), f"{prior_label}*.dat"))[0]

    prior_samples = pd.read_csv(prior_samples_file, index_col="id", sep=",").sample(NUM_PRIOR_POINTS)
    # bilby_pipe.input.Input.get_default_prior_files()["4s"]
    # calculate kicks for the priors before hand and save them in
    prior_samples = bilby.gw.conversion.generate_all_bbh_parameters(prior_samples)
    return prior_samples


VIOLET_COLOR = "#8E44AD"
BILBY_BLUE_COLOR = '#0072C1'

PARAMS = dict(
    chi_eff=dict(l=r"$\chi_{eff}$", r=(-1, 1)),
    chi_p=dict(l=r"$\chi_{p}$", r=(-1, 1)),
    cos_tilt_1=dict(l=r"$\cos(t1)$", r=(-1, 1)),
    # cos_tilt_2=dict(l=r"$\cos(t2)$", r=(-1, 1)),
    cos_theta_12=dict(l=r"$\cos \theta_{12}$", r=(-1, 1)),
    # cos_theta_1L=dict(l=r"$\cos \theta_{1L}$", r=(-1, 1)),
    # tilt_1=dict(l=r"$tilt_{1}$", r=(0, np.pi)),
    # theta_1L=dict(l=r"$\theta_{1L}$", r=(0, np.pi)),
    # diff=dict(l=r"diff", r=(-np.pi, np.pi)),
    remnant_kick_mag=dict(l=r'$|\vec{v}_k|\ $km/s', r=(0,3000))
)

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color='tab:orange',
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)


def add_agn_samples_to_df(df):
    df['s1x'], df['s1y'], df['s1z'] = df['spin_1x'], df['spin_1y'], df['spin_1z']
    df['s2x'], df['s2y'], df['s2z'] = df['spin_2x'], df['spin_2y'], df['spin_2z']
    df['s1_dot_s2'] = (df['s1x'] * df['s2x']) + (df['s1y'] * df['s2y']) + (
            df['s1z'] * df['s2z'])
    df['s1_mag'] = np.sqrt(
        (df['s1x'] * df['s1x']) + (df['s1y'] * df['s1y']) + (df['s1z'] * df['s1z']))
    df['s2_mag'] = np.sqrt(
        (df['s2x'] * df['s2x']) + (df['s2y'] * df['s2y']) + (df['s2z'] * df['s2z']))
    df['cos_theta_12'] = df['s1_dot_s2'] / (df['s1_mag'] * df['s2_mag'])
    # Lhat = [0, 0, 1]
    df['cos_theta_1L'] = df['s1z'] / (df['s1_mag'])
    df['tilt1'], df['theta_1L'] = np.arccos(df['cos_tilt_1']), np.arccos(
        df['cos_theta_1L'])
    df['diff'] = df['tilt1'] - df['theta_1L']
    df = calculate_weight(df, sigma=0.5)
    return df


def get_one_dimensional_median_and_error_bar(posterior, key, fmt='.2f',
                                             quantiles=(0.16, 0.84)):
    """ Calculate the median and error bar for a given key

    Parameters
    ----------
    key: str
        The parameter key for which to calculate the median and error bar
    fmt: str, ('.2f')
        A format string
    quantiles: list, tuple
        A length-2 tuple of the lower and upper-quantiles to calculate
        the errors bars for.

    Returns
    -------
    summary: namedtuple
        An object with attributes, median, lower, upper and string

    """
    summary = namedtuple('summary', ['median', 'lower', 'upper', 'string'])

    if len(quantiles) != 2:
        raise ValueError("quantiles must be of length 2")

    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(posterior[key], quants_to_compute * 100)
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]

    fmt = "{{0:{0}}}".format(fmt).format
    string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    summary.string = string_template.format(
        fmt(summary.median), fmt(summary.minus), fmt(summary.plus))
    return summary


def calculate_weight(df, sigma):
    """Calc weights from AGN prior"""
    mean = 1
    clip_a, clip_b = -1, 1
    a, b = (clip_a - mean) / sigma, (clip_b - mean) / sigma
    costheta_prior = scipy.stats.truncnorm(a=a, b=b, loc=mean, scale=sigma)
    df['weight'] = np.abs(np.exp(costheta_prior.logpdf(df['cos_theta_1L'])))
    df['weight'] = df['weight'] * np.abs(
        np.exp(costheta_prior.logpdf(df['cos_theta_12'])))
    return df


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)


def process_res(r, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    params = list(PARAMS.keys())
    labels = [PARAMS[p]['l'] for p in params]
    param_range = [PARAMS[p]['r'] for p in params]
    label = os.path.basename(r).split("_")[0]
    settings_file = os.path.join(os.path.dirname(r), f"{label}.ini")
    res_df = add_agn_samples_to_df(pd.read_csv(r, index_col="id",  sep=","))  # .sample(NUM_PRIOR_POINTS)
    prior_df = add_agn_samples_to_df(get_prior_samples_from_settings(settings_file))

    plot_corner(labels, prior_df, res_df, params, outdir, label, param_range)


def plot_corner(labels, prior_df, res_df, params, outdir, label, param_range):
    # Plot posterior samples
    print("Plotting")

    kwargs = CORNER_KWARGS.copy()
    kwargs.update(labels=labels, range=param_range)
    fig = corner.corner(prior_df[params], color='C2', **kwargs)
    normalising_weights = get_normalisation_weight(len(res_df),
                                                   max(len(prior_df), len(res_df)))
    corner.corner(res_df[params], fig=fig, weights=normalising_weights, **kwargs,
                  color=BILBY_BLUE_COLOR)

    # plt the quantiles
    axes = fig.get_axes()
    for i, par in enumerate(params):
        ax = axes[i + i * len(params)]
        if ax.title.get_text() == '':
            ax.set_title(get_one_dimensional_median_and_error_bar(
                res_df, par,
                quantiles=kwargs['quantiles']).string,
                         **kwargs['title_kwargs'])

    res_line = mlines.Line2D([], [], color=BILBY_BLUE_COLOR, label="Posterior")
    prior_line = mlines.Line2D([], [], color='C2', label="Prior")
    plt.legend(handles=[res_line, prior_line], fontsize=16,
               frameon=False,
               bbox_to_anchor=(1, len(labels)), loc="upper right")

    fig.suptitle(label, fontsize=24)
    fig.savefig(os.path.join(outdir, f"{label}.png"))
    plt.close(fig)

def main():
    outdir = "../output/"
    res = glob.glob(
        "../data/gwtc1_with_kicks/*samples_with_kicks.dat")
    for r in res:
        try:
            process_res(r, outdir)
            print(f"Processed {r}")
        except Exception:
            print(f"Skipping {r}")


if __name__ == '__main__':
    process_res(
        r=sys.argv[1],
        outdir="../output/gwtc2_with_spins"
    )
