import logging
from collections import namedtuple

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import PARAMS
from .settings import set_matplotlib_style_settings

logger = logging.getLogger(__name__)
set_matplotlib_style_settings()

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


def add_titles_to_corner(fig, params, posterior: pd.DataFrame):
    # plt the quantiles
    axes = fig.get_axes()
    for i, par in enumerate(params):
        ax = axes[i + i * len(params)]
        if ax.title.get_text() == '':
            ax.set_title(get_one_dimensional_median_and_error_bar(
                posterior, par,
                quantiles=CORNER_KWARGS['quantiles']).string,
                         **CORNER_KWARGS['title_kwargs'])


def overlaid_corner(samples_list, sample_labels, params,
                    samples_colors, fname="", title=None, truths={}):
    """Plots multiple corners on top of each other"""
    logger.info(f"Plotting {fname}")
    logger.info(f"Cols in samples: {samples_list[0].columns.values}")
    # sort the sample columns
    samples_list = [s[params] for s in samples_list]

    # get plot range, latex labels, colors and truths
    plot_range = [PARAMS[p]['range'] for p in params]

    axis_labels = [PARAMS[p]['latex_label'] for p in params]

    if len(truths) == 0:
        truths = None
    else:
        truths = [truths[k] for k in params]

    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = max([len(s) for s in samples_list])

    CORNER_KWARGS.update(
        range=plot_range,
        labels=axis_labels,
        truths=truths,
        truth_color=CORNER_KWARGS['truth_color'],
    )

    fig = corner.corner(
        samples_list[0],
        color=samples_colors[0],
        **CORNER_KWARGS,
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), min_len),
            color=samples_colors[idx],
            **CORNER_KWARGS
        )

    plt.legend(
        handles=[
            mlines.Line2D([], [], color=samples_colors[i], label=sample_labels[i])
            for i in range(len(sample_labels))
        ],
        fontsize=20, frameon=False,
        bbox_to_anchor=(1, ndim), loc="upper right"
    )
    if title:
        fig.suptitle(title, y=0.97)
        fig.subplots_adjust(top=0.75)
    fig.savefig(fname)
    plt.close(fig)


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)
