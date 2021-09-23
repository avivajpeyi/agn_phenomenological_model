import matplotlib.pyplot as plt
from typing import List, Optional
import seaborn as sns
import numpy as np
from bilby.core.prior import TruncatedNormal
import random

def get_colors(num_colors: int, alpha: Optional[float]=1) -> List[List[float]]:
    """Get a list of colorblind colors,
    :param num_colors: Number of colors.
    :param alpha: The transparency
    :return: List of colors. Each color is a list of [r, g, b, alpha].
    """
    cs = sns.color_palette(palette="colorblind", n_colors=num_colors)
    cs = [list(c) for c in cs]
    for i in range(len(cs)):
        cs[i].append(alpha)
    return cs

def update_style():
    try:
        plt.style.use(
            "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/55fab35b1c811dc3c0fdc6d169d67607e2c3edfc/publication.mplstyle")
    except Exception:
        pass
    plt.grid(False)




def add_cdf_percentiles_to_ax(posteriors_list, ax, label="", add_each_posterior_cdf=True, color=None, true_distribution_data=[]):

    zor = dict(ci_99=-4, ci_90=-3, ci_line=-2, tru=-1)

    if isinstance(posteriors_list, list):
        min_len = min([len(i) for i in posteriors_list])
        posteriors_list = np.array([np.random.choice(i, min_len, replace=False) for i in posteriors_list])

    # CI for each event
    cumulative_prob = np.linspace(0, 1, len(posteriors_list[:, 0]))  # one bin for each event
    sorted_posterior = np.sort(posteriors_list, axis=0)  # sort amongts various posteriors

    data_05_percentile = np.quantile(sorted_posterior, 0.05, axis=1)  # get 0.05 CI from all events' posteriors
    data_95_percentile = np.quantile(sorted_posterior, 0.95, axis=1)  # get 0.95 CI from all events' posteriors

    data_005_percentile = np.quantile(sorted_posterior, 0.005, axis=1)  # get 0.05 CI from all events' posteriors
    data_995_percentile = np.quantile(sorted_posterior, 0.995, axis=1)  # get 0.95 CI from all events' posteriors


    if len(posteriors_list)==1:
        post = posteriors_list[0]
        ax.plot(np.sort(post), np.linspace(0, 1, len(post)),color=color, lw=2.0, zorder=zor['tru'])

    ax.fill_betweenx(
        y=cumulative_prob,
        x1=data_05_percentile,
        x2=data_95_percentile,
        alpha=0.5, label=label,
        color=color,
        zorder=zor['ci_90']
    )
    ax.fill_betweenx(
        y=cumulative_prob,
        x1=data_005_percentile,
        x2=data_995_percentile,
        alpha=0.2, label=label,
        color=color,
        zorder=zor['ci_99']
    )

    ax.plot(data_05_percentile, cumulative_prob, color=color, lw=0.5, alpha=0.5,zorder=zor['ci_line'])
    ax.plot(data_95_percentile, cumulative_prob, color=color, lw=0.5, alpha=0.5,zorder=zor['ci_line'])
    #
    ax.plot(data_005_percentile, cumulative_prob, color=color, lw=0.5, alpha=0.5,zorder=zor['ci_line'])
    ax.plot(data_995_percentile, cumulative_prob, color=color, lw=0.5, alpha=0.5,zorder=zor['ci_line'])

    if add_each_posterior_cdf:
        for post in posteriors_list:
            post = np.sort(post)
            ax.plot(post, np.linspace(0, 1, len(post)), alpha=0.05, color='k')
    if len(true_distribution_data)>0:
        ax.plot(np.sort(true_distribution_data), np.linspace(0, 1, len(true_distribution_data)), alpha=1.0, color=color, zorder=12)


def plot_posterior_predictive_check(data_sets, rhs_ax_labels, colors=[],  add_posteriors = False, axes =[]):
    """Plots CDF plot_posterior_predictive_check."""
    plt.grid(False)
    if len(axes)==0:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if len(colors)==0:
        colors = get_colors(len(data_sets))
    if len(rhs_ax_labels) < len(data_sets):
        rhs_ax_labels += [""] * (len(data_sets) - len(rhs_ax_labels))

    for i, data in enumerate(data_sets):
        add_cdf_percentiles_to_ax(posteriors_list=data['cos_tilt_1'], ax=axes[0], add_each_posterior_cdf=add_posteriors, color=colors[i], true_distribution_data=[])
        add_cdf_percentiles_to_ax(posteriors_list=data['cos_theta_12'], ax=axes[1], label=rhs_ax_labels[i], add_each_posterior_cdf=add_posteriors, color=colors[i], true_distribution_data=[])

    for i, ax in enumerate(axes):
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 1])

        if (i == 0):
            ax.set_xlabel(r"$\cos\ \theta_1$")
            # ax.set_ylabel("CDF")
        else:
            ax.set_xlabel(r"$\cos\ \theta_{12}$")
            ax.set_yticklabels([])
            if ("".join(rhs_ax_labels) != ""):
                ax.legend(fontsize='small')
        ax.grid(False)

def plot_trues(data_sets, trues, rhs_ax_labels, colors=[], axes=[]):
    if len(axes) == 0:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if len(colors)==0:
        colors = get_colors(len(data_sets))

    for i, (pop_name, t_vals) in enumerate(trues.items()):
        x = np.linspace(-1,1, 100)
        y1 = TruncatedNormal(mu=1, sigma=t_vals[0], minimum=-1, maximum=1).prob(x)
        y2 = TruncatedNormal(mu=1, sigma=t_vals[1], minimum=-1, maximum=1).prob(x)
        axes[1].plot(x,y2, color=colors[i], zorder=10, lw=3 )
        axes[0].plot(x,y1, color=colors[i], zorder=10, lw=3 )
        axes[i].set_xlim([-1, 1])

    hatches = ['///', "\\\\\\"]

    for i, data in enumerate(data_sets):
        c = colors[len(trues) + i]
        kwargs = dict(density=True, histtype='stepfilled',lw=2.0, alpha=0.5, color=c, hatch=hatches[i], edgecolor=c)
        axes[0].hist(data['cos_tilt_1'],  **kwargs)
        axes[1].hist(data['cos_theta_12'],  **kwargs)

        if (i == 0):
            axes[i].set_xlabel(r"$\cos\ \theta_1$")
            axes[i].set_ylabel("PDF (trues)")
            axes[i].set_ylim(0, 2.5)
        else:
            axes[i].set_xlabel(r"$\cos\ \theta_{12}$")
            axes[i].set_yticklabels([])
            axes[i].set_ylim(0, 2.5)
            # axes[i].legend(fontsize='small')

        axes[i].grid(False)
