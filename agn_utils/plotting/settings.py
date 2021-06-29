import re
from typing import List, Optional

import seaborn as sns
from matplotlib import rcParams

VIOLET_COLOR = "#8E44AD"
BILBY_BLUE_COLOR = "#0072C1"


def set_matplotlib_style_settings():
    rcParams["font.size"] = 30
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    rcParams["text.usetex"] = True
    rcParams["axes.labelsize"] = 30
    rcParams["axes.titlesize"] = 30
    rcParams["axes.labelpad"] = 10
    rcParams["axes.linewidth"] = 2.5
    rcParams["axes.edgecolor"] = "black"
    rcParams["xtick.labelsize"] = 25
    rcParams["xtick.major.size"] = 10.0
    rcParams["xtick.minor.size"] = 5.0
    rcParams["ytick.labelsize"] = 25
    rcParams["ytick.major.size"] = 10.0
    rcParams["ytick.minor.size"] = 5.0
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.minor.width"] = 1
    rcParams["xtick.major.width"] = 3
    rcParams["ytick.minor.width"] = 1
    rcParams["ytick.major.width"] = 2.5
    rcParams["xtick.top"] = True
    rcParams["ytick.right"] = True


def get_colors(
    num_colors: int, alpha: Optional[float] = 1
) -> List[List[float]]:
    """Get a list of colorblind samples_colors,
    :param num_colors: Number of samples_colors.
    :param alpha: The transparency
    :return: List of samples_colors. Each color is a list of [r, g, b, alpha].
    """
    palettes = ["colorblind", "ch:start=.2,rot=-.3"]
    cs = sns.color_palette(palettes[0], n_colors=num_colors)
    cs = [list(c) for c in cs]
    for i in range(len(cs)):
        cs[i].append(alpha)
    return cs


def get_event_name(fname):
    name = re.findall(r"(\w*\d{6}[a-z]*)", fname)
    if len(name) == 0:
        name = re.findall(r"inj\d+", fname)
    if len(name) == 0:
        name = re.findall(r"posteriors_list\d+", fname)
    if len(name) == 0:
        name = os.path.basename(fname).split(".")
    return name[0]
