# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + pycharm={"name": "#%%\n"}
from __future__ import print_function
from ipywidgets import interactive, FloatSlider, Layout, IntSlider
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt, random
from make_interaction_gif import plot_one_corner
import warnings

warnings.filterwarnings("ignore")


slider_kwargs = dict(
    style={"description_width": "100px"},
    items_layout=Layout(height="auto", width="auto"),
    continuous_update=False,
)
interactive_plot = interactive(
    plot_one_corner,
    show_aligned=False,
    show_isotropic=False,
    show_agn=True,
    sig1=FloatSlider(
        description=r"Truncnorm $\sigma_{1}$:",
        value=1,
        min=0.1,
        max=5.0,
        step=0.1,
        **slider_kwargs
    ),
    sig12=FloatSlider(
        description=r"Truncnorm $\sigma{12}$:",
        value=1,
        min=0.1,
        max=5.0,
        step=0.1,
        **slider_kwargs
    ),
    n=IntSlider(
        description="Num Samples",
        value=10000,
        min=1000,
        max=100000,
        step=1000,
        **slider_kwargs
    ),
)
output = interactive_plot.children[-1]
output.layout.height = "1000px"
output.layout.align_content
interactive_plot
# -
