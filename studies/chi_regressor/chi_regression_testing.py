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

# # Chi Regressor Testing

# First we will load some modules, logging settings and set some flags for data generation and training.

# +
from __future__ import print_function

import logging
import warnings

from agn_utils.agn_logger import logger
from agn_utils.chi_regressor import (
    chi_regressor_trainer,
    generate_chi_regression_training_data, load_model, prediction_for_different_sigma,
    AvailibleRegressors
)
from ipywidgets import FloatSlider, IntSlider, Layout, interactive
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
# %matplotlib inline


TRAINING_DATA_FNAME = "training_data.h5"
MODEL_TYPE = AvailibleRegressors.TF
OUTDIR = f"{MODEL_TYPE.name.lower()}_regressor_files"

GENERATE_DATA = False
TRAIN_MODEL = True
# -

# Now we will excute the code to generate data/train/load the trained model.

# + pycharm={"name": "#%%\n"}
if GENERATE_DATA:
    print("Generating new training data.")
    generate_chi_regression_training_data(TRAINING_DATA_FNAME)
else:
    print("Using cached training data.")

# + pycharm={"name": "#%%\n"}
if TRAIN_MODEL:
    print("Training new regression model.")
    chi_regressor_trainer(outdir=OUTDIR, training_fname=TRAINING_DATA_FNAME, model_type=MODEL_TYPE, n_samples=10000)
    print("Using trained regression model.")

# + pycharm={"name": "#%%\n"}
print("Loading regression model")
regressor = load_model(outdir=OUTDIR, model_type=MODEL_TYPE)


# -

# Finally, we can plot the regressor's predicted output and compare it with the real output.

# +
def interactive_func(sigma_1, sigma_12, n):
    prediction_for_different_sigma(sigma_1, sigma_12, n, regressor=regressor)


slider_kwargs = dict(style =  {'description_width': '100px'},items_layout = Layout(height='auto', width='auto'), continuous_update=False)
interactive_plot = interactive(
    interactive_func,
    sigma_1=FloatSlider(description=r'Truncnorm $\sigma_{1}$:', value=1,min=0.1, max=4.0, step=0.1, **slider_kwargs),
    sigma_12=FloatSlider(description=r'Truncnorm $\sigma{12}$:', value=1,min=0.1, max=4.0, step=0.1, **slider_kwargs),
    n=IntSlider(description="Num Samples", value=5000,min=1000, max=100000, step=1000, **slider_kwargs),
)
output = interactive_plot.children[-1]
output.layout.height = '1100px'
output.layout.align_content
interactive_plot

# -


