import logging
import os

import matplotlib
import matplotlib.image
import pandas as pd
from agn_utils.create_agn_samples import get_bbh_population_from_agn_params
from agn_utils.create_agn_samples import (
    load_training_data,
    save_agn_samples_for_many_populations
)
from agn_utils.diagnostic_tools import timing
from agn_utils.plotting import overlaid_corner
from agn_utils.regressors.scikit_regressor import ScikitRegressor
from agn_utils.regressors.tf_regressor import TfRegressor
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@timing
def generate_chi_regression_training_data(training_data_fname):
    if os.path.isfile(training_data_fname):
        raise FileExistsError("Training data already exists.")
    logger.info("Generating traininng data")
    save_agn_samples_for_many_populations(num=1, fname=training_data_fname)
    logger.info("Finished generating traininng data")
    df = load_training_data(training_data_fname)
    logging.debug(df.describe())


@timing
def get_training_data(training_data_fname, num_samples=None):
    logger.info(f"Loading training samples.")
    df = load_training_data(training_data_fname)
    init_len = len(df)
    if num_samples:
        df = df.sample(num_samples)
    df['p'] = df['p_cos_theta_1'] * df['p_cos_theta_12']
    logger.info(f"Loaded {len(df)}/{init_len} training samples.")
    return df


def instantiate_regression_model(model="Scikit"):
    kwargs = dict(
        input_parameters=["sigma_1", "sigma_12", "chi_eff", "chi_p"],
        output_parameters=['p']
    )
    if model=="Scikit":
        regressor = ScikitRegressor(
            model_hyper_param=dict(verbose=2),
            **kwargs
        )
    else:
        regressor = TfRegressor(**kwargs)
    return regressor


@timing
def train_model(training_data, model_fname, model_type="Scikit"):
    regressor = instantiate_regression_model(model_type)
    logger.info(f"{model_type} Regressor Training initiated.")
    regressor.train(data=training_data)
    regressor.save(filename=model_fname)
    logger.info(f"Training complete (model saved at {model_fname}).")


def load_model(model_fname, model_type="Scikit"):
    regressor = instantiate_regression_model(model_type)
    regressor.load(model_fname)
    logger.info(f"{model_type} Regressor model loaded from {model_fname}.")
    return regressor


def make_prediction_with_model(input_data, regressor):
    expected_in = set(regressor.input_parameters)
    provided_in = set(input_data.columns.values)
    assert expected_in.issubset(provided_in), f"{expected_in} not in {provided_in}"
    predicted_p = regressor.predict(input_data[regressor.input_parameters])
    plot_prediction_vs_true(input_data, predicted_p)
    return predicted_p


def prediction_for_different_sigma(sigma_1, sigma_12, n, regressor):
    df = get_bbh_population_from_agn_params(
        num_samples=n, sigma_1=sigma_1,
        sigma_12=sigma_12)
    df['sigma_1'], df['sigma_12'] = sigma_1, sigma_12
    title = r"$\sigma_{1}=" + f"{sigma_1:.2f}" + ",\sigma_{12}= " + f"{sigma_12:.2f}$"
    predicted_p = make_prediction_with_model(df, regressor)
    plot_prediction_vs_true(df, predicted_p)
    img = matplotlib.image.imread('model_prediction.png')
    plt.figure(figsize=(15, 15))
    plt.suptitle(title, fontsize=40)
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def plot_prediction_vs_true(data, predicted_p, title=None):
    true = data.copy()
    true['p(chi|sigma)'] = data['p_cos_theta_12'] * data['p_cos_tilt_1']

    predicted = pd.DataFrame({
        "cos_tilt_1": data['cos_tilt_1'],
        "cos_theta_12": data['cos_theta_12'],
        "chi_p": data['chi_p'],
        "chi_eff": data['chi_eff'],
        "p(chi|sigma)": predicted_p
    })

    overlaid_corner(
        samples_list=[true, predicted],
        sample_labels=["True", "Predicted"],
        params=['cos_tilt_1', 'cos_theta_12', 'chi_eff', 'chi_p', "p(chi|sigma)"],
        samples_colors=["tab:blue", "tab:orange"],
        fname="model_prediction.png",
        title=title
    )


def chi_regressor_trainer(model_fname: str, training_fname):
    training_df = get_training_data(training_fname)
    logger.debug(training_df.describe().T[['min', 'mean', 'max']])
    train_model(training_df, model_fname=model_fname)


def chi_regressor_predictor(input_data: pd.DataFrame, model_fname: str):
    regressor = load_model(model_fname)
    return make_prediction_with_model(input_data, regressor)
