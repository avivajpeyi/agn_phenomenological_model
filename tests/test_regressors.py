import os
import unittest
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gwpopulation.models.spin import truncnorm
from numpy.random import seed, uniform
from scipy.interpolate import griddata

from agn_utils.diagnostic_tools import timing
from agn_utils.plotting.settings import get_colors
from agn_utils.regressors.regressor import Regressor
from agn_utils.regressors.scikit_regressor import ScikitRegressor
from agn_utils.regressors.tf_regressor import TfRegressor

plt.rcParams["text.usetex"] = False


class AbstractTestRegressors(unittest.TestCase):
    def setUp(self):
        self.outdir = "regression_outdir_test"
        os.makedirs(self.outdir, exist_ok=True)
        self.N = 1000
        self.set_training_params()
        self.visualise_training_data()

    def tearDown(self):
        import shutil
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def run_regressor_tests(
        self, my_regressor: Regressor, label: str, hyper_params
    ):
        model_dir = os.path.join(self.outdir, label)
        os.makedirs(model_dir, exist_ok=True)
        regressor_kwargs = dict(
            input_parameters=self.in_parameters,
            output_parameters=self.out_parameters,
            model_hyper_param=hyper_params,
            outdir=model_dir,
        )
        # Train and test
        r = my_regressor(**regressor_kwargs)
        r.train(data=self.training_data)
        r.test(
            data=self.training_data[self.in_parameters],
            labels=self.training_data[self.out_parameters],
        )
        # predict
        predictor_func = timing(r.predict)
        predicted_vals = predictor_func(self.prediction_data)
        self.assertIsNotNone(predicted_vals)

        # save
        r.save()

        # load and predict
        r = my_regressor(**regressor_kwargs)
        r.load()
        predicted_vals = predictor_func(self.prediction_data)
        self.assertIsNotNone(predicted_vals)
        self.visualise_predicted_data(
            predicted_vals=predicted_vals, label=label, n_trees=r.n_trees
        )

    def test_scikit_regressor(self):
        label = "SklearnModel"
        self.run_regressor_tests(
            ScikitRegressor, label, hyper_params=dict(verbose=2)
        )

    def test_tf_regressor(self):
        label = "TFmodel"
        model_dir = os.path.join(self.outdir, label)
        self.run_regressor_tests(
            TfRegressor, label, hyper_params=dict(model_dir=model_dir)
        )

    @abstractmethod
    def set_training_params(self):
        pass

    @abstractmethod
    def generate_fake_data(self):
        pass

    @abstractmethod
    def visualise_training_data(self):
        pass

    @abstractmethod
    def visualise_predicted_data(self, predicted_vals, n_trees, label):
        pass


def z_funct(x, y):
    return x * np.exp(-(x ** 2) - y ** 2)


@np.vectorize
def z_func_norm(x, y):
    return truncnorm(x, 1, y, 1, -1)


class FunctRegressionTest(AbstractTestRegressors):
    def set_training_params(self):
        self.training_data, self.prediction_data = self.generate_fake_data(
            z_funct
        )
        self.in_parameters = ["x", "y"]
        self.out_parameters = ["z"]

    def generate_fake_data(self, func):
        """
        Lets simulate training posteriors_list using the following formula:
        z = x * np.exp(-x ** 2 - y ** 2)
        Where (z) is the dependent variable you are trying to predict and (x) and (y) are the features
        :return:
        :rtype:
        """
        x_range = (-1, 1)
        y_range = (0, 1)

        # Create fake posteriors_list
        seed(0)
        npts = self.N
        x = uniform(x_range[0], x_range[1], npts)
        y = uniform(y_range[0], y_range[1], npts)
        z = func(x, y)

        # Prep posteriors_list for training.
        training_df = pd.DataFrame({"x": x, "y": y, "z": z})

        xi = (np.linspace(x_range[0], x_range[1], 200),)
        yi = (np.linspace(y_range[0], y_range[1], 210),)
        xi, yi = np.meshgrid(xi, yi)

        predict_df = pd.DataFrame(
            {
                "x": xi.flatten(),
                "y": yi.flatten(),
            }
        )
        return training_df, predict_df

    def visualise_training_data(self):
        x, y, z = (
            self.training_data.x,
            self.training_data.y,
            self.training_data.z,
        )
        xy = np.zeros((2, np.size(x)))
        xy[0], xy[1] = x, y
        xy = xy.T
        grid_x, grid_y = self.prediction_data.x, self.prediction_data.y
        grid_z = griddata(
            points=xy,
            values=z,
            xi=(grid_x, grid_y),
            method="linear",
            fill_value="0",
        )
        self.plot_contour(grid_z)
        plt.scatter(self.training_data.x, self.training_data.y, marker=".")
        plt.title("Contour on training posteriors_list")
        plt.savefig(os.path.join(self.outdir, "training_data.png"))

    def visualise_predicted_data(self, predicted_vals, n_trees, label):
        self.plot_contour(predicted_vals)
        plt.text(
            -1.8,
            2.1,
            f"{label} predictions: # trees: {n_trees}",
            color="w",
            backgroundcolor="black",
            size=20,
        )
        plt.savefig(os.path.join(self.outdir, f"{label}_predictions.png"))

    def plot_contour(self, pred_z):
        x, y, z = (
            self.training_data.x,
            self.training_data.y,
            self.training_data.z,
        )
        xi = (np.linspace(-1.0, 1.0, 200),)
        yi = (np.linspace(0, 1, 210),)
        xi, yi = np.meshgrid(xi, yi)
        xy = np.zeros((2, np.size(x)))
        xy[0], xy[1] = x, y
        xy = xy.T
        true_z = griddata(xy, z, (xi, yi), method="linear", fill_value="0")
        pred_z = pred_z.reshape(xi.shape)
        plt.figure(figsize=(10, 8))
        plt.contour(xi, yi, pred_z, 15, linewidths=0.5, colors="k")
        plt.contourf(
            xi,
            yi,
            pred_z,
            15,
            vmax=abs(true_z).max(),
            vmin=-abs(true_z).max(),
            cmap="RdBu_r",
        )
        plt.colorbar(label="z(x,y)")  # Draw colorbar
        plt.xlim(-1, 1)
        plt.ylim(0, 1)


class NormDistribRegressionTest(AbstractTestRegressors):
    def set_training_params(self):
        sigmas = [1]
        self.training_data = self.generate_fake_data(num_x=100, sigmas=sigmas)
        self.prediction_data = self.generate_fake_data(
            num_x=300, sigmas=sigmas
        )
        self.in_parameters = ["x", "sigma"]
        self.out_parameters = ["p"]
        self.prediction_data = self.prediction_data[self.in_parameters]

    def generate_fake_data(self, num_x, sigmas):
        """
        Lets simulate training posteriors_list using the following formula:
        pi(x) = TruncNorm(x | mu=0, sigma=1)
        Where (p) is the dependent variable you are trying to predict and (x) and (sigma) are the features
        :return:
        :rtype:
        """
        # Create fake posteriors_list
        seed(0)
        x_range = (-10, 10)
        x = uniform(*x_range, num_x)
        # Prep posteriors_list for training.
        df = pd.DataFrame({"x": [], "sigma": [], "p": []})
        for s in sigmas:
            p = truncnorm(xx=x, mu=1, sigma=s, high=x_range[1], low=x_range[0])
            xvals = x.copy()
            sig = [s] * len(xvals)
            assert len(sig) == len(p) == len(xvals)
            df = df.append(pd.DataFrame({"x": xvals, "sigma": sig, "p": p}))

        return df

    def visualise_training_data(self):
        unique_sigma = set(self.training_data.sigma.values)
        colors = get_colors(num_colors=len(unique_sigma))
        for i, s in enumerate(unique_sigma):
            df = self.training_data.copy()
            df = df[df["sigma"] == s]
            plt.scatter(
                df.x, df.p, label=f"sigma={s:.1f}", color=colors[i], alpha=0.4
            )
        plt.legend()
        plt.xlabel("X-vals")
        plt.ylabel("p(x|sigma)")
        plt.title("Training Data")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, "training_data.png"))
        plt.close("all")

    def visualise_predicted_data(self, predicted_vals, n_trees, label):
        df_train = self.training_data.copy()
        df_pred = self.prediction_data.copy()
        df_pred["p"] = predicted_vals
        unique_sigma = set(df_train.sigma.values)
        colors = get_colors(num_colors=len(unique_sigma))
        for i, s in enumerate(unique_sigma):
            df = df_train.copy()
            df = df[df["sigma"] == s]
            plt.scatter(
                df.x, df.p, label=f"s={s:.1f}", color=colors[i], alpha=0.4
            )
            df_p = df_pred.copy()
            df_p = df_p[df_p["sigma"] == s]
            plt.scatter(df_p.x, df_p.p, color=colors[i], marker="+")
        plt.legend()
        plt.xlabel("X-vals")
        plt.ylabel("p(x|sigma)")
        plt.title(f"{label}: #tree:{n_trees}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"{label}_predictions.png"))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
