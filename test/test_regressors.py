import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import seed, uniform
from scipy.interpolate import griddata

from agn_utils.regressors.regressor import Regressor
from agn_utils.regressors.scikit_regressor import ScikitRegressor


class TestRegressors(unittest.TestCase):

    def setUp(self):
        self.outdir = "test"
        os.makedirs(self.outdir, exist_ok=True)
        self.training_data, self.prediction_data = self.generate_fake_data()

        self.in_parameters = ['x', 'y']
        self.out_parameters = ['z']
        self.visualise_training_data()

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def generate_fake_data(self):
        """
        Lets simulate training data using the following formula:
        pred_z = grid_x * np.exp(-grid_x ** 2 - grid_y ** 2)
        Where (pred_z) is the dependent variable you are trying to predict and (grid_x) and (grid_y) are the features
        :return:
        :rtype:
        """
        # Create fake data
        seed(0)
        npts = 5000
        x = uniform(-2, 2, npts)
        y = uniform(-2, 2, npts)
        z = x * np.exp(-x ** 2 - y ** 2)

        # Prep data for training.
        training_df = pd.DataFrame({'x': x, 'y': y, 'z': z})

        xi = np.linspace(-2.0, 2.0, 200),
        yi = np.linspace(-2.1, 2.1, 210),
        xi, yi = np.meshgrid(xi, yi)

        predict_df = pd.DataFrame({
            'x': xi.flatten(),
            'y': yi.flatten(),
        })
        return training_df, predict_df

    def visualise_training_data(self):
        x, y, z = self.training_data.x, self.training_data.y, self.training_data.z
        xy = np.zeros((2, np.size(x)))
        xy[0], xy[1] = x, y
        xy = xy.T
        grid_x, grid_y = self.prediction_data.x, self.prediction_data.y
        grid_z = griddata(points=xy, values=z, xi=(grid_x, grid_y), method='linear',
                          fill_value='0')
        self.plot_contour(grid_z)
        plt.scatter(self.training_data.x, self.training_data.y, marker='.')
        plt.title('Contour on training data')
        plt.savefig(os.path.join(self.outdir, "training_data.png"))

    def visualise_predicted_data(self, predicted_vals, n_trees, label):
        self.plot_contour(predicted_vals)
        plt.text(-1.8, 2.1, f'{label} predictions: # trees: {n_trees}', color='w',
                 backgroundcolor='black', size=20)
        plt.savefig(os.path.join(self.outdir, f"{label}_predictions.png"))

    def run_regressor_tests(self, my_regressor: Regressor):
        r = my_regressor(input_parameters=self.in_parameters,
                         output_parameters=self.out_parameters)
        r.train(data=self.training_data)
        r.test(data=self.training_data[self.in_parameters],
               labels=self.training_data[self.out_parameters])
        model_path = os.path.join(self.outdir, "scikit.model")
        r.save(model_path)
        r = my_regressor(input_parameters=self.in_parameters,
                         output_parameters=self.out_parameters)
        r.load(model_path)
        predicted_vals = r.predict(self.prediction_data)
        self.visualise_predicted_data(predicted_vals, n_trees=r.n_trees, label="scikit")
        self.assertIsNotNone(predicted_vals)

    def test_scikit_regressos(self):
        self.run_regressor_tests(ScikitRegressor)

    def plot_contour(self, pred_z):
        x, y, z = self.training_data.x, self.training_data.y, self.training_data.z
        xi = np.linspace(-2.0, 2.0, 200),
        yi = np.linspace(-2.1, 2.1, 210),
        xi, yi = np.meshgrid(xi, yi)
        xy = np.zeros((2, np.size(x)))
        xy[0], xy[1] = x, y
        xy = xy.T
        true_z = griddata(xy, z, (xi, yi), method='linear', fill_value='0')
        pred_z = pred_z.reshape(xi.shape)
        plt.figure(figsize=(10, 8))
        plt.contour(xi, yi, pred_z, 15, linewidths=0.5, colors='k')
        plt.contourf(xi, yi, pred_z, 15, vmax=abs(true_z).max(),
                     vmin=-abs(true_z).max(), cmap='RdBu_r')
        plt.colorbar(label="z(x,y)")  # Draw colorbar
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)


if __name__ == '__main__':
    unittest.main()
