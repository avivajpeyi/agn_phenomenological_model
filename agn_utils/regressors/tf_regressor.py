import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .regressor import Regressor
from ..agn_logger import logger


class ScikitRegressor(Regressor):
    """
    https://www.tensorflow.org/tutorials/estimator/boosted_trees_model_understanding

    NOTE:
    ref the following do determine how to tune training hyper-params
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

    """

    def __init__(self, input_parameters: List[str], output_parameters: List[str],
                 model_hyper_param: Optional[Dict] = {}):
        super().__init__(input_parameters, output_parameters)
        self.model_hyper_param = dict(
            n_estimators=100, criterion='mse',
            max_depth=None, min_samples_split=2,
            min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None, bootstrap=True,
            oob_score=False, n_jobs=None, random_state=None,
            verbose=0, warm_start=False, ccp_alpha=0.0,
            max_samples=None)
        self.model_hyper_param.update(model_hyper_param)
        self.model = RandomForestRegressor(**self.model_hyper_param)

    def train(self, data: pd.DataFrame):
        train, test, train_labels, test_labels = self.train_test_split(data)
        self.model.fit(X=train, y=train_labels)
        logger.info("Training complete")
        self.test(test, test_labels)

    def test(self, data: pd.DataFrame, labels: pd.DataFrame):
        predicted_labels = self.model.predict(data)
        errors = abs(predicted_labels - labels.values)
        model_testing_data_mae = round(np.mean(errors), 2)
        model_testing_score = self.model.score(data, labels)
        logger.info(
            f'MODEL TESTING: '
            f'Score={model_testing_score * 100:.2f}%, '
            f'Mean Abosulte Error={model_testing_data_mae}'
        )

    def save(self, filename: str):
        joblib.dump(self.model, filename)

    def load(self, filename: str):
        self.model = joblib.load(filename)

    def visualise(self):
        "https://mljar.com/blog/visualize-tree-from-random-forest/"
        raise NotImplementedError()

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    @property
    def n_trees(self) -> int:
        return len(self.model.estimators_)