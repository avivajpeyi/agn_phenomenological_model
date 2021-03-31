import logging
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from tensorflow.estimator import BoostedTreesRegressor

from .regressor import Regressor

logger = logging.getLogger()


class TensorflowRegressor(Regressor):
    """
    https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesRegressor
    """
    def __init__(self, input_parameters):
        super().__init__(input_parameters)
        self.training_tuning_parameters = dict(n_estimators=50, random_state=42)
        self.model = RandomForestRegressor(**self.training_tuning_parameters)

    def train(self, data: pd.DataFrame, training_hyper_parameters: Dict):
        train, test, train_labels, test_labels = self.train_test_split(data)
        self.model.fit(X=train, y=train_labels)
        logger.info("Training complete")
        self.test(test, test_labels)

    def test(self, data: pd.DataFrame, labels: pd.DataFrame):
        predicted_labels = self.model.predict(data)
        errors = abs(predicted_labels - labels)
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
        pass

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

