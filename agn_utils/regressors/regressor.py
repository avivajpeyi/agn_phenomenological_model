# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from ..agn_logger import logger

RANDOM_STATE = 42


class Regressor(ABC):
    def __init__(self,
                 input_parameters: List[str], output_parameters: List[str],
                 outdir: str, model_hyper_param: Optional[Dict] = {}
                 ):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.savepath = os.path.join(self.outdir, "saved_model")
        self.input_parameters = input_parameters
        self.output_parameters = output_parameters
        self.model_hyper_param = model_hyper_param
        self.model = None

    def train_test_split(self, data: pd.DataFrame,
                         testing_frac: Optional[float] = 0.2):
        train, test = train_test_split(
            data, test_size=testing_frac,
            random_state=RANDOM_STATE, shuffle=True
        )
        train_labels = train[self.output_parameters]
        test_labels = test[self.output_parameters]
        train_data = train[self.input_parameters]
        test_data = test[self.input_parameters]
        return train_data, test_data, train_labels, test_labels

    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        training_info = (
            f"TRAINING INFO:\n"
            f"Commencing training for "
            f"{self.input_parameters}-->{self.output_parameters} "
            f"with hyper params: \n"
        )
        for k, v in self.model_hyper_param.items():
            training_info += f"\t {k} : {v}\n"
        labels = self.input_parameters + self.output_parameters
        training_info += (
            f"Training Data ({len(data)} rows):\n"
            f"{data[labels].describe().T[['min', 'mean', 'max']]}")
        logger.info(training_info)

    @abstractmethod
    def test(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def visualise(self):
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def n_trees(self) -> int:
        pass
