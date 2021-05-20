"""
Module to help with regression
"""
from enum import Enum
from .scikit_regressor import ScikitRegressor
from .tf_regressor import TfRegressor


class AvailibleRegressors(Enum):
    SCIKIT = 0
    TF = 1
