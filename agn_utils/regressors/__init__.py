"""
Module to help with regression
"""
from enum import Enum
from .scikit_regressor import ScikitRegressor



class AvailibleRegressors(Enum):
    SCIKIT = 0
    TF = 1
