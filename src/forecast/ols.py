"""
Oridnary Least Squares method
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS as _OLS
import statsmodels.api as sm

from src.forecast.naive import Forecast


class OLS(Forecast):
    model: LinearRegression
    forecast: np.ndarray
    n_jobs: int
    r_2: float

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = n_jobs
        self.model = LinearRegression(n_jobs=n_jobs)

    def predict(self, test_X: np.ndarray) -> np.array:
        """
        Forecasts values for provided set of feautres.
        args:
            test_X : np.ndarray of features
        returns : forecasted values
        """
        # self.h = test_X.shape[0]
        self.forecast = self.model.predict(test_X)

        return self.forecast.flatten()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> OLS:
        """
        Fits model on provided train features and labels.
        args:
            train_X : numpy array containing all features
            train_y : numpy array containing labels
        """

        self.model.fit(X=train_X, y=train_y)
        self.r_2 = self.model.score(train_X, train_y)

        return self

    def summary(self) -> str:
        const = self.model.intercept_
        coef = self.model.coef_

        return f"const: {const}\n coef: {coef}\n R-score: {self.r_2}"


class OLSS(Forecast):
    """
    Ordinary Least Squares using stastmodel library
    """

    model: _OLS
    forecast: np.ndarray
    n_jobs: int

    def __init__(self) -> None:
        ...

    def predict(self, test_X: np.ndarray) -> np.array:
        """
        Forecasts values for provided set of feautres.
        args:
            test_X : np.ndarray of features
        returns : forecasted values
        """
        # self.h = test_X.shape[0]
        self.forecast = self.model.predict(test_X)

        return self.forecast.flatten()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> OLS:
        """
        Fits model on provided train features and labels.]
        args:
            train_X : numpy array containing all features
            train_y : numpy array containing labels
        """
        train_X = sm.add_constant(train_X)
        self.model = _OLS(train_X, train_y)
        self.model.fit()

        return self
