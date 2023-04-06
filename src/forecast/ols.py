"""
Oridnary Least Squares method
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from src.forecast.naive import Forecast


class OLS(Forecast):
    model: LinearRegression
    forecast: np.ndarray
    n_jobs: int

    def __init__(self, n_jobs: int = 1) -> None:
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
        Fits model on provided train features and labels.]
        args:
            train_X : numpy array containing all features
            train_y : numpy array containing labels
        """

        self.model.fit(X=train_X, y=train_y)

        return self
