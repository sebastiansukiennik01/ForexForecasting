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
    
    def get_residuals(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs):
        """ Calculates insample residuals """
        self.h = self.h if hasattr(self, 'h') else 5
        first_idx = self.k if hasattr(self, 'k') and self.k else 5
        
        train_X = train_X.values
        train_y = train_y.values
        pred_y = list(train_y[:first_idx].flatten())
        
        [
            pred_y.extend(self._fit_predict(train_X, train_y, i))
            for i in range(first_idx, train_y.shape[0], self.h)
        ]
        
        # calc resid and insample prediction
        self.pred_insample = pred_y[:len(train_y)]
        self.residuals = np.subtract(train_y.flatten(), self.pred_insample)

    def _fit_predict(self, train_X: np.ndarray, train_y: np.ndarray, i: int) -> np.ndarray:
        temp_train_X = train_X[:i]
        temp_train_y = train_y[:i]
        temp_test_X = train_X[i:i+self.h]
        print(temp_train_y.shape)
        
        self.fit(train_X=temp_train_X, train_y=temp_train_y)
        
        return self.predict(temp_test_X)

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
