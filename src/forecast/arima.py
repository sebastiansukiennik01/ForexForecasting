"""
Autoregressive Integrated Moving Average method
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from statsmodels.tsa.arima import model

from src.forecast.naive import Forecast


class ARIMA(Forecast):
    model: model.ARIMA
    model_fit: model.ARIMAResultsWrapper
    forecast: np.ndarray

    def __init__(self) -> None:
        ...

    def predict(self, test_y: np.ndarray, **kwargs) -> np.array:
        """
        Forecasts values for provided set of feautres.
        args:
            test_X : np.ndarray of features
        returns : forecasted values
        """
        h = len(test_y)
        endog_len = self.model.endog.shape[0]

        try:
            self.forecast = self.model_fit.predict(
                start=endog_len, end=endog_len + h - 1
            ).values.flatten()
        except AttributeError:
            self.forecast = self.model_fit.predict(
                start=endog_len, end=endog_len + h - 1
            )

        return self.forecast

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> ARIMA:
        """
        Fits model on provided train features and labels.
        args:
            train_X : for compatibility
            train_y : numpy array containing labels
            kwargs : ARIMA parameters
        """
        p = kwargs.pop("p", 1)
        d = kwargs.pop("d", 1)
        q = kwargs.pop("q", 1)

        self.model = model.ARIMA(train_y, order=(p, d, q))
        self.model_fit = self.model.fit()

        return self
