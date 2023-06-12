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
        self.p = kwargs.pop("p", 1)
        self.d = kwargs.pop("d", 1)
        self.q = kwargs.pop("q", 1)

        self.model = model.ARIMA(train_y, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()

        return self
    
    def _fit_predict(self, train_y: np.ndarray, h: int = 200) -> None:
        """
        Fits and predicts to get residualals
        """
        _forecasts = list(train_y.iloc[:h].values.flatten())
        for i in range(h, train_y.shape[0], h):
            _model = model.ARIMA(train_y.iloc[:i], order=(self.p, self.d, self.q))
            model_fitted = _model.fit()
            
            print(train_y.iloc[:i].shape)
            test_y = train_y.iloc[i : i+h]
            h = len(test_y)
            endog_len = _model.endog.shape[0]

            try:
                forecast = model_fitted.predict(
                    start=endog_len, end=endog_len + h - 1
                ).values.flatten()
            except AttributeError:
                forecast = model_fitted.predict(
                    start=endog_len, end=endog_len + h - 1
                )
            _forecasts.extend(forecast)

        return _forecasts

    
    def get_residuals(self, train_y: np.ndarray, **kwargs):
        
        self.residuals = self.model_fit.resid
        # self.pred_insample = self.model_fit.predict()
        
        self.pred_insample = self._fit_predict(train_y=train_y) #todo neeeeeew
        
        self.pred_insample[0], self.residuals.iat[0] =  self.residuals.values[0], self.pred_insample[0] 
        
        return self.model_fit.resid
